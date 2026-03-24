from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
import sys
import math
from geopy.geocoders import Nominatim

# Ensure backend modules are discoverable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your data source modules
try:
    from backend.data_sources.bhuvan_api import bhuvan_api
    from backend.data_sources.ksdma_zones import ksdma_zones
    from backend.data_sources.osm_processor import osm_processor
    from backend.data_sources.imd_api import imd_api
    from backend.data_sources.sentinel_processor import sentinel_processor
except ImportError as e:
    print(f"⚠️ Warning: Some data sources could not be imported. Ensure folder structure is correct. Error: {e}")

app = Flask(__name__)
CORS(app)

# Initialize Geocoder for manual coordinate identification
geolocator = Nominatim(user_agent="safeland_explorer")

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "ml", "flood_risk_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "..", "ml", "label_encoder.pkl")

# --- LOAD MODEL ---
if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    print("✅ Model and Encoder loaded successfully.")
    
    if hasattr(model, "feature_names_in_"):
        FEATURE_COLUMNS = list(model.feature_names_in_)
    else:
        FEATURE_COLUMNS = [
            'latitude', 'longitude', 'flooded_2018', 'flooded_2019', 'flooded_2021',
            'flood_history_count', 'ksdma_zone', 'elevation', 'slope',
            'river_distance', 'drainage_density', 'annual_rainfall_mm', 'extreme_rain_events'
        ]
else:
    model = None
    encoder = None
    FEATURE_COLUMNS = []
    print("⚠️ Model files not found. Prediction will fail until model is trained.")

def calculate_slope(lat, lon, offset=0.005):
    """Calculate slope using 4-point elevation sampling"""
    try:
        n_elev = bhuvan_api.get_elevation(lat + offset, lon)
        s_elev = bhuvan_api.get_elevation(lat - offset, lon)
        e_elev = bhuvan_api.get_elevation(lat, lon + offset)
        w_elev = bhuvan_api.get_elevation(lat, lon - offset)
        
        dist_m = offset * 111000
        ns_grad = abs(n_elev - s_elev) / (2 * dist_m)
        ew_grad = abs(e_elev - w_elev) / (2 * dist_m)
        slope_deg = math.degrees(math.atan(max(ns_grad, ew_grad)))
        return round(slope_deg, 2)
    except Exception:
        return 0.0

# --- API ENDPOINTS ---

@app.route("/predict", methods=["POST"])
def predict():
    if not model:
        return jsonify({"error": "Model not loaded on server"}), 500

    data = request.json
    lat = data.get("latitude")
    lon = data.get("longitude")

    if lat is None or lon is None:
        return jsonify({"error": "Latitude and longitude are required"}), 400

    # 1. Reverse Geocode to find place name
    place_name = "Kerala Region"
    try:
        location = geolocator.reverse(f"{lat}, {lon}", timeout=5)
        if location:
            addr = location.raw.get('address', {})
            place_name = addr.get('city') or addr.get('town') or addr.get('village') or addr.get('county', "Kerala")
    except Exception:
        pass

    print(f"📍 Processing Prediction for: {place_name} ({lat}, {lon})")

    # 2. Fetch environmental data from your integrated APIs
    elevation = bhuvan_api.get_elevation(lat, lon)
    slope = calculate_slope(lat, lon)
    k_zone = ksdma_zones.get_vulnerability_zone(lat, lon)
    river_dist = osm_processor.get_nearest_river_distance(lat, lon)
    drainage = osm_processor.get_drainage_density(lat, lon)
    ann_rainfall = imd_api.get_annual_rainfall(lat, lon)
    ext_rain = imd_api.get_extreme_rainfall_events(lat, lon)

    # 3. Get flood history (Fall back to Sentinel if not provided in JSON)
    sentinel_info = sentinel_processor.get_flood_events_detail(lat, lon)
    f18 = int(data.get("flooded_2018", sentinel_info["flooded_2018"]))
    f19 = int(data.get("flooded_2019", sentinel_info["flooded_2019"]))
    f21 = int(data.get("flooded_2021", sentinel_info["flooded_2021"]))
    flood_count = f18 + f19 + f21

    # 4. Prepare Feature Dataframe
    feature_dict = {
        "latitude": lat, "longitude": lon,
        "flooded_2018": f18, "flooded_2019": f19, "flooded_2021": f21,
        "flood_history_count": flood_count, "ksdma_zone": k_zone,
        "elevation": elevation, "slope": slope, "river_distance": river_dist,
        "drainage_density": drainage, "annual_rainfall_mm": ann_rainfall,
        "extreme_rain_events": ext_rain
    }

    df = pd.DataFrame([feature_dict])[FEATURE_COLUMNS]

    # 5. Execute Prediction
    prediction_idx = model.predict(df)[0]
    risk_label = encoder.inverse_transform([prediction_idx])[0]
    
    probabilities = model.predict_proba(df)[0]
    prob_dict = {encoder.classes_[i]: round(float(probabilities[i]) * 100, 1) 
                 for i in range(len(encoder.classes_))}

    # 6. Response for Frontend
    return jsonify({
        "location": {
            "latitude": lat, 
            "longitude": lon,
            "place_name": place_name
        },
        "environmental_data": {
            "rainfall": ann_rainfall,
            "elevation": elevation,
            "soil_moisture": round(drainage * 100, 2),
            "water_level": ext_rain,
            "river_distance": round(river_dist, 2)
        },
        "historical_floods": {
            "2018": bool(f18), "2019": bool(f19), "2021": bool(f21),
            "total_count": flood_count
        },
        "flood_risk": risk_label,
        "confidence": prob_dict
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "SafeLand Backend Active",
        "model_ready": model is not None,
        "device": "CPU"
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)