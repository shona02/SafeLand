from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our backend data sources
from backend.data_sources.bhuvan_api import bhuvan_api
from backend.data_sources.ksdma_zones import ksdma_zones
from backend.data_sources.osm_processor import osm_processor
from backend.data_sources.imd_api import imd_api
from backend.data_sources.sentinel_processor import sentinel_processor

app = Flask(__name__)
CORS(app)

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "ml", "flood_risk_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "..", "ml", "label_encoder.pkl")
TRAIN_DATA_PATH = os.path.join(BASE_DIR, "..", "data", "enhanced_training_data.csv")

# --- LOAD MODEL ---
if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    print("✅ Model loaded successfully.")
    
    # Pre-fetch the exact model expected columns
    if hasattr(model, "feature_names_in_"):
        FEATURE_COLUMNS = list(model.feature_names_in_)
    else:
        # Fallback to the known 13 columns
        FEATURE_COLUMNS = [
            'latitude', 'longitude', 'flooded_2018', 'flooded_2019', 'flooded_2021',
            'flood_history_count', 'ksdma_zone', 'elevation', 'slope',
            'river_distance', 'drainage_density', 'annual_rainfall_mm', 'extreme_rain_events'
        ]
else:
    model = None
    encoder = None
    FEATURE_COLUMNS = []
    print("⚠️ Model files not found. Please train the model first.")

def calculate_slope(lat, lon, offset=0.005):
    """Calculate slope using 4-point elevation sampling (similar to training data)"""
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
    except Exception as e:
        print(f"Error calculating slope: {e}")
        return 0.0

# --- API ENDPOINT ---
@app.route("/predict", methods=["POST"])
def predict():
    if not model:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.json
    lat = data.get("latitude")
    lon = data.get("longitude")

    if lat is None or lon is None:
        return jsonify({"error": "latitude and longitude are required"}), 400

    print(f"\n📍 Request received for: {lat}, {lon}")

    # 1. Fetch environmental data
    print("  Fetching environmental features...")
    elevation = bhuvan_api.get_elevation(lat, lon)
    slope = calculate_slope(lat, lon)
    ksdma_zone = ksdma_zones.get_vulnerability_zone(lat, lon)
    river_distance = osm_processor.get_nearest_river_distance(lat, lon)
    drainage_density = osm_processor.get_drainage_density(lat, lon)
    
    # Note: IMD is rate limited by open-meteo proxy so this might be slow for real-time traffic
    # but it's fine for single testing queries.
    annual_rainfall = imd_api.get_annual_rainfall(lat, lon)
    extreme_rain = imd_api.get_extreme_rainfall_events(lat, lon)

    # 2. Get flood history
    # The frontend can provide explicit history, otherwise fall back to sentinel inference
    flooded_2018 = int(data.get("flooded_2018", sentinel_processor.get_flood_events_detail(lat, lon)["flooded_2018"]))
    flooded_2019 = int(data.get("flooded_2019", sentinel_processor.get_flood_events_detail(lat, lon)["flooded_2019"]))
    flooded_2021 = int(data.get("flooded_2021", sentinel_processor.get_flood_events_detail(lat, lon)["flooded_2021"]))
    flood_count = flooded_2018 + flooded_2019 + flooded_2021

    # 3. Assemble features exactly as the Random Forest model expects them
    feature_dict = {
        "latitude": lat,
        "longitude": lon,
        "flooded_2018": flooded_2018,
        "flooded_2019": flooded_2019,
        "flooded_2021": flooded_2021,
        "flood_history_count": flood_count,
        "ksdma_zone": ksdma_zone,
        "elevation": elevation,
        "slope": slope,
        "river_distance": river_distance,
        "drainage_density": drainage_density,
        "annual_rainfall_mm": annual_rainfall,
        "extreme_rain_events": extreme_rain
    }

    # Extract strictly only the columns the model was trained on
    df = pd.DataFrame([feature_dict])[FEATURE_COLUMNS]

    # 4. Predict
    prediction_idx = model.predict(df)[0]
    risk_label = encoder.inverse_transform([prediction_idx])[0]
    
    # Get probabilities
    probabilities = model.predict_proba(df)[0]
    prob_dict = {encoder.classes_[i]: round(float(probabilities[i]) * 100, 1) for i in range(len(encoder.classes_))}

    print(f"  Prediction: {risk_label} {prob_dict}")

    return jsonify({
        "location": {"latitude": lat, "longitude": lon},
        "environmental_data": {
            "elevation": elevation,
            "slope": slope,
            "ksdma_zone": ksdma_zone,
            "river_distance_km": round(river_distance, 2),
            "drainage_density": round(drainage_density, 3),
            "annual_rainfall_mm": annual_rainfall,
            "extreme_rain_events": extreme_rain
        },
        "historical_floods": {
            "2018": bool(flooded_2018),
            "2019": bool(flooded_2019),
            "2021": bool(flooded_2021),
            "total_count": flood_count
        },
        "flood_risk": risk_label,
        "confidence": prob_dict
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "Backend is running",
        "model_loaded": model is not None,
        "features_expected": FEATURE_COLUMNS
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)