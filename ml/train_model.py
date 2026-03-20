"""
Train SafeLand flood risk model on enhanced dataset.

Features (11 total):
  Flood history  : latitude, longitude, flooded_2018, flooded_2019,
                   flooded_2021, flood_history_count
  Environmental  : ksdma_zone, elevation, slope,
                   river_distance, drainage_density
  (rainfall to be added later once IMD data is available)

Usage:
    python ml/train_model.py
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import numpy as np

# ── Paths ───────────────────────────────────────────────────────────
ENHANCED_DATA = os.path.join("data", "enhanced_training_data.csv")
BASELINE_DATA = os.path.join("data", "balanced_training_data.csv")
MODEL_PATH    = os.path.join("ml", "flood_risk_model.pkl")
ENCODER_PATH  = os.path.join("ml", "label_encoder.pkl")

# ── Features ────────────────────────────────────────────────────────
BASE_FEATURES = [
    'latitude', 'longitude',
    'flooded_2018', 'flooded_2019', 'flooded_2021',
    'flood_history_count'
]
ENHANCED_FEATURES = BASE_FEATURES + [
    'ksdma_zone', 'elevation', 'slope',
    'river_distance', 'drainage_density',
    'annual_rainfall_mm', 'extreme_rain_events',   # added after rainfall download
]

# Drop any enhanced features not yet present in the CSV (graceful degradation)
def _available_features(df, cols):
    return [c for c in cols if c in df.columns]

print("=" * 60)
print("TRAINING FLOOD RISK MODEL")
print("=" * 60)

# ── Load dataset (enhanced preferred, baseline fallback) ─────────────
if os.path.exists(ENHANCED_DATA):
    DATA_PATH       = ENHANCED_DATA
    feature_columns = ENHANCED_FEATURES
    print(f"\n✓ Using ENHANCED dataset: {DATA_PATH}")
else:
    DATA_PATH       = BASELINE_DATA
    feature_columns = BASE_FEATURES
    print(f"\n⚠  Enhanced data not found. Using baseline: {BASELINE_DATA}")
    print(f"   Run scripts/enrich_with_indian_sources.py to generate it.")

df = pd.read_csv(DATA_PATH)
feature_columns = _available_features(df, feature_columns)   # skip missing cols
print(f"  Rows     : {len(df):,}")
print(f"  Columns  : {list(df.columns)}")

# Drop any rows with NaN in feature columns
df = df.dropna(subset=feature_columns + ['risk'])
print(f"  After NA drop: {len(df):,} rows")

# ── Features & target ───────────────────────────────────────────────
X = df[feature_columns]
y = df["risk"]

print(f"\nRisk distribution:\n{y.value_counts()}")
print(f"\nFeatures ({len(feature_columns)}):")
for f in feature_columns:
    print(f"  - {f}")

# ── Label encoding ──────────────────────────────────────────────────
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f"\nClasses: {list(label_encoder.classes_)}")

# ── Train / test split ──────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
    # Note: no stratify — only 1 "High" risk sample in dataset
)
print(f"\nSplit: {len(X_train):,} train / {len(X_test):,} test")

# ── Train model ─────────────────────────────────────────────────────
print("\nTraining Random Forest...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("✓ Training complete")

# ── Evaluate ─────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
train_acc = model.score(X_train, y_train)
test_acc  = accuracy_score(y_test, y_pred)

print(f"\n{'='*60}")
print("EVALUATION")
print(f"{'='*60}")
print(f"\n  Train accuracy : {train_acc*100:.2f}%")
print(f"  Test accuracy  : {test_acc*100:.2f}%")

unique_labels      = sorted(list(set(y_test) | set(y_pred)))
target_names_filt  = [label_encoder.classes_[i] for i in unique_labels]
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
                            labels=unique_labels,
                            target_names=target_names_filt,
                            zero_division=0))

# Confusion matrix
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
header = "".join(f"{label_encoder.classes_[i]:>10}" for i in unique_labels)
print(f"{'':>10}{header}")
for i, li in enumerate(unique_labels):
    row = "".join(f"{cm[i][j]:>10}" for j in range(len(unique_labels)))
    print(f"{label_encoder.classes_[li]:>10}{row}")

# ── Feature importance ───────────────────────────────────────────────
print("\nFeature Importance:")
fi = pd.DataFrame({
    'feature':    feature_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
for _, r in fi.iterrows():
    bar = "█" * int(r['importance'] * 50)
    print(f"  {r['feature']:22s}: {r['importance']:.4f}  {bar}")

# ── Save ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("SAVING MODEL")
print(f"{'='*60}")
joblib.dump(model, MODEL_PATH)
joblib.dump(label_encoder, ENCODER_PATH)
print(f"✓ Model   → {MODEL_PATH}")
print(f"✓ Encoder → {ENCODER_PATH}")

print(f"\n{'='*60}")
print("DONE! Run ml/test_model.py to validate predictions.")
print(f"{'='*60}")
