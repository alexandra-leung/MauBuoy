# backend/train_prediction_model.py
"""
ReefGuardian AI - Bleaching Risk Model
Trains Random Forest to predict binary bleaching risk using the dataset's target column.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

print("="*70)
print("REEFGUARDIAN AI - 4-Class Model Training")
print("="*70)

# ============================================================
# STEP 1: LOAD DATASET
# ============================================================
print(f"\nSTEP 1: Loading dataset...")

dataset_path = 'data/synthetic_bleaching_dataset.csv'
if not os.path.exists(dataset_path):
    print(f"File not found: {dataset_path}")
    exit(1)

df = pd.read_csv(dataset_path)
print(f"Loaded {len(df)} records")

# ============================================================
# STEP 2: USING BINARY BLEACHING TARGET
# ============================================================
print(f"\nSTEP 2: Using binary bleaching target...")

y = df["Bleaching_Label"]

print("Class distribution:")
print(y.value_counts())

# ============================================================
# STEP 3: PREPARE FEATURES AND TARGET
# ============================================================
print(f"\nSTEP 3: Preparing features...")

# Feature columns
feature_cols = [
    "Latitude",
    "Longitude",
    "Depth_m",
    "Year",
    "Sea_Surface_Temperature_C",
    "Degree_Heating_Weeks"
]

X = df[feature_cols].copy()

# Drop any remaining missing values
mask = X.notna().all(axis=1)
X = X[mask]
y = y[mask]

# Drop any remaining missing values
mask = X.notna().all(axis=1)
X = X[mask]
y = y[mask]

print(f"   Features: {feature_cols}")
print(f"   Final dataset: {len(X)} records")

# ============================================================
# STEP 4: TRAIN/VAL/TEST SPLIT (70/15/15)
# ============================================================
print(f"\nSTEP 5: Splitting data (70/15/15)...")

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"   Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"   Val:   {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
print(f"   Test:  {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

# ============================================================
# STEP 6: TRAIN RANDOM FOREST (BINARY BLEACHING RISK)
# ============================================================
print(f"\nSTEP 6: Training binary Random Forest for bleaching risk...")

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ============================================================
# STEP 7: EVALUATE
# ============================================================
print(f"\nSTEP 7: Evaluation")
print("="*70)

for name, X_set, y_set in [('Train', X_train, y_train), 
                             ('Val', X_val, y_val), 
                             ('Test', X_test, y_test)]:
    y_pred = model.predict(X_set)
    acc = accuracy_score(y_set, y_pred)
    print(f"{name} Accuracy: {acc:.2%}")

print(f"\nTest Set Classification Report:")
print(classification_report(
    y_test,
    model.predict(X_test),
    target_names=[
        "No Bleaching",
        "Bleaching"
    ]
))

# Feature importance
print("\nFeature Importance:")
for col, imp in sorted(zip(feature_cols, model.feature_importances_), 
                        key=lambda x: x[1], reverse=True):
    print(f"   {col:<25} => {imp:.3f}")

# ============================================================
# STEP 8: SAVE MODEL + METADATA
# ============================================================
print(f"\nSTEP 8: Saving...")

os.makedirs('backend', exist_ok=True)
joblib.dump(model, 'backend/bleaching_predictor.pkl')
print(f"Model saved: backend/bleaching_predictor.pkl")

metadata = {
    "feature_columns": feature_cols,
    "classes": ["No Bleaching", "Bleaching"]
}
joblib.dump(metadata, 'backend/model_metadata.pkl')
print(f"Metadata saved: backend/model_metadata.pkl")

# ============================================================
# STEP 9: DEMO PREDICTION (BLEACHING PROBABILITY)
# ============================================================
print(f"\nSTEP 9: Demo prediction (4-class output)...")

sample_high = pd.DataFrame([{
    "Latitude": -20.4,
    "Longitude": 57.7,
    "Depth_m": 5.0,
    "Year": 2024,
    "Sea_Surface_Temperature_C": 303.2,
    "Degree_Heating_Weeks": 8.5
}])

prob_high = model.predict_proba(sample_high)[0][1]
print(f"\nHigh-risk sample bleaching probability: {prob_high:.1%}")

sample_low = pd.DataFrame([{
    "Latitude": -20.46,
    "Longitude": 57.32,
    "Depth_m": 18.0,
    "Year": 2024,
    "Sea_Surface_Temperature_C": 299.2,
    "Degree_Heating_Weeks": 0.5
}])

prob_low = model.predict_proba(sample_low)[0][1]
print(f"Low-risk sample bleaching probability: {prob_low:.1%}")

print("\nTraining complete! Run: python backend/predict_risk.py")
