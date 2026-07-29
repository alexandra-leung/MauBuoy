# backend/train_prediction_model.py
"""
ReefGuardian AI - Model Trainer
Trains Random Forest on the combined (balanced) dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

print("="*70)
print("🌊 REEFGUARDIAN AI - Model Training")
print("="*70)

# ============================================================
# STEP 1: LOAD COMBINED DATASET
# ============================================================
print("\n📂 STEP 1: Loading combined dataset...")

combined_path = 'data/combined_dataset.csv'
if not os.path.exists(combined_path):
    print(f"❌ File not found: {combined_path}")
    print("\nRun these scripts first:")
    print("  1. python backend/create_synthetic_data.py")
    print("  2. python backend/combine_datasets.py")
    exit(1)

df = pd.read_csv(combined_path)
print(f"✅ Loaded {len(df)} records")

# Separate features and target
feature_cols = [col for col in df.columns if col != 'target']
X = df[feature_cols]
y = df['target']

print(f"   Features: {feature_cols}")
print(f"   Target distribution:")
print(f"      Bleaching (1): {y.sum()} ({y.mean()*100:.1f}%)")
print(f"      No event (0): {(y==0).sum()} ({(1-y.mean())*100:.1f}%)")

# ============================================================
# STEP 2: TRAIN/VAL/TEST SPLIT (70/15/15)
# ============================================================
print(f"\n🔀 STEP 2: Splitting data (70/15/15)...")

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
# STEP 3: TRAIN RANDOM FOREST
# ============================================================
print(f"\n🧠 STEP 3: Training Random Forest...")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ============================================================
# STEP 4: EVALUATE
# ============================================================
print(f"\n📊 STEP 4: Evaluation")
print("="*70)

for name, X_set, y_set in [('Train', X_train, y_train), 
                             ('Val', X_val, y_val), 
                             ('Test', X_test, y_test)]:
    y_pred = model.predict(X_set)
    acc = accuracy_score(y_set, y_pred)
    print(f"{name} Accuracy: {acc:.2%}")

print(f"\nTest Set Classification Report:")
print(classification_report(y_test, model.predict(X_test), 
                            target_names=['No Event', 'Bleaching Event'],
                            zero_division=0))

# Feature importance
print("\n🔍 Feature Importance:")
for col, imp in sorted(zip(feature_cols, model.feature_importances_), 
                        key=lambda x: x[1], reverse=True):
    print(f"   {col:<25} → {imp:.3f}")

# ============================================================
# STEP 5: SAVE MODEL + METADATA
# ============================================================
print(f"\n💾 STEP 5: Saving...")

os.makedirs('backend', exist_ok=True)
joblib.dump(model, 'backend/bleaching_predictor.pkl')
print(f"✅ Model saved: backend/bleaching_predictor.pkl")

metadata = {
    'feature_columns': feature_cols,
    'classes': ['no_event', 'bleaching_event']
}
joblib.dump(metadata, 'backend/model_metadata.pkl')
print(f"✅ Metadata saved: backend/model_metadata.pkl")

# ============================================================
# STEP 6: DEMO PREDICTION
# ============================================================
print(f"\n🎯 STEP 6: Demo prediction...")

sample_high = pd.DataFrame([{
    feature_cols[0]: -20.4,   # Blue Bay
    feature_cols[1]: 57.7,
    feature_cols[2]: 5.0,
    feature_cols[3]: 2024,
    feature_cols[4]: 303.5,   # High SST (30.35°C)
    feature_cols[5]: 8.5,     # High DHW
}])

probs_high = model.predict_proba(sample_high)[0]
event_prob_high = probs_high[1] if len(probs_high) > 1 else probs_high[0]
risk_score_high = int(event_prob_high * 100)

print(f"\n📍 Blue Bay (SST=303.5K, DHW=8.5):")
print(f"   Bleaching Risk Score: {risk_score_high}/100")
print(f"   Disease Probability: {min(95, risk_score_high + 15)}%")
print(f"   Recovery Potential: {max(5, 100 - risk_score_high)}%")

sample_low = pd.DataFrame([{
    feature_cols[0]: -20.46,  # Le Morne
    feature_cols[1]: 57.32,
    feature_cols[2]: 5.0,
    feature_cols[3]: 2024,
    feature_cols[4]: 300.7,   # Low SST (27.55°C)
    feature_cols[5]: 1.2,     # Low DHW
}])

probs_low = model.predict_proba(sample_low)[0]
event_prob_low = probs_low[1] if len(probs_low) > 1 else probs_low[0]
risk_score_low = int(event_prob_low * 100)

print(f"\n📍 Le Morne (SST=300.7K, DHW=1.2):")
print(f"   Bleaching Risk Score: {risk_score_low}/100")
print(f"   Disease Probability: {min(95, risk_score_low + 15)}%")
print(f"   Recovery Potential: {max(5, 100 - risk_score_low)}%")

print("\n🎉 Training complete! Run: python backend/predict_risk.py")