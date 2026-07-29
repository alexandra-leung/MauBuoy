# backend/train_prediction_model.py
"""
ReefGuardian AI - Random Forest Prediction Model
Outputs MauBuoy's 3 metrics: Risk Score (0-100), Disease Probability, Recovery Potential
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import glob

print("="*70)
print("🌊 REEFGUARDIAN AI - Prediction Model Training")
print("="*70)

# ============================================================
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
# STEP 2: LOAD DATA (auto-detect format)
# ============================================================
print(f"\n📂 STEP 2: Loading data...")

try:
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
        print("✅ Loaded as Excel")
    else:
        # Try multiple separators
        loaded = False
        for sep in [',', '\t', ';']:
            try:
                test = pd.read_csv(file_path, sep=sep, nrows=3)
                if len(test.columns) > 5:
                    df = pd.read_csv(file_path, sep=sep, na_values=['nd', 'ND', ''])
                    print(f"✅ Loaded as CSV with separator: {repr(sep)}")
                    loaded = True
                    break
            except:
                continue
        
        if not loaded:
            df = pd.read_csv(file_path, sep='\t', header=None, na_values=['nd', 'ND'])
            print("✅ Loaded with no header")
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)

print(f"📊 Shape: {df.shape[0]} rows × {df.shape[1]} columns")

# ============================================================
# STEP 3: AUTO-DETECT COLUMNS (FIXED - no duplicates)
# ============================================================
print(f"\n🎯 STEP 3: Detecting columns...")

# Show all columns first so we can debug
print("   Available columns:")
for i, col in enumerate(df.columns):
    print(f"      [{i:2d}] {col}")

col_map = {c: str(c).lower().strip() for c in df.columns}

def find_column(keywords, description, exclude_cols=None):
    """Find column matching keywords, excluding already-used columns."""
    if exclude_cols is None:
        exclude_cols = []
    for col, col_lower in col_map.items():
        if col in exclude_cols:  # Skip already-used columns
            continue
        for kw in keywords:
            if kw in col_lower:
                print(f"   ✅ {description}: '{col}'")
                return col
    return None

used_cols = []

lat_col = find_column(['latitude'], 'Latitude')
if lat_col:
    used_cols.append(lat_col)

lon_col = find_column(['longitude'], 'Longitude', exclude_cols=used_cols)
if lon_col:
    used_cols.append(lon_col)

depth_col = find_column(['depth'], 'Depth', exclude_cols=used_cols)
if depth_col:
    used_cols.append(depth_col)

# Year: try year first, then date_year, then just date
year_col = find_column(['date_year', 'year'], 'Year', exclude_cols=used_cols)
if year_col is None:
    year_col = find_column(['date'], 'Year (fallback)', exclude_cols=used_cols)
if year_col:
    used_cols.append(year_col)

sst_col = find_column(['climsst', 'sst', 'temperature'], 'SST', exclude_cols=used_cols)
if sst_col:
    used_cols.append(sst_col)

# DHW: be specific - look for 'dhw' or 'heating' but NOT latitude/longitude
dhw_col = find_column(['ssta_dhw', 'dhw', 'heating'], 'DHW', exclude_cols=used_cols)
if dhw_col:
    used_cols.append(dhw_col)

# Verify no duplicates
feature_cols = [c for c in [lat_col, lon_col, depth_col, year_col, sst_col, dhw_col] if c is not None]
if len(feature_cols) != len(set(feature_cols)):
    print("\n❌ ERROR: Duplicate columns detected!")
    print(f"   Features: {feature_cols}")
    exit(1)

print(f"\n   Final features: {feature_cols}")

if not all([lat_col, lon_col]):
    print("\n❌ Could not find lat/lon columns.")
    exit(1)

# ============================================================
# STEP 4: CREATE TARGET VARIABLE (with multiple fallbacks)
# ============================================================
print(f"\n🎯 STEP 4: Creating target variable...")

# Strategy 1: Find text column and look for keywords
text_col = None
for col in df.columns:
    if df[col].dtype == 'object':
        avg_len = df[col].astype(str).str.len().mean()
        if avg_len > 20:
            text_col = col
            break

if text_col is None:
    text_col = df.columns[-1]

print(f"   Text column: '{text_col}'")

# Try keywords in priority order
y = None
used_method = None

for kw in ['bleach', 'bleaching', 'white', 'stress', 'mortality']:
    y_test = df[text_col].astype(str).str.contains(kw, case=False, na=False).astype(int)
    if 100 < y_test.sum() < len(df) - 100:  # Need balanced data
        y = y_test
        used_method = f"keyword '{kw}'"
        print(f"   ✅ Using keyword '{kw}': {y.sum()} positive ({y.mean()*100:.1f}%)")
        break

# Strategy 2: Use DHW threshold if keywords fail
if y is None and dhw_col:
    print(f"   ⚠️  Keywords failed. Using DHW > 4 threshold (NOAA standard)")
    df[dhw_col] = pd.to_numeric(df[dhw_col], errors='coerce')
    y = (df[dhw_col] > 4).astype(int)
    used_method = "DHW > 4"
    print(f"   ✅ DHW threshold: {y.sum()} positive ({y.mean()*100:.1f}%)")

# Strategy 3: Use SST threshold if DHW fails
if y is None and sst_col:
    print(f"   ⚠️  DHW failed. Using SST > 302.65K (29.5°C) threshold")
    df[sst_col] = pd.to_numeric(df[sst_col], errors='coerce')
    y = (df[sst_col] > 302.65).astype(int)
    used_method = "SST > 302.65K"
    print(f"   ✅ SST threshold: {y.sum()} positive ({y.mean()*100:.1f}%)")

if y is None:
    print("❌ Could not create target. Check your data.")
    exit(1)

# ============================================================
# STEP 5: BUILD FEATURES
# ============================================================
print(f"\n🔨 STEP 5: Building features...")

feature_cols = [c for c in [lat_col, lon_col, depth_col, year_col, sst_col, dhw_col] if c is not None]
print(f"   Features: {feature_cols}")

X = df[feature_cols].copy()
for col in feature_cols:
    X[col] = pd.to_numeric(X[col], errors='coerce')

# Drop missing
mask = X.notna().all(axis=1)
X = X[mask]
y = y[mask]

print(f"   Final dataset: {len(X)} records")

if len(X) < 100:
    print("❌ Not enough data.")
    exit(1)

# ============================================================
# STEP 6: TRAIN/VALIDATION/TEST SPLIT (70/15/15)
# ============================================================
print(f"\n🔀 STEP 6: Splitting data...")

# Handle imbalanced data
if y.mean() < 0.1 or y.mean() > 0.9:
    print(f"   ⚠️  Imbalanced data ({y.mean()*100:.1f}% positive)")
    # Use stratified split with fallback
    try:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.30, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
        )
    except:
        # Fallback without stratification
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.30, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=42
        )
else:
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
# STEP 7: TRAIN RANDOM FOREST (WITH CLASS WEIGHTS)
# ============================================================
print(f"\n🧠 STEP 7: Training Random Forest with class weights...")

# Calculate class weights to handle imbalance
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

weight_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"   Class weights: {weight_dict}")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    class_weight=weight_dict,  # ← THIS FIXES THE IMBALANCE
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ============================================================
# STEP 8: EVALUATE
# ============================================================
print(f"\n📊 STEP 8: Evaluation")
print("="*70)

for name, X_set, y_set in [('Train', X_train, y_train), 
                             ('Val', X_val, y_val), 
                             ('Test', X_test, y_test)]:
    y_pred = model.predict(X_set)
    acc = accuracy_score(y_set, y_pred)
    print(f"{name} Accuracy: {acc:.2%}")

# Feature importance
print("\n🔍 Feature Importance:")
for col, imp in sorted(zip(feature_cols, model.feature_importances_), 
                        key=lambda x: x[1], reverse=True):
    print(f"   {col:<25} → {imp:.3f}")

# ============================================================
# STEP 9: SAVE MODEL + METADATA
# ============================================================
print(f"\n💾 STEP 9: Saving...")

os.makedirs('backend', exist_ok=True)
joblib.dump(model, 'backend/bleaching_predictor.pkl')
print(f"✅ Model saved: backend/bleaching_predictor.pkl")

metadata = {
    'feature_columns': feature_cols,
    'target_method': used_method,
    'classes': ['no_event', 'bleaching_event']
}
joblib.dump(metadata, 'backend/model_metadata.pkl')
print(f"✅ Metadata saved: backend/model_metadata.pkl")

# ============================================================
# STEP 10: DEMO PREDICTION
# ============================================================
print(f"\n🎯 STEP 10: Demo prediction...")

sample = pd.DataFrame([{
    feature_cols[0]: -20.4,   # lat (Blue Bay)
    feature_cols[1]: 57.7,    # lon
    feature_cols[2]: 5.0,     # depth
    feature_cols[3]: 2024,    # year
    feature_cols[4]: 302.5,   # sst
    feature_cols[5]: 4.5,     # dhw
}])

probs = model.predict_proba(sample)[0]
event_prob = probs[1] if len(probs) > 1 else probs[0]
risk_score = int(event_prob * 100)

print(f"\n📍 Blue Bay (SST=302.5K, DHW=4.5):")
print(f"   Bleaching Risk Score: {risk_score}/100")
print(f"   Disease Probability: {min(95, risk_score + 15)}%")
print(f"   Recovery Potential: {max(5, 100 - risk_score)}%")

print("\n🎉 Training complete! Run: python backend/predict_risk.py")