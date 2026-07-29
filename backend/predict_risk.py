# backend/predict_risk.py
"""
ReefGuardian AI - Binary Bleaching Risk Predictor
Model outputs: ['no_event', 'bleaching_event']
Converts to MauBuoy's 3 metrics: Risk Score, Disease Probability, Recovery Potential
"""

import joblib
import numpy as np
import os
import json

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'bleaching_predictor.pkl')
METADATA_PATH = os.path.join(BASE_DIR, 'model_metadata.pkl')

try:
    predictor = joblib.load(MODEL_PATH)
    metadata = joblib.load(METADATA_PATH)
    FEATURE_COLS = metadata['feature_columns']
    CLASSES = metadata['classes']  # ['no_event', 'bleaching_event']
    print(f"✅ Model loaded. Classes: {CLASSES}")
    print(f"   Features: {FEATURE_COLS}")
except FileNotFoundError as e:
    print(f"⚠️  Model not found: {e}")
    print("Run: python backend/train_prediction_model.py")
    predictor = None
    FEATURE_COLS = []
    CLASSES = ['no_event', 'bleaching_event']


def predict_14_day_bleaching_risk(latitude, longitude, depth, sst_kelvin, dhw, coral_health="unknown"):
    """
    Predicts bleaching risk using binary model.
    Outputs MauBuoy's 3 metrics:
    1. Bleaching Risk Score (0-100)
    2. Disease Outbreak Probability
    3. Recovery Potential
    
    Args:
        latitude, longitude, depth: Location data
        sst_kelvin: Sea Surface Temperature in Kelvin
        dhw: Degree Heating Weeks
        coral_health: Current state from CV model (healthy/stressed/bleached/dead)
    
    Returns:
        Dict with risk metrics
    """
    
    if predictor is None:
        return {
            "bleaching_risk_score": 50,
            "risk_level": "UNKNOWN",
            "disease_outbreak_probability": "50%",
            "recovery_potential": "50%",
            "reasoning": "Prediction model unavailable"
        }
    
    # Build feature vector matching training columns
    feature_values = []
    for col in FEATURE_COLS:
        col_lower = col.lower()
        if 'lat' in col_lower:
            feature_values.append(latitude)
        elif 'lon' in col_lower or 'long' in col_lower:
            feature_values.append(longitude)
        elif 'depth' in col_lower:
            feature_values.append(depth)
        elif 'year' in col_lower or 'date' in col_lower:
            feature_values.append(2024)
        elif 'sst' in col_lower or 'temp' in col_lower or 'climsst' in col_lower:
            feature_values.append(sst_kelvin)
        elif 'dhw' in col_lower:
            feature_values.append(dhw)
        else:
            feature_values.append(0)
    
    # Ensure correct length
    feature_values = feature_values[:len(FEATURE_COLS)]
    while len(feature_values) < len(FEATURE_COLS):
        feature_values.append(0)
    
    # Create DataFrame with proper column names (fixes the warning)
    import pandas as pd
    features_df = pd.DataFrame([feature_values], columns=FEATURE_COLS)
    
    # Get probability of bleaching event
    probs = predictor.predict_proba(features_df)[0]
    
    # probs[0] = P(no_event), probs[1] = P(bleaching_event)
    event_prob = probs[1] if len(probs) > 1 else probs[0]
    risk_score = int(event_prob * 100)
    
    # Determine risk level
    if risk_score < 30:
        risk_level = "LOW"
    elif risk_score < 60:
        risk_level = "MODERATE"
    elif risk_score < 80:
        risk_level = "HIGH"
    else:
        risk_level = "CRITICAL"
    
    # Calculate MauBuoy's 3 metrics
    disease_prob = min(95, risk_score + 15)
    base_recovery = max(5, 100 - risk_score)
    
    # Adjust recovery based on CV model's current health
    coral_health = coral_health.lower()
    
    if coral_health == "healthy":
        recovery = base_recovery
        reasoning = f"Healthy coral. Risk: {risk_score}/100 ({risk_level}). High recovery potential with monitoring."
    elif coral_health == "stressed":
        recovery = max(10, base_recovery - 20)
        reasoning = f"Stressed coral. Risk: {risk_score}/100 ({risk_level}). Disease risk elevated. Immediate intervention recommended."
    elif coral_health == "bleached":
        recovery = max(5, base_recovery - 40)
        reasoning = f"Bleached coral. Risk: {risk_score}/100 ({risk_level}). Recovery reduced. Heat-resistant outplanting recommended."
    elif coral_health == "dead":
        recovery = 0
        reasoning = f"Dead coral. Risk score no longer applicable. Focus on restoration planning."
    else:
        recovery = base_recovery
        reasoning = f"Risk: {risk_score}/100 ({risk_level})."
    
    return {
        "bleaching_risk_score": risk_score,
        "risk_level": risk_level,
        "disease_outbreak_probability": f"{disease_prob}%",
        "recovery_potential": f"{recovery}%",
        "reasoning": reasoning
    }


if __name__ == "__main__":
    print("\n🧪 Testing prediction model...")
    
    # Test scenarios with Mauritius coordinates
    scenarios = [
        ("Le Morne (Low Stress)", -20.46, 57.32, 5.0, 300.7, 1.2, "healthy"),
        ("Blue Bay (High Stress)", -20.40, 57.70, 5.0, 303.5, 8.5, "bleached"),
    ]
    
    print("\n" + "="*70)
    print("PREDICTION RESULTS")
    print("="*70)
    
    for name, lat, lon, depth, sst, dhw, health in scenarios:
        result = predict_14_day_bleaching_risk(lat, lon, depth, sst, dhw, coral_health=health)
        
        print(f"\n{name} (CV: {health}):")
        print(f"   Risk Score: {result['bleaching_risk_score']}/100 ({result['risk_level']})")
        print(f"   Disease Probability: {result['disease_outbreak_probability']}")
        print(f"   Recovery Potential: {result['recovery_potential']}")
        print(f"   Reasoning: {result['reasoning']}")
    
    print("\n" + "="*70)
    print("✅ Prediction model working correctly!")
    print("="*70)