# backend/predict_risk.py
"""
ReefGuardian AI - Binary Bleaching Prediction Tool
Outputs probabilities for: No Bleaching (0), Bleaching (1)
Matches the Random Forest model trained in train_prediction_model.py.
"""

import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'bleaching_predictor.pkl')
METADATA_PATH = os.path.join(BASE_DIR, 'model_metadata.pkl')

# Load model and metadata
try:
    predictor = joblib.load(MODEL_PATH)
    metadata = joblib.load(METADATA_PATH)
    
    # Get feature columns from metadata or model
    FEATURE_COLS = metadata.get('feature_columns', [])
    if not FEATURE_COLS and hasattr(predictor, 'feature_names_in_'):
        FEATURE_COLS = predictor.feature_names_in_.tolist()
        
    # Get classes
    CLASSES = metadata.get('classes', ['No Bleaching', 'Bleaching'])
    if not CLASSES and hasattr(predictor, 'classes_'):
        CLASSES = [int(c) for c in predictor.classes_.tolist()]
        
    print(f"Model loaded. Classes: {CLASSES}")
    print(f"   Features: {FEATURE_COLS}")
except FileNotFoundError:
    print(f"Model not found. Run: python backend/train_prediction_model.py")
    predictor = None
    FEATURE_COLS = []
    CLASSES = [0, 1]


def predict_bleaching_risk(latitude, longitude, depth_m, sst_celsius, dhw, year=None):
    """
    Predicts binary bleaching risk (No Bleaching vs Bleaching).
    """
    if predictor is None:
        return {
            "error": "Model not loaded",
            "probabilities": {"No Bleaching": 0.5, "Bleaching": 0.5},
            "predicted_class": "unknown",
            "confidence": 0.0,
            "bleaching_risk_score": 50,
            "risk_level": "UNKNOWN",
            "reasoning": "Prediction model unavailable"
        }
    
    if year is None:
        year = datetime.now().year

    # Build feature dictionary matching the EXACT column names from training
    features_dict = {
        "Latitude": latitude,
        "Longitude": longitude,
        "Depth_m": depth_m,
        "Year": year,
        "Sea_Surface_Temperature_C": sst_celsius,
        "Degree_Heating_Weeks": dhw
    }
    
    # Ensure order matches FEATURE_COLS
    features_df = pd.DataFrame([{col: features_dict[col] for col in FEATURE_COLS}])

    # Get probabilities
    probs = predictor.predict_proba(features_df)[0]
    model_classes = predictor.classes_.tolist()
    
    # Map to standard names (0 -> No Bleaching, 1 -> Bleaching)
    prob_dict = {}
    for cls, p in zip(model_classes, probs):
        if cls == 0:
            prob_dict["No Bleaching"] = float(p)
        elif cls == 1:
            prob_dict["Bleaching"] = float(p)
        else:
            prob_dict[str(cls)] = float(p)
            
    # Ensure both keys exist
    prob_dict.setdefault("No Bleaching", 0.0)
    prob_dict.setdefault("Bleaching", 0.0)

    # Find predicted class
    predicted_class = max(prob_dict, key=prob_dict.get)
    confidence = float(prob_dict[predicted_class])
    
    # Calculate risk metrics
    bleaching_prob = prob_dict.get("Bleaching", 0.0)
    bleaching_risk_score = int(bleaching_prob * 100)
    
    if bleaching_risk_score < 30:
        risk_level = "LOW"
    elif bleaching_risk_score < 60:
        risk_level = "MODERATE"
    elif bleaching_risk_score < 80:
        risk_level = "HIGH"
    else:
        risk_level = "CRITICAL"
    
    # Generate reasoning
    reasoning = (
        f"Model predicts '{predicted_class}' state ({confidence*100:.1f}% confidence). "
        f"Bleaching risk score: {bleaching_risk_score}/100 ({risk_level}). "
        f"Based on SST: {sst_celsius:.1f}°C and DHW: {dhw:.1f}."
    )
    
    return {
        "probabilities": {k: round(v, 3) for k, v in prob_dict.items()},
        "predicted_class": predicted_class,
        "confidence": round(confidence, 3),
        "bleaching_risk_score": bleaching_risk_score,
        "risk_level": risk_level,
        "reasoning": reasoning
    }


# Backward-compatible wrapper for orchestrator
def predict_14_day_bleaching_risk(latitude, longitude, depth, sst_kelvin, dhw, coral_health="unknown"):
    """Wrapper that converts Kelvin to Celsius and calls the main function."""
    
    # Convert Kelvin to Celsius
    sst_celsius = sst_kelvin - 273.15
    
    # Call the main prediction function
    result = predict_bleaching_risk(
        latitude=latitude,
        longitude=longitude,
        depth_m=depth,
        sst_celsius=sst_celsius,
        dhw=dhw
    )
    
    # Add state-aware reasoning if coral_health is provided
    if coral_health and coral_health.lower() != "unknown":
        health = coral_health.lower()
        if health == "healthy":
            result["reasoning"] += " CV model confirms healthy state. Focus on preventive monitoring."
        elif health == "stressed":
            result["reasoning"] += " CV model shows stress. High disease risk - immediate intervention recommended."
        elif health == "bleached":
            result["reasoning"] += " CV model confirms bleaching. Mortality risk elevated - consider heat-resistant outplanting."
        elif health == "dead":
            result["reasoning"] += " CV model shows dead coral. Focus on restoration planning."
    
    return result


if __name__ == "__main__":
    print("\nTesting binary bleaching prediction model...")
    
    # Test 1: Le Morne (Low stress)
    # sst_kelvin = 27.5 + 273.15 = 300.65
    r1 = predict_14_day_bleaching_risk(-20.46, 57.32, 18.0, 300.65, 0.5)
    print("\n1. Le Morne (Low Stress):")
    for cls, prob in r1['probabilities'].items():
        print(f"   {cls:<15} -> {prob*100:5.1f}%")
    print(f"   Predicted: {r1['predicted_class']} ({r1['confidence']*100:.1f}%)")
    print(f"   Risk Score: {r1['bleaching_risk_score']}/100 ({r1['risk_level']})")
    print(f"   Reasoning: {r1['reasoning']}")
    
    # Test 2: Blue Bay (High stress)
    # sst_kelvin = 31.2 + 273.15 = 304.35
    r2 = predict_14_day_bleaching_risk(-20.4, 57.7, 5.0, 304.35, 8.5)
    print("\n2. Blue Bay (High Stress):")
    for cls, prob in r2['probabilities'].items():
        print(f"   {cls:<15} -> {prob*100:5.1f}%")
    print(f"   Predicted: {r2['predicted_class']} ({r2['confidence']*100:.1f}%)")
    print(f"   Risk Score: {r2['bleaching_risk_score']}/100 ({r2['risk_level']})")
    print(f"   Reasoning: {r2['reasoning']}")