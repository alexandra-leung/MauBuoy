# backend/predict_risk.py
"""
ReefGuardian AI - 4-Class Prediction Tool
Outputs probabilities for: healthy, stressed, bleached, dead
Matches CV model's 4 classes for cross-validation with Gemma 4.
"""

import joblib
import numpy as np
import pandas as pd
import os

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'bleaching_predictor.pkl')
METADATA_PATH = os.path.join(BASE_DIR, 'model_metadata.pkl')

try:
    predictor = joblib.load(MODEL_PATH)
    metadata = joblib.load(METADATA_PATH)
    FEATURE_COLS = (
        predictor.feature_names_in_.tolist()
        if hasattr(predictor, 'feature_names_in_')
        else metadata.get('feature_columns', [])
    )
    CLASSES = predictor.classes_.tolist()
    print(f"Model loaded. Classes: {CLASSES}")
    print(f"   Features: {FEATURE_COLS}")
except FileNotFoundError:
    print(f"Model not found. Run: python backend/train_prediction_model.py")
    predictor = None
    FEATURE_COLS = []
    CLASSES = ['healthy', 'stressed', 'bleached', 'dead']


def predict_health_probabilities(latitude, longitude, sst_celsius, ph_level, species_observed, marine_heatwave):
    """
    Predicts probability distribution across 4 health classes.
    """
    
    if predictor is None:
        return {
            "error": "Model not loaded",
            "probabilities": {cls: 0.25 for cls in CLASSES},
            "predicted_class": "unknown",
            "confidence": 0.0,
            "bleaching_risk_score": 50,
            "risk_level": "UNKNOWN",
            "disease_outbreak_probability": "50%",
            "recovery_potential": "50%",
            "reasoning": "Prediction model unavailable"
        }
    
    # Build feature vector with names matching the saved model
    features_dict = {}
    for col in FEATURE_COLS:
        key = col.lower()
        if 'latitude' in key:
            features_dict[col] = latitude
        elif 'longitude' in key:
            features_dict[col] = longitude
        elif 'sst' in key:
            features_dict[col] = sst_celsius
        elif 'ph' in key:
            features_dict[col] = ph_level
        elif 'species' in key:
            features_dict[col] = species_observed
        elif 'heatwave' in key:
            features_dict[col] = int(marine_heatwave)
        else:
            raise ValueError(f"Unexpected feature name: {col}")

    features_df = pd.DataFrame([features_dict])

    # Get probabilities for all classes in the model's class order
    probs = predictor.predict_proba(features_df)[0]
    model_classes = predictor.classes_.tolist()
    model_prob_dict = {cls: float(p) for cls, p in zip(model_classes, probs)}

    # Normalize output order to healthy/stressed/bleached/dead
    ordered_classes = ['healthy', 'stressed', 'bleached', 'dead']
    prob_dict = {cls: round(model_prob_dict.get(cls, 0.0), 3) for cls in ordered_classes}

    # Find predicted class (highest probability)
    predicted_class = max(prob_dict, key=prob_dict.get)
    confidence = float(prob_dict[predicted_class])
    
    # Calculate risk metrics (MauBuoy's 3 metrics)
    bleaching_risk_score = int((prob_dict['bleached'] + prob_dict['dead']) * 100)
    
    if bleaching_risk_score < 30:
        risk_level = "LOW"
    elif bleaching_risk_score < 60:
        risk_level = "MODERATE"
    elif bleaching_risk_score < 80:
        risk_level = "HIGH"
    else:
        risk_level = "CRITICAL"
    
    # Derive MauBuoy's 3 metrics
    disease_prob = int(prob_dict['dead'] * 100)
    recovery_potential = int((prob_dict['healthy'] + prob_dict['stressed'] * 0.5) * 100)
    
    # Generate reasoning
    reasoning = (
        f"Model predicts {predicted_class} state ({confidence*100:.1f}% confidence). "
        f"Risk score: {bleaching_risk_score}/100 ({risk_level})."
    )
    
    return {
        "probabilities": prob_dict,
        "predicted_class": predicted_class,
        "confidence": round(confidence, 3),
        "bleaching_risk_score": bleaching_risk_score,
        "risk_level": risk_level,
        "disease_outbreak_probability": f"{disease_prob}%",
        "recovery_potential": f"{recovery_potential}%",
        "reasoning": reasoning
    }


# Backward-compatible wrapper for orchestrator
def predict_14_day_bleaching_risk(latitude, longitude, depth, sst_kelvin, dhw, coral_health="unknown"):
    """Wrapper that converts Kelvin to Celsius and calls the main function."""
    
    # Convert Kelvin to Celsius
    sst_celsius = sst_kelvin - 273.15
    
    # Estimate other parameters
    marine_heatwave = sst_celsius > 30.0
    ph_level = 8.1  # Default ocean pH
    species_observed = 20  # Default species count
    
    # Call the main prediction function
    result = predict_health_probabilities(
        latitude=latitude,
        longitude=longitude,
        sst_celsius=sst_celsius,
        ph_level=ph_level,
        species_observed=species_observed,
        marine_heatwave=marine_heatwave
    )
    
    # Add state-aware reasoning based on CV model's classification
    coral_health = coral_health.lower()
    
    if coral_health == "healthy":
        result["reasoning"] += " CV model confirms healthy state. Focus on preventive monitoring."
    elif coral_health == "stressed":
        result["reasoning"] += " CV model shows stress. High disease risk - immediate intervention recommended."
    elif coral_health == "bleached":
        result["reasoning"] += " CV model confirms bleaching. Mortality risk elevated - consider heat-resistant outplanting."
    elif coral_health == "dead":
        result["reasoning"] += " CV model shows dead coral. Focus on restoration planning."
    
    return result


if __name__ == "__main__":
    print("\nTesting 4-class prediction model...")
    
    # Test 1: Le Morne (Low stress)
    r1 = predict_health_probabilities(-20.46, 57.32, 27.5, 8.1, 45, False)
    print("\n1. Le Morne (Low Stress):")
    for cls, prob in r1['probabilities'].items():
        print(f"   {cls:<10} -> {prob*100:5.1f}%")
    print(f"   Predicted: {r1['predicted_class']} ({r1['confidence']*100:.1f}%)")
    print(f"   Risk Score: {r1['bleaching_risk_score']}/100 ({r1['risk_level']})")
    print(f"   Disease Probability: {r1['disease_outbreak_probability']}")
    print(f"   Recovery Potential: {r1['recovery_potential']}")
    
    # Test 2: Blue Bay (High stress)
    r2 = predict_health_probabilities(-20.4, 57.7, 31.2, 7.8, 5, True)
    print("\n2. Blue Bay (High Stress):")
    for cls, prob in r2['probabilities'].items():
        print(f"   {cls:<10} -> {prob*100:5.1f}%")
    print(f"   Predicted: {r2['predicted_class']} ({r2['confidence']*100:.1f}%)")
    print(f"   Risk Score: {r2['bleaching_risk_score']}/100 ({r2['risk_level']})")
    print(f"   Disease Probability: {r2['disease_outbreak_probability']}")
    print(f"   Recovery Potential: {r2['recovery_potential']}")
    
    # Test 3: Compare with CV model
    print("\n3. Comparison with CV model:")
    cv_output = {"health": "bleached", "confidence": 0.92}
    pred_output = r2
    print(f"   CV model says:       {cv_output['health']} ({cv_output['confidence']*100:.1f}%)")
    print(f"   Prediction model:    {pred_output['predicted_class']} ({pred_output['confidence']*100:.1f}%)")
    print(f"   Agreement:           {'YES' if cv_output['health'] == pred_output['predicted_class'] else 'NO'}")