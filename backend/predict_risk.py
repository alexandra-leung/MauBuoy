# backend/predict_risk.py
import joblib
import numpy as np
import os

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'bleaching_predictor.pkl')
try:
    predictor = joblib.load(MODEL_PATH)
except FileNotFoundError:
    print("⚠️ Model not found. Run train_prediction_model.py first!")
    predictor = None


def predict_14_day_bleaching_risk(latitude, longitude, depth, sst_kelvin, dhw, coral_health="unknown"):
    """
    State-aware prediction: predicts multiple risks based on current coral health.
    
    Args:
        latitude, longitude, depth, sst_kelvin, dhw: Environmental data
        coral_health: Current state from CV model ("healthy", "stressed", "bleached", "dead")
    
    Returns:
        Dict with multiple risk metrics based on coral state
    """
    
    if predictor is None:
        return {
            "error": "Model not loaded",
            "bleaching_risk_score": 50,
            "risk_level": "UNKNOWN"
        }
    
    # Format input for the model
    features = np.array([[latitude, longitude, depth, 2024, sst_kelvin, sst_kelvin, dhw, dhw, dhw]])
    
    # Get base bleaching probability from the model
    bleaching_probability = predictor.predict_proba(features)[0][1]
    bleaching_risk_score = int(bleaching_probability * 100)
    
    # Determine risk level
    risk_level = "LOW" if bleaching_risk_score < 30 else "MODERATE" if bleaching_risk_score < 60 else "HIGH" if bleaching_risk_score < 80 else "CRITICAL"
    
    # STATE-AWARE PREDICTIONS
    coral_health = coral_health.lower()
    
    if coral_health == "healthy":
        # Healthy coral: Focus on PREVENTION
        return {
            "bleaching_risk_score": bleaching_risk_score,
            "risk_level": risk_level,
            "disease_outbreak_probability": f"{min(30, bleaching_risk_score // 2)}%",
            "mortality_risk": f"{min(20, bleaching_risk_score // 3)}%",
            "recovery_potential": "95%",
            "reasoning": f"Healthy coral detected. Bleaching risk is {risk_level} ({bleaching_risk_score}/100). Focus on preventive monitoring."
        }
    
    elif coral_health == "stressed":
        # Stressed coral: Focus on EARLY WARNING
        disease_prob = min(70, bleaching_risk_score + 20)
        mortality_risk = min(50, bleaching_risk_score)
        return {
            "bleaching_risk_score": bleaching_risk_score,
            "risk_level": risk_level,
            "disease_outbreak_probability": f"{disease_prob}%",
            "mortality_risk": f"{mortality_risk}%",
            "recovery_potential": f"{max(30, 100 - bleaching_risk_score)}%",
            "reasoning": f"Stressed coral detected. High risk of disease outbreak ({disease_prob}%) and mortality ({mortality_risk}%). Immediate intervention recommended."
        }
    
    elif coral_health == "bleached":
        # Bleached coral: Focus on INTERVENTION
        mortality_risk = min(80, bleaching_risk_score + 30)
        recovery_potential = max(10, 100 - bleaching_risk_score - 20)
        return {
            "bleaching_risk_score": bleaching_risk_score,
            "risk_level": risk_level,
            "disease_outbreak_probability": f"{min(85, bleaching_risk_score + 25)}%",
            "mortality_risk": f"{mortality_risk}%",
            "recovery_potential": f"{recovery_potential}%",
            "reasoning": f"Bleached coral detected. Mortality risk is {mortality_risk}%. Recovery potential is {recovery_potential}% with immediate intervention (e.g., shading, cooling)."
        }
    
    elif coral_health == "dead":
        # Dead coral: Focus on RESTORATION PLANNING
        return {
            "bleaching_risk_score": 0,  # Already dead, bleaching risk irrelevant
            "risk_level": "N/A",
            "disease_outbreak_probability": "0%",
            "mortality_risk": "100%",
            "recovery_potential": "0%",
            "ecosystem_recovery_estimate": "5-10 years without intervention",
            "reasoning": f"Coral is dead. Bleaching risk no longer applicable. Focus on restoration planning: substrate stabilization or coral gardening required."
        }
    
    else:
        # Unknown state: Return base prediction
        return {
            "bleaching_risk_score": bleaching_risk_score,
            "risk_level": risk_level,
            "disease_outbreak_probability": f"{min(95, bleaching_risk_score + 15)}%",
            "recovery_potential": f"{max(5, 100 - bleaching_risk_score)}%",
            "reasoning": f"Unknown coral health state. Base bleaching risk is {risk_level} ({bleaching_risk_score}/100)."
        }


# Quick test
if __name__ == "__main__":
    print("Testing state-aware predictions...\n")
    
    # Test 1: Healthy coral
    result1 = predict_14_day_bleaching_risk(-20.4, 57.7, 5.0, 301.5, 2.0, coral_health="healthy")
    print("1. HEALTHY coral:")
    print(f"   Bleaching Risk: {result1['bleaching_risk_score']}/100 ({result1['risk_level']})")
    print(f"   Disease Risk: {result1['disease_outbreak_probability']}")
    print(f"   Mortality Risk: {result1['mortality_risk']}")
    print(f"   Recovery: {result1['recovery_potential']}")
    print()
    
    # Test 2: Stressed coral
    result2 = predict_14_day_bleaching_risk(-20.4, 57.7, 5.0, 302.0, 3.5, coral_health="stressed")
    print("2. STRESSED coral:")
    print(f"   Bleaching Risk: {result2['bleaching_risk_score']}/100 ({result2['risk_level']})")
    print(f"   Disease Risk: {result2['disease_outbreak_probability']}")
    print(f"   Mortality Risk: {result2['mortality_risk']}")
    print(f"   Recovery: {result2['recovery_potential']}")
    print()
    
    # Test 3: Bleached coral
    result3 = predict_14_day_bleaching_risk(-20.4, 57.7, 5.0, 302.5, 4.5, coral_health="bleached")
    print("3. BLEACHED coral:")
    print(f"   Bleaching Risk: {result3['bleaching_risk_score']}/100 ({result3['risk_level']})")
    print(f"   Disease Risk: {result3['disease_outbreak_probability']}")
    print(f"   Mortality Risk: {result3['mortality_risk']}")
    print(f"   Recovery: {result3['recovery_potential']}")
    print()
    
    # Test 4: Dead coral
    result4 = predict_14_day_bleaching_risk(-20.4, 57.7, 5.0, 302.5, 4.5, coral_health="dead")
    print("4. DEAD coral:")
    print(f"   Bleaching Risk: {result4['bleaching_risk_score']}/100 ({result4['risk_level']})")
    print(f"   Ecosystem Recovery: {result4.get('ecosystem_recovery_estimate', 'N/A')}")
    print(f"   Reasoning: {result4['reasoning']}")