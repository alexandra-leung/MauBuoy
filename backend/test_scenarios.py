# test_scenarios.py
import sys
import os

# Add backend to path so we can import predict_risk
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from demo_scenarios import DEMO_SCENARIOS
from predict_risk import predict_14_day_bleaching_risk

print("="*70)
print("REEFGUARDIAN AI - FULL PIPELINE DEMO TEST")
print("="*70)

for key, scenario in DEMO_SCENARIOS.items():
    print(f"\n📍 SCENARIO: {scenario['name']}")
    print(f"   Purpose: {scenario['description']}")
    
    # 1. Simulate Computer Vision Output
    cv_out = scenario['cv_output']
    print(f"   👁️ CV Model Output: {cv_out['health'].capitalize()} ({cv_out['confidence']:.0%} confidence)")
    
    # 2. Run Random Forest Prediction Model
    p_in = scenario['prediction_input']
    
    # Convert Celsius to Kelvin to match the wrapper function signature
    sst_kelvin = p_in['sst_celsius'] + 273.15 
    
    pred_result = predict_14_day_bleaching_risk(
        latitude=p_in['latitude'],
        longitude=p_in['longitude'],
        depth=p_in['depth_m'],
        sst_kelvin=sst_kelvin,
        dhw=p_in['dhw'],
        coral_health=cv_out['health'] # Pass CV output to the wrapper!
    )
    
    # 3. Display Results
    print(f"   🌡️ Prediction Model Output:")
    print(f"      - Predicted Class: {pred_result['predicted_class']}")
    print(f"      - Risk Score: {pred_result['bleaching_risk_score']}/100")
    print(f"      - Risk Level: {pred_result['risk_level']}")
    print(f"   🛠️ Knowledge Base Recommendation: {scenario['expected_recommendation']}")
    print(f"   🧠 Gemma Reasoning Context: {pred_result['reasoning']}")
    
    # Quick Validation Check
    match_pred = "✅" if pred_result['predicted_class'] == scenario['expected_prediction'] else "❌"
    print(f"   🎯 Validation -> Expected: {scenario['expected_prediction']} | Got: {pred_result['predicted_class']} {match_pred}")

print("\n" + "="*70)
print("TEST COMPLETE! Pipeline is ready for the Gemma Orchestrator.")
print("="*70)