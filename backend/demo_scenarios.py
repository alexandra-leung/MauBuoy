# demo_scenarios.py
"""
Pre-built demo scenarios for the hackathon presentation.
Each scenario tests the COMPLETE AI PIPELINE:
1. Computer Vision (health classification)
2. Random Forest (bleaching risk prediction)
3. Restoration Knowledge Base (intervention recommendation)
4. Gemma Orchestrator (final explanation)
"""

DEMO_SCENARIOS = {
    "scenario_1_healthy": {
        "name": "Le Morne – Healthy Reef",
        "description": "Demonstrate prevention. Healthy coral in optimal conditions requiring standard maintenance.",
        "cv_output": {
            "health": "healthy",
            "confidence": 0.95
        },
        "prediction_input": {
            "latitude": -20.46,
            "longitude": 57.32,
            "depth_m": 18,
            "sst_celsius": 27.5,
            "dhw": 0.5
        },
        "location": "Le Morne",
        "expected_prediction": "No Bleaching",
        "expected_risk": "LOW",
        "expected_recommendation": "coral_gardening"
    },
    
    "scenario_2_stressed": {
        "name": "Balaclava – Early Thermal Stress",
        "description": "Demonstrate early warning before bleaching. Coral showing signs of stress due to rising temperatures.",
        "cv_output": {
            "health": "stressed",
            "confidence": 0.89
        },
        "prediction_input": {
            "latitude": -20.11,
            "longitude": 57.53,
            "depth_m": 8,
            "sst_celsius": 29.2,
            "dhw": 3.5
        },
        "location": "Balaclava",
        "expected_prediction": "Bleaching",
        "expected_risk": "MODERATE",
        "expected_recommendation": "substrate_stabilization"
    },
    
    "scenario_3_bleached": {
        "name": "Blue Bay – Active Bleaching Event",
        "description": "Demonstrate emergency response. Active bleaching event requiring immediate, high-intensity intervention.",
        "cv_output": {
            "health": "bleached",
            "confidence": 0.93
        },
        "prediction_input": {
            "latitude": -20.44,
            "longitude": 57.72,
            "depth_m": 5,
            "sst_celsius": 31.2,
            "dhw": 8.5
        },
        "location": "Blue Bay",
        "expected_prediction": "Bleaching",
        "expected_risk": "CRITICAL",
        "expected_recommendation": "heat_resistant_outplanting"
    }
}