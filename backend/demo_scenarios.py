# demo_scenarios.py
"""
Pre-built demo scenarios for the hackathon presentation.
Each scenario tests a different decision path.
"""

DEMO_SCENARIOS = {
    "scenario_1_optimal": {
        "name": "Le Morne - Optimal Conditions",
        "description": "Healthy coral in pristine conditions → Should recommend coral gardening",
        "cv_output": {"health": "healthy", "confidence": 0.94},
        "sensor_data": {
            "water_temp_c": 27.5,
            "ph_level": 8.2,
            "salinity_ppt": 35.5,
            "turbidity_ntu": 1.8,
            "depth_m": 5.0
        },
        "location": "Le Morne Lagoon",
        "expected_recommendation": "coral_gardening",
        "expected_alert": "LOW"
    },
    
    "scenario_2_thermal_stress": {
        "name": "Blue Bay - Thermal Stress",
        "description": "Bleached coral in hot water → Should recommend heat-resistant outplanting",
        "cv_output": {"health": "bleached", "confidence": 0.92},
        "sensor_data": {
            "water_temp_c": 29.8,
            "ph_level": 8.1,
            "salinity_ppt": 35.2,
            "turbidity_ntu": 2.1,
            "depth_m": 4.5
        },
        "location": "Blue Bay Marine Park",
        "expected_recommendation": "heat_resistant_outplanting",
        "expected_alert": "HIGH"
    },
    
    "scenario_3_high_turbidity": {
        "name": "Balaclava - High Turbidity",
        "description": "Stressed coral in murky water → Should recommend substrate stabilization",
        "cv_output": {"health": "stressed", "confidence": 0.87},
        "sensor_data": {
            "water_temp_c": 28.0,
            "ph_level": 7.9,
            "salinity_ppt": 34.8,
            "turbidity_ntu": 8.5,
            "depth_m": 6.0
        },
        "location": "Balaclava Marine Park",
        "expected_recommendation": "substrate_stabilization",
        "expected_alert": "MODERATE"
    },
    
    "scenario_4_critical": {
        "name": "Pointe-aux-Feuilles - Critical Conditions",
        "description": "Severely bleached coral in hostile environment → Should recommend no intervention",
        "cv_output": {"health": "bleached", "confidence": 0.96},
        "sensor_data": {
            "water_temp_c": 31.2,
            "ph_level": 7.7,
            "salinity_ppt": 33.5,
            "turbidity_ntu": 12.0,
            "depth_m": 8.0
        },
        "location": "Pointe-aux-Feuilles",
        "expected_recommendation": "no_intervention_monitoring",
        "expected_alert": "CRITICAL"
    }
}