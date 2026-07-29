# restoration_knowledge_base.py
"""
ReefGuardian AI - Restoration Knowledge Base
Contains proven coral restoration techniques with their environmental thresholds.
Gemma 4 uses this to match current conditions with optimal techniques.
"""

RESTORATION_TECHNIQUES = {
    "coral_gardening": {
        "name": "Coral Gardening (Underwater Nurseries)",
        "description": "Fragments of healthy coral are grown on underwater tree structures for 6-12 months, then outplanted onto degraded reefs.",
        "ideal_conditions": {
            "max_temp_c": 29.0,
            "min_temp_c": 24.0,
            "max_turbidity_ntu": 5.0,
            "min_ph": 8.0,
            "max_depth_m": 15.0
        },
        "cost_per_hectare_usd": 10000,
        "base_survival_rate_percent": 75,
        "time_to_outplant_months": 8,
        "best_for": ["moderate_degradation", "stable_climate"],
        "limitations": ["Fails under thermal stress >29°C", "Requires clear water for photosynthesis"]
    },
    
    "substrate_stabilization": {
        "name": "Substrate Stabilization (Reef Balls / Biorock)",
        "description": "Deploying artificial limestone or electrified structures to stabilize rubble and provide new attachment points for coral larvae.",
        "ideal_conditions": {
            "max_temp_c": 31.0,
            "min_temp_c": 23.0,
            "max_turbidity_ntu": 10.0,
            "min_ph": 7.8,
            "max_depth_m": 25.0
        },
        "cost_per_hectare_usd": 25000,
        "base_survival_rate_percent": 60,
        "time_to_outplant_months": 12,
        "best_for": ["rubble_zones", "high_turbidity", "post-storm_damage"],
        "limitations": ["High upfront cost", "Requires electricity for Biorock systems"]
    },
    
    "heat_resistant_outplanting": {
        "name": "Assisted Evolution (Heat-Resistant Coral Species)",
        "description": "Outplanting genetically selected or naturally heat-adapted coral strains bred to survive elevated temperatures.",
        "ideal_conditions": {
            "max_temp_c": 32.0,
            "min_temp_c": 25.0,
            "max_turbidity_ntu": 8.0,
            "min_ph": 7.9,
            "max_depth_m": 20.0
        },
        "cost_per_hectare_usd": 30000,
        "base_survival_rate_percent": 85,
        "time_to_outplant_months": 10,
        "best_for": ["thermal_stress_zones", "frequent_bleaching", "climate_change_adaptation"],
        "limitations": ["Requires specialized nursery stock", "Limited genetic diversity"]
    },
    
    "larval_propagation": {
        "name": "Larval Reseeding (Coral IVF)",
        "description": "Collecting coral spawn during mass spawning events, rearing larvae in controlled conditions, and releasing them onto degraded reefs.",
        "ideal_conditions": {
            "max_temp_c": 28.5,
            "min_temp_c": 24.0,
            "max_turbidity_ntu": 3.0,
            "min_ph": 8.1,
            "max_depth_m": 12.0
        },
        "cost_per_hectare_usd": 15000,
        "base_survival_rate_percent": 40,
        "time_to_outplant_months": 6,
        "best_for": ["large_scale_restoration", "genetic_diversity", "pristine_conditions"],
        "limitations": ["Requires precise timing with spawning events", "Low survival rate in stressed environments"]
    },
    
    "no_intervention_monitoring": {
        "name": "Passive Recovery (Monitoring Only)",
        "description": "No active restoration. Reef is monitored while natural recovery occurs. Recommended when conditions are too hostile for intervention.",
        "ideal_conditions": {
            "max_temp_c": 35.0,  # Accepts anything
            "min_temp_c": 0.0,
            "max_turbidity_ntu": 50.0,
            "min_ph": 7.0,
            "max_depth_m": 50.0
        },
        "cost_per_hectare_usd": 2000,  # Just monitoring costs
        "base_survival_rate_percent": 20,
        "time_to_outplant_months": 0,
        "best_for": ["extreme_conditions", "insufficient_budget", "severe_thermal_stress"],
        "limitations": ["Very low recovery rate", "Does not address root causes"]
    }
}


def get_technique_recommendation(cv_output, sensor_data, location_name):
    """
    Helper function to evaluate which technique is best.
    (Your friend can call this OR let Gemma 4 reason through it directly)
    """
    health = cv_output.get("health", "unknown").lower()
    confidence = cv_output.get("confidence", 0.5)
    
    temp = sensor_data.get("water_temp_c", 28.0)
    turbidity = sensor_data.get("turbidity_ntu", 3.0)
    ph = sensor_data.get("ph_level", 8.1)
    
    # Score each technique
    scores = {}
    for tech_id, tech in RESTORATION_TECHNIQUES.items():
        score = 0
        conditions = tech["ideal_conditions"]
        
        # Temperature check
        if conditions["min_temp_c"] <= temp <= conditions["max_temp_c"]:
            score += 30
        elif temp > conditions["max_temp_c"]:
            score -= (temp - conditions["max_temp_c"]) * 10
        
        # Turbidity check
        if turbidity <= conditions["max_turbidity_ntu"]:
            score += 25
        else:
            score -= (turbidity - conditions["max_turbidity_ntu"]) * 5
        
        # pH check
        if ph >= conditions["min_ph"]:
            score += 25
        else:
            score -= (conditions["min_ph"] - ph) * 20
        
        # Bleached coral bonus for active restoration
        if "bleached" in health and tech_id != "no_intervention_monitoring":
            score += 20
        
        scores[tech_id] = {
            "score": score,
            "technique": tech["name"],
            "survival_rate": tech["base_survival_rate_percent"],
            "cost": tech["cost_per_hectare_usd"]
        }
    
    # Sort by score
    ranked = sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)
    return ranked[0]  # Best match