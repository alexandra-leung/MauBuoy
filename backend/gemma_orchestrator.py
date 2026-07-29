# backend/gemma_orchestrator.py
"""
ReefGuardian AI - Gemma 4 Orchestrator
The autonomous agent that:
1. Receives coral health + environmental data
2. Calls Gemma 4 with function calling enabled
3. Executes tool calls (ROI, stress, species, prediction)
4. Returns structured JSON recommendation
"""

import json
import ollama
from typing import Dict, Any

# Import ALL your tools
from roi_calculator import calculate_restoration_roi
from environmental_stress import calculate_environmental_stress
# from species_compatibility import check_species_compatibility
from predict_risk import predict_14_day_risk
from gemma_prompt import build_analysis_prompt
from tool_schemas import TOOLS


# Map tool names to Python functions
TOOL_FUNCTIONS = {
    "calculate_restoration_roi": calculate_restoration_roi,
    "calculate_environmental_stress": calculate_environmental_stress,
    "check_species_compatibility": check_species_compatibility,
    "predict_14_day_risk": predict_14_day_risk  # <-- NEW PREDICTION TOOL
}


def call_gemma_with_tools(prompt: str, max_iterations: int = 5) -> Dict[str, Any]:
    """
    Calls Gemma 4 with tool calling enabled.
    Handles the tool-calling loop automatically.
    """
    
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    for iteration in range(max_iterations):
        # Call Gemma 4 with tools available
        response = ollama.chat(
            model='gemma4',  # Change to 'gemma3' if gemma4 not available
            messages=messages,
            tools=TOOLS,
            options={
                'temperature': 0.3,  # Lower = more deterministic
                'num_predict': 2000
            }
        )
        
        message = response['message']
        
        # Check if Gemma wants to call a tool
        if message.get('tool_calls'):
            messages.append(message)
            
            # Execute each tool call
            for tool_call in message['tool_calls']:
                function_name = tool_call['function']['name']
                arguments = tool_call['function']['arguments']
                
                print(f"🔧 Gemma calling tool: {function_name}")
                print(f"   Arguments: {arguments}")
                
                # Execute the tool
                if function_name in TOOL_FUNCTIONS:
                    try:
                        result = TOOL_FUNCTIONS[function_name](**arguments)
                        print(f"   ✅ Result: {result}")
                        
                        messages.append({
                            "role": "tool",
                            "content": json.dumps(result),
                            "name": function_name
                        })
                    except Exception as e:
                        print(f"   ❌ Error: {e}")
                        messages.append({
                            "role": "tool",
                            "content": json.dumps({"error": str(e)}),
                            "name": function_name
                        })
                else:
                    print(f"   ⚠️  Unknown tool: {function_name}")
                    messages.append({
                        "role": "tool",
                        "content": json.dumps({"error": f"Unknown tool: {function_name}"}),
                        "name": function_name
                    })
            
            continue
        
        # No tool calls - Gemma is done
        return message.get('content', '{}')
    
    return json.dumps({
        "error": "Max iterations reached",
        "alert_level": "UNKNOWN",
        "reasoning": "System timeout during analysis"
    })


def parse_gemma_response(response_text: str) -> Dict[str, Any]:
    """
    Parses Gemma 4's response into clean JSON.
    """
    try:
        cleaned = response_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        
        result = json.loads(cleaned.strip())
        return result
    
    except json.JSONDecodeError as e:
        print(f"⚠️  Failed to parse JSON: {e}")
        
        return {
            "alert_level": "UNKNOWN",
            "coral_health_summary": "Analysis failed",
            "environmental_assessment": "System error",
            "recommended_technique": "no_intervention_monitoring",
            "technique_name": "No Intervention (System Error)",
            "reasoning": f"Failed to parse AI response: {str(e)}",
            "predicted_survival_rate": "N/A",
            "estimated_cost_per_hectare": "N/A",
            "risk_factors": ["System error - manual review required"],
            "next_steps": ["Contact technical support", "Retry analysis"],
            "confidence_score": 0.0,
            "raw_response": response_text[:1000]
        }


def analyze_reef(cv_output: Dict[str, Any], sensor_data: Dict[str, Any], location_name: str) -> Dict[str, Any]:
    """
    Main orchestrator function.
    This is what the backend team will call.
    """
    
    print(f"\n{'='*80}")
    print(f"🌊 REEFGUARDIAN AI ANALYSIS")
    print(f"{'='*80}")
    print(f"Location: {location_name}")
    print(f"Coral Health: {cv_output.get('health')} ({cv_output.get('confidence', 0):.0%} confidence)")
    print(f"Water Temp: {sensor_data.get('water_temp_c')}°C")
    print(f"pH: {sensor_data.get('ph_level')}")
    print(f"Turbidity: {sensor_data.get('turbidity_ntu')} NTU")
    print(f"{'='*80}\n")
    
    # Step 1: Build the prompt
    print("📝 Building analysis prompt...")
    prompt = build_analysis_prompt(cv_output, sensor_data, location_name)
    
    # Step 2: Call Gemma 4 with tools
    print("🧠 Calling Gemma 4 orchestrator...")
    try:
        response_text = call_gemma_with_tools(prompt)
    except Exception as e:
        print(f"❌ Gemma 4 call failed: {e}")
        return {
            "alert_level": "ERROR",
            "reasoning": f"Failed to connect to Gemma 4: {str(e)}",
            "recommended_technique": "no_intervention_monitoring",
            "technique_name": "System Error",
            "predicted_survival_rate": "N/A",
            "estimated_cost_per_hectare": "N/A",
            "risk_factors": ["AI system unavailable"],
            "next_steps": ["Retry later", "Contact technical support"],
            "confidence_score": 0.0
        }
    
    # Step 3: Parse response
    print("📊 Parsing Gemma 4 response...")
    result = parse_gemma_response(response_text)
    
    print(f"\n✅ Analysis complete!")
    print(f"   Alert Level: {result.get('alert_level')}")
    print(f"   Recommended: {result.get('technique_name')}")
    print(f"   Survival: {result.get('predicted_survival_rate')}")
    print(f"   Cost: {result.get('estimated_cost_per_hectare')}")
    print(f"{'='*80}\n")
    
    return result


# Test the orchestrator
if __name__ == "__main__":
    # Test with Blue Bay scenario
    cv_output = {"health": "bleached", "confidence": 0.92}
    sensor_data = {
        "water_temp_c": 29.8,
        "ph_level": 8.1,
        "salinity_ppt": 35.2,
        "turbidity_ntu": 2.1,
        "latitude": -20.4,
        "longitude": 57.7,
        "depth_m": 5.0,
        "sst_kelvin": 302.5,
        "dhw": 4.5,
        "timestamp": "2026-07-28T10:30:00"
    }
    
    result = analyze_reef(cv_output, sensor_data, "Blue Bay Marine Park")
    
    print("\n" + "="*80)
    print("FINAL OUTPUT:")
    print("="*80)
    print(json.dumps(result, indent=2))