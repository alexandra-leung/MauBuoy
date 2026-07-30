# backend/public_awareness_kreol.py
"""
ReefGuardian AI - Public Awareness Kreol Generator
Robust JSON extraction from Gemma 4 responses.
"""

import ollama
import json
import re


def extract_json_from_text(text):
    """
    Extracts JSON from text that may contain markdown, explanations, etc.
    Handles various formats Gemma might output.
    """
    
    # Try 1: Direct parse (if it's pure JSON)
    try:
        return json.loads(text)
    except:
        pass
    
    # Try 2: Remove markdown code blocks
    cleaned = text
    if '```json' in cleaned:
        cleaned = cleaned.split('```json')[1].split('```')[0]
    elif '```' in cleaned:
        cleaned = cleaned.split('```')[1].split('```')[0]
    
    try:
        return json.loads(cleaned.strip())
    except:
        pass
    
    # Try 3: Find JSON object using regex (first { to last })
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass
    
    # Try 4: Manual extraction (find first { and last })
    start = text.find('{')
    end = text.rfind('}')
    
    if start != -1 and end > start:
        json_str = text[start:end+1]
        
        # Fix common issues
        # Remove trailing commas before }
        json_str = re.sub(r',\s*}', '}', json_str)
        # Remove trailing commas before ]
        json_str = re.sub(r',\s*\]', ']', json_str)
        
        try:
            return json.loads(json_str)
        except:
            pass
    
    return None


def generate_public_awareness_message(case_data):
    """
    Generates a public awareness message in Mauritian Kreol.
    """
    
    prompt = f"""You are an expert environmental communicator and native speaker of Mauritian Creole (Kreol Morisien).

Your task is to translate and adapt the following technical coral reef alert into an engaging, easy-to-understand PUBLIC AWARENESS message in authentic Mauritian Creole.

TARGET AUDIENCE: Local fishermen, community members, and tourists. 
TONE: Urgent but hopeful, respectful, clear, and community-focused.

RULES:
1. Use authentic Mauritian Creole grammar and vocabulary (e.g., "nou", "bann", "la", "finn", "akoz").
2. Keep location names in English/French (e.g., "Blue Bay", "Le Morne").
3. Keep scientific species names unchanged, but explain them simply if needed.
4. Make it sound like a community announcement, not a robotic translation.
5. Output MUST be valid JSON with exactly three keys: "title", "message", "call_to_action".

TECHNICAL DATA:
- Location: {case_data.get('location', 'Unknown')}
- Health Status: {case_data.get('health_status', 'Unknown')}
- Environmental Cause: {case_data.get('environmental_cause', 'Unknown')}
- Action Taken: {case_data.get('action_taken', 'Unknown')}
- Call to Action: {case_data.get('call_to_action', 'Unknown')}

OUTPUT FORMAT (JSON ONLY, no markdown, no explanations):
{{
  "title": "Short, attention-grabbing Kreol title (e.g., 'Alert: Koray dan Blue Bay an Danger')",
  "message": "2-3 sentences explaining the situation simply in Kreol.",
  "call_to_action": "1-2 sentences telling the public what they can do to help, in Kreol."
}}
"""

    try:
        print("🔄 Calling Gemma 4...")
        response = ollama.chat(
            model='gemma4',
            messages=[{"role": "user", "content": prompt}],
            options={
                'temperature': 0.4,
                'num_predict': 500
            }
        )
        
        raw_content = response['message']['content'].strip()
        
        print(f"📝 Raw response ({len(raw_content)} chars):")
        print("-" * 70)
        print(raw_content[:500])  # Show first 500 chars
        print("-" * 70)
        
        # Extract JSON using robust method
        parsed_response = extract_json_from_text(raw_content)
        
        if parsed_response is None:
            print("❌ Could not extract JSON from response")
            return {
                "success": False,
                "error": "Failed to extract JSON",
                "raw_response": raw_content
            }
        
        return {
            "success": True,
            "kreol_title": parsed_response.get("title", ""),
            "kreol_message": parsed_response.get("message", ""),
            "kreol_call_to_action": parsed_response.get("call_to_action", ""),
            "original_data": case_data
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    print("="*70)
    print("📢 REEFGUARDIAN AI - Public Awareness Kreol Test")
    print("="*70)
    
    sample_critical_case = {
        "location": "Blue Bay Marine Park",
        "health_status": "78% of coral colonies show severe bleaching.",
        "environmental_cause": "Sea surface temperature has reached 30.4°C with a Degree Heating Week (DHW) of 8.5.",
        "action_taken": "Marine teams are deploying heat-resistant coral fragments and installing temporary shading.",
        "call_to_action": "Please avoid anchoring boats in the area and report any further coral damage to the Marine Conservation Society."
    }
    
    print("\n📥 Input Data:")
    for key, value in sample_critical_case.items():
        print(f"   {key}: {value}")
    
    print("\n🔄 Calling Gemma 4 to generate Kreol public awareness message...")
    result = generate_public_awareness_message(sample_critical_case)
    
    if result["success"]:
        print("\n✅ SUCCESS! Generated Kreol Message:")
        print("-" * 70)
        print(f"📌 TITLE:\n{result['kreol_title']}\n")
        print(f"📝 MESSAGE:\n{result['kreol_message']}\n")
        print(f"📢 CALL TO ACTION:\n{result['kreol_call_to_action']}")
        print("-" * 70)
    else:
        print(f"\n❌ FAILED: {result['error']}")
        if 'raw_response' in result:
            print(f"\n📄 Full raw response:\n{result['raw_response']}")