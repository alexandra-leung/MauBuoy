# backend/kreol_assistant.py
"""
ReefGuardian AI - Kreol Assistant Module
Single entry point for teammates: translate + speak.

USAGE:
    from backend.kreol_assistant import translate_and_speak
    
    audio_path = translate_and_speak("The coral is bleached.")
    # Returns: "backend/audio/response_1234567890.mp3"
"""

from translation import translate_to_kreol
from tts import text_to_speech
import os


def translate_and_speak(english_text, generate_audio=True, rate="+0%"):
    """
    Main function for teammates. Translates English → Kreol → Audio.
    
    Args:
        english_text: English recommendation text
        generate_audio: If True, also generates MP3 audio
        rate: Speech rate (e.g., "-10%" for slower)
    
    Returns:
        Dict with:
        {
            "english": original text,
            "kreol": translated text,
            "audio_path": path to MP3 (or None if generate_audio=False)
        }
    """
    
    if not english_text or not english_text.strip():
        return {
            "english": "",
            "kreol": "",
            "audio_path": None
        }
    
    print(f"\n🌍 Translating to Kreol...")
    kreol_text = translate_to_kreol(english_text)
    print(f"   ✅ Kreol: {kreol_text[:80]}...")
    
    audio_path = None
    if generate_audio and kreol_text:
        print(f"🔊 Generating audio...")
        audio_path = text_to_speech(kreol_text, rate=rate)
        if audio_path:
            print(f"   ✅ Audio: {audio_path}")
    
    return {
        "english": english_text.strip(),
        "kreol": kreol_text,
        "audio_path": audio_path
    }


def translate_recommendation(recommendation_dict):
    """
    Translates a full recommendation JSON from Gemma orchestrator.
    
    Args:
        recommendation_dict: Output from gemma_orchestrator.py
    
    Returns:
        Dict with both English and Kreol versions + audio
    """
    
    # Build readable English text from the JSON
    parts = []
    
    if recommendation_dict.get('alert_level'):
        parts.append(f"Alert level: {recommendation_dict['alert_level']}.")
    
    if recommendation_dict.get('coral_health_summary'):
        parts.append(recommendation_dict['coral_health_summary'])
    
    if recommendation_dict.get('environmental_assessment'):
        parts.append(recommendation_dict['environmental_assessment'])
    
    if recommendation_dict.get('technique_name'):
        parts.append(f"Recommended action: {recommendation_dict['technique_name']}.")
    
    if recommendation_dict.get('reasoning'):
        parts.append(recommendation_dict['reasoning'])
    
    if recommendation_dict.get('next_steps'):
        steps = recommendation_dict['next_steps']
        if isinstance(steps, list):
            steps_text = ". Then, ".join(steps)
            parts.append(f"Next steps: {steps_text}.")
    
    english_text = " ".join(parts)
    
    # Translate and generate audio
    result = translate_and_speak(english_text)
    
    # Add original recommendation for frontend
    result['original_recommendation'] = recommendation_dict
    
    return result


if __name__ == "__main__":
    print("="*70)
    print("🇲🇺 REEFGUARDIAN AI - Kreol Assistant Test")
    print("="*70)
    
    # Test 1: Simple text
    print("\n📝 Test 1: Simple text")
    print("-"*70)
    result1 = translate_and_speak(
        "The coral is bleached due to high temperature. We recommend heat-resistant outplanting."
    )
    print(f"\n🇬🇧 English: {result1['english']}")
    print(f"🇲🇺 Kreol:   {result1['kreol']}")
    print(f"🔊 Audio:    {result1['audio_path']}")
    
    # Test 2: Full recommendation (simulating orchestrator output)
    print("\n\n📝 Test 2: Full recommendation")
    print("-"*70)
    sample_recommendation = {
        "alert_level": "HIGH",
        "coral_health_summary": "Bleaching detected at Blue Bay Marine Park.",
        "environmental_assessment": "Sea temperature is 30.4°C with severe thermal stress.",
        "technique_name": "Heat-Resistant Coral Outplanting",
        "reasoning": "Current conditions show critical thermal stress.",
        "next_steps": ["Deploy heat-resistant coral fragments", "Install shading structures"]
    }
    
    result2 = translate_recommendation(sample_recommendation)
    print(f"\n🇬🇧 English: {result2['english']}")
    print(f"🇲🇺 Kreol:   {result2['kreol']}")
    print(f"🔊 Audio:    {result2['audio_path']}")