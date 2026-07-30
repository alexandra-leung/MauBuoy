# backend/tts.py
"""
ReefGuardian AI - Text-to-Speech for Mauritian Creole
Uses edge-tts with French voice (closest to Kreol pronunciation).
"""

import asyncio
import edge_tts
import os
import time


# French voice (best for Kreol pronunciation)
# Other options: "fr-FR-DeniseNeural" (female), "fr-FR-HenriNeural" (male)
KREOL_VOICE = "fr-FR-DeniseNeural"

AUDIO_DIR = os.path.join(os.path.dirname(__file__), 'audio')
os.makedirs(AUDIO_DIR, exist_ok=True)


async def _generate_audio_async(text, output_path, rate="+0%", pitch="+0Hz"):
    """Async helper for edge-tts."""
    communicate = edge_tts.Communicate(text, KREOL_VOICE, rate=rate, pitch=pitch)
    await communicate.save(output_path)


def text_to_speech(text, filename=None, rate="+0%"):
    """
    Converts text to speech audio file.
    
    Args:
        text: Text to speak (Kreol or any language)
        filename: Optional custom filename (without extension)
        rate: Speech rate (e.g., "-10%" slower, "+10%" faster)
    
    Returns:
        Path to generated audio file (MP3)
    """
    
    if not text or not text.strip():
        return None
    
    # Generate unique filename
    if filename is None:
        filename = f"response_{int(time.time())}"
    
    output_path = os.path.join(AUDIO_DIR, f"{filename}.mp3")
    
    try:
        # Run async function
        asyncio.run(_generate_audio_async(text, output_path, rate=rate))
        print(f"✅ Audio saved: {output_path}")
        return output_path
    
    except Exception as e:
        print(f"❌ TTS error: {e}")
        return None


if __name__ == "__main__":
    # Quick test
    test_kreol = "Koray-la finn blansi akoz la chalè dan delo."
    print(f"Text: {test_kreol}")
    audio_path = text_to_speech(test_kreol, filename="test_kreol")
    print(f"Audio: {audio_path}")