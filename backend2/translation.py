# backend/translation.py
"""
ReefGuardian AI - English to Mauritian Creole Translator
Uses Gemma 4 via Ollama for offline translation.
"""

import ollama


def translate_to_kreol(english_text):
    """
    Translates English text to Mauritian Creole.
    
    Args:
        english_text: English string to translate
    
    Returns:
        Mauritian Creole string
    """
    
    if not english_text or not english_text.strip():
        return ""
    
    prompt = f"""Translate the following English text into Mauritian Creole (Kreol Morisien).

RULES:
- Use authentic Mauritian Creole (French-based creole)
- Keep scientific/species names unchanged (e.g., "Acropora", "Blue Bay")
- Use simple language for local fishermen and marine officers
- Do NOT add explanations, just the translation

REFERENCE VOCABULARY:
- coral = koray
- ocean = losean
- reef = resif
- bleached = blansi
- heat = lasaler
- water = delo
- recommend = rekomande
- immediate = imediat
- fisherman = peser
- marine = marin

TEXT:
{english_text}

KREOL TRANSLATION:"""

    try:
        response = ollama.chat(
            model='gemma4',
            messages=[{"role": "user", "content": prompt}],
            options={'temperature': 0.3, 'num_predict': 1000}
        )
        
        translation = response['message']['content'].strip()
        
        # Clean markdown if Gemma adds it
        if translation.startswith('```'):
            lines = translation.split('\n')
            translation = '\n'.join(lines[1:-1]) if len(lines) > 2 else translation
        
        return translation
    
    except Exception as e:
        print(f"❌ Translation error: {e}")
        return f"[Translation error: {str(e)[:50]}]"


if __name__ == "__main__":
    # Quick test
    test = "The coral is bleached due to high temperature. We recommend heat-resistant outplanting."
    print(f"English: {test}")
    print(f"Kreol:   {translate_to_kreol(test)}")