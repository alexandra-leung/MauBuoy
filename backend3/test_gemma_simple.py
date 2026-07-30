# backend3/test_gemma_simple.py
import ollama

print("Testing if Gemma 4 responds...")

try:
    response = ollama.chat(
        model='gemma4',
        messages=[{"role": "user", "content": "Say hello"}],
        options={'num_predict': 10}
    )
    
    if response['message']['content']:
        print(f"✅ Gemma works! Response: {response['message']['content']}")
    else:
        print("❌ Gemma returned empty response")
        
except Exception as e:
    print(f"❌ Error: {e}")