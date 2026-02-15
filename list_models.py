import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("Error: GOOGLE_API_KEY not found in environment.")
else:
    genai.configure(api_key=api_key)
    print("Successfully configured GenAI API key.")
    
    print("\nListing available models:")
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"Model: {m.name}, Methods: {m.supported_generation_methods}")
    except Exception as e:
        print(f"Error listing models: {e}")
