"""
Quick check: lists which Gemini models your API key can actually call
generateContent on. Re-run this if a hardcoded model name 404s again.
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

print("Models your key can use with generateContent:\n")
for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print(" ", m.name)