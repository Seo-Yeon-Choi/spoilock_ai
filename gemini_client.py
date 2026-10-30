"""Gemini connection shared by server and event extraction."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def get_client():
    from dotenv import dotenv_values
    from google import genai
    settings = dotenv_values(ROOT / '.env', encoding='utf-8-sig')
    # Explicit .env values win to avoid accidentally selecting a different shell key/project.
    key = (settings.get('GEMINI_API_KEY') or os.getenv('GEMINI_API_KEY', '')).strip()
    model = (settings.get('GEMINI_MODEL') or 'gemini-3.5-flash-lite').strip()
    if not key or key == 'PASTE_YOUR_KEY_HERE':
        raise ValueError('Set GEMINI_API_KEY in the .env in the project root')
    return genai.Client(api_key=key, http_options={'timeout': 60000}), model
