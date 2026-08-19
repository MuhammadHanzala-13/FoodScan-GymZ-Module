from dotenv import load_dotenv
import os

load_dotenv()

def get_gemini_api_key():
    load_dotenv(override=True)
    return os.getenv("GEMINI_API_KEY")

def get_mistral_api_key():
    load_dotenv(override=True)
    return os.getenv("MISTRAL_API_KEY")

GEMINI_API_KEY = get_gemini_api_key()
MISTRAL_API_KEY = get_mistral_api_key()