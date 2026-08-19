from fastapi import FastAPI, File, UploadFile
from config import get_gemini_api_key, get_mistral_api_key
from google import genai
from google.genai import types
import json
import base64
import requests

app = FastAPI(title="Nutrition Detection API")

PROMPT = """You are a strict food nutrition extraction system.

Analyze the image carefully and estimate REAL PORTION SIZE based on visible quantity.

Return ONLY valid JSON. No explanation, no text.

Output format must be exactly:

{
"food_name": "",
"calories": 0,
"protein": 0,
"fats": 0,
"carbs": 0,
"serving": "",
"confidence": "high|medium|low",
"ingredients": [{"name": "", "portion": 0}]
}

Rules:
- Return ONLY json
- Estimate portion size in grams (VERY IMPORTANT)
- Use visual size (plate, pieces, thickness)
- If unsure, give best approximation
- Do NOT use generic "1 burger" only, always include grams
- If ingredients are visible, list them in array
- If not visible, return empty array []
- Do not include any extra fields
- U must give the json format answer only dont show any error 
- return ingredients with name and portion in small medium high or pieces etc like 1 piece or 1 cup etc if visible, otherwise return empty array
"""


def analyze_with_mistral(contents: bytes, mime_type: str, mistral_key: str):
    """Fallback function to analyze image with Mistral Pixtral Vision API."""
    base64_image = base64.b64encode(contents).decode("utf-8")
    
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {mistral_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "pixtral-12b-2409",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": PROMPT
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:{mime_type};base64,{base64_image}"
                    }
                ]
            }
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"}
    }
    
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    
    result_data = response.json()
    raw_content = result_data["choices"][0]["message"]["content"].strip()
    
    if raw_content.startswith("```"):
        raw_content = raw_content.replace("```json", "").replace("```", "").strip()
        
    return json.loads(raw_content)


@app.get("/")
def home():
    return {"message": "Welcome to the Nutrition Detection API!"}


@app.post("/analyze-food")
async def analyze_food(file: UploadFile = File(...)):
    contents = await file.read()
    mime_type = file.content_type or "image/jpeg"
    
    gemini_key = get_gemini_api_key()
    mistral_key = get_mistral_api_key()
    
    gemini_error = None

    # --- 1. TRY GEMINI API (PRIMARY) ---
    if gemini_key:
        try:
            client = genai.Client(api_key=gemini_key)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    types.Part.from_bytes(
                        data=contents,
                        mime_type=mime_type,
                    ),
                    PROMPT
                ],
                config=types.GenerateContentConfig(
                    temperature=0.2
                )
            )

            clean_text = response.text.strip()
            if clean_text.startswith("```"):
                clean_text = clean_text.replace("```json", "").replace("```", "").strip()

            return json.loads(clean_text)

        except Exception as e:
            gemini_error = str(e)
            print(f"⚠️ Gemini API failed/exceeded limit: {gemini_error}. Falling back to Mistral API...")
    else:
        gemini_error = "GEMINI_API_KEY is missing."

    # --- 2. TRY MISTRAL API (FALLBACK) ---
    if mistral_key:
        try:
            return analyze_with_mistral(contents, mime_type, mistral_key)
        except Exception as mistral_e:
            return {
                "error": "Both Gemini and Mistral APIs failed.",
                "gemini_error": gemini_error,
                "mistral_error": str(mistral_e)
            }
            
    return {
        "error": "Gemini API failed and MISTRAL_API_KEY is not set for fallback.",
        "gemini_error": gemini_error
    }
