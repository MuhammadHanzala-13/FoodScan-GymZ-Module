from fastapi import FastAPI, File, UploadFile
from config import get_gemini_api_key, MISTRAL_API_KEY
from google import genai
from google.genai import types
import json

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Welcome to the Nutrition Detection API!"}


@app.post("/analyze-food")
async def analyze_food(file: UploadFile = File(...)):
    api_key = get_gemini_api_key()
    if not api_key:
        return {"error": "GEMINI_API_KEY is missing. Please set it in your .env file or environment variables."}

    contents = await file.read()

    prompt = """You are a strict food nutrition extraction system.

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
-U must give the json format answer only dont show any error 
-return ingredients with name and portion in small medium high or pieces etc like 1 piece or 1 cup etc if visible, otherwise return empty array
"""

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=[
                types.Part.from_bytes(
                    data=contents,
                    mime_type=file.content_type,
                ),
                prompt
            ],
            config=types.GenerateContentConfig(
                temperature=0.3
            )
        )

        clean_text = response.text.strip()

        if clean_text.startswith("```"):
            clean_text = clean_text.replace("```json", "").replace("```", "").strip()

        return json.loads(clean_text)

    except Exception as e:
        return {"error": str(e)}
