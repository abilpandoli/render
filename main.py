import base64
import io
import os
from fastapi import FastAPI, Header, HTTPException
from google import genai
from PIL import Image
from pydantic import BaseModel
import requests

app = FastAPI(title="ALPR Gemini Proxy")

# Load environment variables configured on Render
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SECRET_CLIENT_TOKEN = os.getenv("SECRET_CLIENT_TOKEN")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

gemini_client = (
    genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
)


class PlateRequest(BaseModel):
  image_base64: str


@app.get("/")
def health_check():
  return {"status": "online", "message": "ALPR Proxy is running"}


@app.post("/process-plate")
def process_plate(
    payload: PlateRequest,
    x_client_token: str = Header(None, alias="X-Client-Token"),
):
  # 1. Security Check: Reject callers who do not have the client secret
  if not SECRET_CLIENT_TOKEN or x_client_token != SECRET_CLIENT_TOKEN:
    raise HTTPException(status_code=401, detail="Unauthorized client token")

  if not GEMINI_API_KEY or not gemini_client:
    raise HTTPException(
        status_code=500, detail="Gemini API key missing on server"
    )

  try:
    # 2. Convert base64 string back to an image
    image_bytes = base64.b64decode(payload.image_base64)
    image = Image.open(io.BytesIO(image_bytes))

    # 3. Fast direct request to Gemini API
    response = gemini_client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=[
            "Extract license plate details from this plate image."
            "Return ONLY a valid JSON object with these keys."
            '- "number": license plate number'
            '- "code": plate code/prefix (or null if none)'
            '- "state_or_region": state, emirate, or region, not country (or null if none)'
            '- "country": country name (or null if none)'
            '- "type": vehicle type (or null if unknown)'
            '- "brand": vehicle brand (or null if unknown)'
            '- "model": vehicle model (or null if unknown)'
            '- "color": vehicle color (or null if unknown)',
            image,
        ],
    )

    plate_text = response.text.strip() if response.text else "UNKNOWN"
    action = "OPEN" if plate_text != "UNKNOWN" else "DO NOT OPEN"

    # 4. Asynchronous log to Supabase directly from Render backend
    if SUPABASE_URL and SUPABASE_KEY:
      try:
        requests.post(
            f"{SUPABASE_URL}/rest/v1/trial_plates",
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal",
            },
            json={"plate_ocr": plate_text},
            timeout=3,
        )
      except Exception as log_err:
        print(f"Supabase logging error: {log_err}")

    return {
        "action": action,
        "plate": plate_text,
        "message": (
            "Access Granted" if action == "OPEN" else "Plate Unrecognized"
        ),
    }

  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))
