import base64
import json
import os
import time

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException
from openai import OpenAI
from pydantic import BaseModel
import requests

app = FastAPI(title="ALPR Qwen3-VL Proxy")

# Load environment variables configured on Render
QWEN_API_KEY = os.getenv("QWEN_API_KEY")
SECRET_CLIENT_TOKEN = os.getenv("SECRET_CLIENT_TOKEN")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

qwen_client = (
  OpenAI(
    api_key=QWEN_API_KEY,
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
  )
  if QWEN_API_KEY
  else None
)


class PlateRequest(BaseModel):
  image_base64: str


def parse_plate_response(response_text: str) -> dict:
  text = response_text.strip()
  if text.startswith("```"):
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
      lines = lines[1:]
    if lines and lines[-1].strip() == "```":
      lines = lines[:-1]
    text = "\n".join(lines).strip()

  plate_data = json.loads(text)
  if not isinstance(plate_data, dict):
    raise ValueError("Qwen returned JSON that is not an object")
  return plate_data


def log_plate_to_supabase(plate_text: str) -> None:
  if not SUPABASE_URL or not SUPABASE_KEY:
    return

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


@app.get("/")
def health_check():
  return {"status": "online", "message": "ALPR Proxy is running"}


@app.post("/process-plate")
def process_plate(
    background_tasks: BackgroundTasks,
    payload: PlateRequest,
    x_client_token: str = Header(None, alias="X-Client-Token"),
):
  request_started = time.perf_counter()

  # 1. Security Check: Reject callers who do not have the client secret
  if not SECRET_CLIENT_TOKEN or x_client_token != SECRET_CLIENT_TOKEN:
    raise HTTPException(status_code=401, detail="Unauthorized client token")

  if not QWEN_API_KEY or not qwen_client:
    raise HTTPException(
        status_code=500, detail="Qwen API key missing on server"
    )

  try:
    # 2. Validate the JPEG data while preserving it for direct forwarding.
    image_started = time.perf_counter()
    encoded_image = payload.image_base64.split(",", 1)[-1]
    base64.b64decode(encoded_image, validate=True)
    image_processing_ms = (time.perf_counter() - image_started) * 1000

    # 3. Fast direct request to Qwen API
    qwen_started = time.perf_counter()
    response = qwen_client.chat.completions.create(
      model="qwen3-vl-flash",
      messages=[
        {
          "role": "user",
          "content": [
            {
              "type": "image_url",
              "image_url": {
                "url": f"data:image/jpeg;base64,{encoded_image}"
              },
            },
            {
              "type": "text",
              "text": (
                "Extract license plate details from this plate image. "
                "Return ONLY a valid JSON object with these keys: "
                "- 'number': license plate number without code, "
                "- 'code': plate code/prefix (or null if none), "
                "- 'state_or_region': state, emirate, or region, not country (or null if none), "
                "- 'country': country name (or null if none), "
                "- 'type': vehicle type (or null if unknown), "
                "- 'brand': vehicle brand (or null if unknown), "
                "- 'model': vehicle model (or null if unknown), "
                "- 'color': vehicle color (or null if unknown)."
              ),
            },
          ],
        }
      ],
      extra_body={"enable_thinking": False},
    )
    qwen_ms = (time.perf_counter() - qwen_started) * 1000

    response_text = response.choices[0].message.content or ""
    if not response_text:
      raise ValueError("Qwen returned an empty response")
    plate_data = parse_plate_response(response_text)

    plate_text = (
        f"Number:        {plate_data.get('number')}\n"
        f"Code:          {plate_data.get('code')}\n"
        f"Country:       {plate_data.get('country')}\n"
        f"State/Region:  {plate_data.get('state_or_region')}\n"
        f"Type:          {plate_data.get('type')}\n"
        f"Brand:         {plate_data.get('brand')}\n"
        f"Model:         {plate_data.get('model')}\n"
        f"Color:         {plate_data.get('color')}\n\n"
        "Metrics\n"
    ) if plate_data.get("number") else "UNKNOWN"

    action = "OPEN" if plate_text != "UNKNOWN" else "DO NOT OPEN"

    # 4. Log after returning the gate decision to the device.
    background_tasks.add_task(log_plate_to_supabase, plate_text)
    server_total_ms = (time.perf_counter() - request_started) * 1000

    return {
        "action": action,
        "plate": plate_text,
        "message": (
            "Access Granted" if action == "OPEN" else "Plate Unrecognized"
        ),
          "timings": {
            "image_processing_ms": round(image_processing_ms),
            "qwen_ms": round(qwen_ms),
            "server_total_ms": round(server_total_ms),
          },
    }

  except (ValueError, OSError) as error:
    raise HTTPException(status_code=400, detail=str(error)) from error
  except HTTPException:
    raise
  except Exception as error:
    raise HTTPException(
        status_code=502, detail=f"Plate processing service failed: {error}"
    ) from error
