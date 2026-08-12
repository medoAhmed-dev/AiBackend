import os
import time
from pathlib import Path

import jwt
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

load_dotenv()

from api.chat import router as chat_router  # noqa: E402  (must load .env first)

app = FastAPI(title="Aether Threads AI Backend")

app.include_router(chat_router)

STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/test")
def test_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "test.html")


@app.get("/dev/token")
def dev_token(patient_id: str = "a1b2c3d4-e5f6-7890-abcd-ef1234567890") -> dict:
    """Local-only helper: mints a test JWT so the /test page can call /chat
    without you having to hand-generate one. Refuses to run unless
    ENVIRONMENT=development — never enable this in a deployed environment."""
    if os.getenv("ENVIRONMENT") != "development":
        raise HTTPException(status_code=404)
    token = jwt.encode(
        {"sub": patient_id, "exp": int(time.time()) + 3600},
        os.getenv("SUPABASE_JWT_SECRET"),
        algorithm="HS256",
    )
    return {"token": token}
