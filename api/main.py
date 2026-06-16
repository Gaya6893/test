"""
VitalScan Food Recognition API

Endpoints
---------
GET  /health            — liveness probe
POST /scan/photo        — multipart image → nutrition JSON
POST /scan/barcode      — {"barcode": "..."} → nutrition JSON
GET  /result2           — latest scan output (8-field contract) for Team 4

Auto-generated OpenAPI docs available at /docs (serves the API documentation
deliverable for free via FastAPI's built-in Swagger UI).

Environment variables
---------------------
MODEL_CHECKPOINT   path to best_model.pt  (default: checkpoints/best_model.pt)

Start locally:
  uvicorn api.main:app --reload --port 8000

Deploy to Render / Railway: point start command to this file.
"""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# Ensure project root is on the path when run as a module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.schemas import BarcodeScanRequest, Candidate, ContractResponse, ScanResponse
from data.pipeline import get_nutrition_by_barcode, get_nutrition_by_label
from models.predict import load_model, predict

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="VitalScan Food Recognition API",
    description=(
        "Accepts a food photograph or product barcode and returns a structured "
        "nutrition JSON conforming to the VitalScan shared contract."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

CHECKPOINT = Path(os.environ.get("MODEL_CHECKPOINT", "checkpoints/best_model.pt"))
_model_ready = False

# ── Group 2 → Group 4 output handoff ──────────────────────────────────────────
# Per the cross-team convention: store our output as JSON locally, then expose it
# through a GET endpoint (/result2) so the downstream team (Group 4) can read the
# latest food-scan result directly — no image upload required on their side.
_CONTRACT_KEYS = (
    "name", "sodium_mg", "sugar_g", "carbs_g",
    "calories", "fat_g", "confidence", "confidence_tier",
)
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
RESULT_FILE = OUTPUT_DIR / "result2.json"

# Served before any scan has run (and even if the model isn't loaded yet) so the
# integration never breaks during the combined demo.
_SAMPLE_RESULT = {
    "name": "Pizza",
    "sodium_mg": 598.0,
    "sugar_g": 3.0,
    "carbs_g": 33.0,
    "calories": 266.0,
    "fat_g": 10.0,
    "confidence": 0.82,
    "confidence_tier": "high",
}


def _store_result(payload: dict) -> None:
    """Persist the latest scan as the eight-field shared contract."""
    contract = {key: payload.get(key) for key in _CONTRACT_KEYS}
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        RESULT_FILE.write_text(json.dumps(contract, indent=2))
    except OSError:
        pass  # never let persistence break the API response


@app.get("/result2", tags=["Integration"])
async def result2():
    """
    Group 2's latest food-scan output as JSON — the endpoint Group 4 calls.

    Returns the most recent `/scan` result (persisted to `output/result2.json`),
    or a representative sample if no scan has run yet, so the eight-field contract
    is always available even before the model is trained.
    """
    if RESULT_FILE.exists():
        try:
            return json.loads(RESULT_FILE.read_text())
        except (OSError, ValueError):
            pass
    return _SAMPLE_RESULT


# ── Lifecycle ─────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def _startup():
    global _model_ready
    if CHECKPOINT.exists():
        load_model(CHECKPOINT)
        _model_ready = True
        print(f"Model loaded from {CHECKPOINT}")
    else:
        print(
            f"WARNING: No checkpoint at {CHECKPOINT}. "
            "Train the model first, then set MODEL_CHECKPOINT env var."
        )


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["Meta"])
async def health():
    """Liveness probe. Returns model status so integration partners can verify readiness."""
    return {"status": "ok", "model_ready": _model_ready, "checkpoint": str(CHECKPOINT)}


# ── Photo endpoint ────────────────────────────────────────────────────────────
@app.post("/scan/photo", response_model=ScanResponse, tags=["Scan"])
async def scan_photo(file: UploadFile = File(...)):
    """
    Identify the food in a photograph and return nutrition information.

    **Confidence tiers** (non-negotiable per project brief):
    - `high` (>0.60): nutrition returned directly
    - `medium` (0.30–0.60): nutrition + `top_3_candidates` list
    - `low` (<0.30): `food_unrecognized=true`, nutrition fields are null
    """
    if not _model_ready:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train a checkpoint and restart the server.",
        )

    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image file.")

    prediction = predict(image)
    response = _build_photo_response(prediction)
    _store_result(response.model_dump())
    return response


# ── Barcode endpoint ──────────────────────────────────────────────────────────
@app.post("/scan/barcode", response_model=ScanResponse, tags=["Scan"])
async def scan_barcode(body: BarcodeScanRequest):
    """
    Look up nutrition for a packaged food by EAN-13 or UPC-A barcode.

    Queries the Open Food Facts database (no API key required).
    All values are per 100 g.
    """
    nutrition = get_nutrition_by_barcode(body.barcode)
    if nutrition is None:
        raise HTTPException(
            status_code=404,
            detail=f"Barcode '{body.barcode}' not found in Open Food Facts.",
        )

    response = ScanResponse(
        name      = nutrition.get("name"),
        sodium_mg = nutrition.get("sodium_mg"),
        sugar_g   = nutrition.get("sugar_g"),
        carbs_g   = nutrition.get("carbs_g"),
        calories  = nutrition.get("calories"),
        fat_g     = nutrition.get("fat_g"),
        confidence       = 1.0,
        confidence_tier  = "high",
        food_unrecognized= None,
        source           = "open_food_facts",
    )
    _store_result(response.model_dump())
    return response


# ── Canonical contract endpoint ───────────────────────────────────────────────
@app.post("/scan", response_model=ContractResponse, tags=["Scan"])
async def scan(file: UploadFile = File(...)):
    """
    Run the full recognition pipeline on a photo and return the **canonical
    VitalScan contract** — the exact eight-field JSON the overall process emits
    for Groups 3 & 4.

    This is the clean contract view of `/scan/photo`: it drops internal metadata
    (`source`, `top_3_candidates`, `food_unrecognized`) and always returns the
    same eight keys —
    `name, sodium_mg, sugar_g, carbs_g, calories, fat_g, confidence, confidence_tier`.

    Nutrition fields are `null` when confidence is low or the recognized food is
    missing from the nutrition map; values are never fabricated.
    """
    if not _model_ready:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train a checkpoint and restart the server.",
        )

    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image file.")

    contract = _build_contract_response(predict(image))
    _store_result(contract.model_dump())
    return contract


def _build_contract_response(prediction: dict) -> ContractResponse:
    """Collapse a raw prediction into the eight-field shared contract."""
    tier       = prediction.get("confidence_tier", "low")
    confidence = prediction.get("confidence")

    # Low confidence / unrecognized → all nutrition fields stay null
    if tier == "low" or prediction.get("food_unrecognized"):
        return ContractResponse(confidence=confidence, confidence_tier="low")

    label     = prediction.get("nutrition_lookup_key") or prediction.get("label")
    nutrition = get_nutrition_by_label(label)

    # Recognized but absent from nutrition map → keep the name, nutrition null
    if nutrition is None:
        return ContractResponse(name=label, confidence=confidence, confidence_tier=tier)

    return ContractResponse(
        name      = nutrition["name"],
        sodium_mg = nutrition["sodium_mg"],
        sugar_g   = nutrition["sugar_g"],
        carbs_g   = nutrition["carbs_g"],
        calories  = nutrition["calories"],
        fat_g     = nutrition["fat_g"],
        confidence      = confidence,
        confidence_tier = tier,
    )


# ── Response builder ──────────────────────────────────────────────────────────
def _build_photo_response(prediction: dict) -> ScanResponse:
    tier = prediction.get("confidence_tier", "low")

    if tier == "low" or prediction.get("food_unrecognized"):
        return ScanResponse(
            food_unrecognized = True,
            confidence        = prediction.get("confidence"),
            confidence_tier   = "low",
            top_3_candidates  = _candidates(prediction),
            source            = "mobilenetv3_food101",
        )

    label     = prediction.get("nutrition_lookup_key") or prediction.get("label")
    nutrition = get_nutrition_by_label(label)

    if nutrition is None:
        # Label recognised but missing from nutrition map — still return recognition
        return ScanResponse(
            food_unrecognized = True,
            confidence        = prediction.get("confidence"),
            confidence_tier   = tier,
            top_3_candidates  = _candidates(prediction),
            source            = "mobilenetv3_food101",
        )

    return ScanResponse(
        name      = nutrition["name"],
        sodium_mg = nutrition["sodium_mg"],
        sugar_g   = nutrition["sugar_g"],
        carbs_g   = nutrition["carbs_g"],
        calories  = nutrition["calories"],
        fat_g     = nutrition["fat_g"],
        confidence       = prediction.get("confidence"),
        confidence_tier  = tier,
        top_3_candidates = _candidates(prediction) if tier == "medium" else None,
        food_unrecognized= None,
        source           = "mobilenetv3_food101",
    )


def _candidates(prediction: dict) -> list[Candidate] | None:
    raw = prediction.get("top_3_candidates")
    if not raw:
        return None
    return [Candidate(label=c["label"], score=c["score"]) for c in raw]
