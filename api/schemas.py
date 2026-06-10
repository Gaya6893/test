"""
Pydantic models for the VitalScan Food Recognition API.

NutritionResponse carries the six shared-contract fields consumed by
Groups 3 and 4.  Field names are non-negotiable per the project brief.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


# ── Request bodies ─────────────────────────────────────────────────────────────
class BarcodeScanRequest(BaseModel):
    barcode: str = Field(..., example="0049000028911", description="EAN-13 or UPC-A barcode")


# ── Shared contract (Groups 3 & 4 consume exactly these field names) ───────────
class NutritionResponse(BaseModel):
    name:      Optional[str]   = Field(None, description="Human-readable food name")
    sodium_mg: Optional[float] = Field(None, description="Sodium in milligrams per 100 g")
    sugar_g:   Optional[float] = Field(None, description="Sugar in grams per 100 g")
    carbs_g:   Optional[float] = Field(None, description="Total carbohydrates in grams per 100 g")
    calories:  Optional[float] = Field(None, description="Energy in kcal per 100 g")
    fat_g:     Optional[float] = Field(None, description="Total fat in grams per 100 g")


# ── Candidate label for medium-confidence responses ───────────────────────────
class Candidate(BaseModel):
    label: str
    score: float


# ── Canonical 8-field contract (consumed by Groups 3 & 4) ────────────────────
class ContractResponse(NutritionResponse):
    confidence:      Optional[float] = None
    confidence_tier: Optional[str]   = None


# ── Full scan response (superset of shared contract) ──────────────────────────
class ScanResponse(NutritionResponse):
    """
    Extends NutritionResponse with recognition metadata.

    confidence_tier values:
      "high"   (>0.60)  — nutrition fields populated, no candidates list
      "medium" (0.30–0.60) — nutrition fields populated, top_3_candidates present
      "low"    (<0.30)  — food_unrecognized=True, nutrition fields all null
    """
    confidence:        Optional[float]          = None
    confidence_tier:   Optional[str]            = None
    top_3_candidates:  Optional[List[Candidate]] = None
    food_unrecognized: Optional[bool]           = None
    source:            Optional[str]            = None
