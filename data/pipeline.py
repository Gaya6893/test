"""
Nutrition data pipeline.

  get_nutrition_by_label(label)   → USDA-backed nutrition_map.json lookup
  get_nutrition_by_barcode(code)  → Open Food Facts API (no key required)

Both return the shared contract dict or None.
Missing values are kept as None — never fabricated.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import requests

_MAP_PATH = Path(__file__).parent / "nutrition_map.json"
_nutrition_map: Optional[dict] = None


def _load_map() -> dict:
    global _nutrition_map
    if _nutrition_map is None:
        _nutrition_map = json.loads(_MAP_PATH.read_text(encoding="utf-8"))
    return _nutrition_map


# ── Raw-food / restaurant-food lookup ─────────────────────────────────────────
def get_nutrition_by_label(label: str) -> Optional[dict]:
    """
    Return shared-contract nutrition for a Food-101 label, or None if not found.
    Values are per 100 g serving.
    """
    entry = _load_map().get(label)
    if entry is None:
        return None
    return {
        "name":      entry.get("name"),
        "sodium_mg": entry.get("sodium_mg"),
        "sugar_g":   entry.get("sugar_g"),
        "carbs_g":   entry.get("carbs_g"),
        "calories":  entry.get("calories"),
        "fat_g":     entry.get("fat_g"),
    }


# ── Barcode / packaged-food lookup ────────────────────────────────────────────
_OFF_URL = "https://world.openfoodfacts.org/api/v0/product/{barcode}.json"
_HEADERS = {"User-Agent": "VitalScan-FoodAI/1.0 (gayamanukyan9@gmail.com)"}


def get_nutrition_by_barcode(barcode: str) -> Optional[dict]:
    """
    Query Open Food Facts by EAN/UPC barcode.
    Returns shared-contract nutrition per 100 g, or None if not found.
    """
    url = _OFF_URL.format(barcode=barcode.strip())
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return None

    if data.get("status") != 1:
        return None

    product    = data.get("product", {})
    nutriments = product.get("nutriments", {})

    def per100(key: str) -> Optional[float]:
        for candidate in (f"{key}_100g", key):
            val = nutriments.get(candidate)
            if val is not None:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    pass
        return None

    # Sodium: Open Food Facts stores in g/100g → convert to mg
    sodium_g  = per100("sodium")
    sodium_mg = round(sodium_g * 1000, 1) if sodium_g is not None else None

    # Calories: prefer kcal field; fall back to kJ ÷ 4.184
    calories = per100("energy-kcal")
    if calories is None:
        kj = per100("energy")
        if kj is not None:
            calories = round(kj / 4.184, 1)

    name = (
        product.get("product_name")
        or product.get("product_name_en")
        or product.get("generic_name")
    )

    return {
        "name":      name,
        "sodium_mg": sodium_mg,
        "sugar_g":   per100("sugars"),
        "carbs_g":   per100("carbohydrates"),
        "calories":  calories,
        "fat_g":     per100("fat"),
    }
