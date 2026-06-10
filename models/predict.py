"""
Food recognition inference with three-tier confidence thresholding.

Confidence tiers (from project brief — non-negotiable):
  high   > 0.60  → return nutrition directly
  medium 0.30–0.60 → return nutrition + top_3_candidates
  low    < 0.30  → return food_unrecognized=True, skip nutrition
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

# ── Constants ─────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

HIGH_CONF   = 0.60
MEDIUM_CONF = 0.30

_PREPROCESS = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Alphabetical Food-101 class list (fallback when class_names.json is absent)
FOOD101_CLASSES: list[str] = [
    "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
    "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
    "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
    "ceviche", "cheese_plate", "cheesecake", "chicken_curry", "chicken_quesadilla",
    "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder",
    "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes",
    "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict",
    "escargots", "falafel", "filet_mignon", "fish_and_chips", "foie_gras",
    "french_fries", "french_onion_soup", "french_toast", "fried_calamari", "fried_rice",
    "frozen_yogurt", "garlic_bread", "gnocchi", "greek_salad", "grilled_cheese_sandwich",
    "grilled_salmon", "guacamole", "gyoza", "hamburger", "hot_and_sour_soup",
    "hot_dog", "huevos_rancheros", "hummus", "ice_cream", "lasagna",
    "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese", "macarons", "miso_soup",
    "mussels", "nachos", "omelette", "onion_rings", "oysters",
    "pad_thai", "paella", "pancakes", "panna_cotta", "peking_duck",
    "pho", "pizza", "pork_chop", "poutine", "prime_rib",
    "pulled_pork_sandwich", "ramen", "ravioli", "red_velvet_cake", "risotto",
    "samosa", "sashimi", "scallops", "seaweed_salad", "shrimp_and_grits",
    "spaghetti_bolognese", "spaghetti_carbonara", "spring_rolls", "steak", "strawberry_shortcake",
    "sushi", "tacos", "takoyaki", "tiramisu", "tuna_tartare", "waffles",
]

# ── Module-level model cache ───────────────────────────────────────────────────
_model:       Optional[nn.Module]      = None
_class_names: Optional[list[str]]     = None
_device:      Optional[torch.device]  = None


# ── Public API ────────────────────────────────────────────────────────────────
def load_model(checkpoint_path: str | Path) -> None:
    """Load a .pt checkpoint into the module-level cache. Call once at startup."""
    global _model, _class_names, _device

    checkpoint_path = Path(checkpoint_path)
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Class names from checkpoint directory (written by train.py)
    names_file = checkpoint_path.parent / "class_names.json"
    _class_names = (
        json.loads(names_file.read_text()) if names_file.exists() else FOOD101_CLASSES
    )

    m = models.mobilenet_v3_large(weights=None)
    in_features = m.classifier[3].in_features
    m.classifier[3] = nn.Linear(in_features, len(_class_names))

    ckpt = torch.load(checkpoint_path, map_location=_device)
    m.load_state_dict(ckpt["model_state_dict"])
    m.to(_device).eval()
    _model = m
    print(f"Model loaded from {checkpoint_path} on {_device}")


def predict(image: Image.Image | str | Path, top_k: int = 3) -> dict:
    """
    Run inference on a single image.

    Returns a dict with confidence_tier and relevant fields:
      high   → {label, confidence, confidence_tier, nutrition_lookup_key}
      medium → same + top_3_candidates list
      low    → {food_unrecognized: True, confidence, confidence_tier, top_3_candidates}
    """
    if _model is None:
        raise RuntimeError("Call load_model() before predict().")

    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")
    else:
        image = image.convert("RGB")

    tensor = _PREPROCESS(image).unsqueeze(0).to(_device)

    with torch.no_grad():
        logits = _model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    top_probs, top_idxs = probs.topk(top_k)
    top_probs = top_probs.cpu().tolist()
    top_idxs  = top_idxs.cpu().tolist()

    best_prob  = top_probs[0]
    best_label = _class_names[top_idxs[0]]
    candidates = [
        {"label": _class_names[i], "score": round(p, 4)}
        for i, p in zip(top_idxs, top_probs)
    ]

    if best_prob >= HIGH_CONF:
        return {
            "label":               best_label,
            "confidence":          round(best_prob, 4),
            "confidence_tier":     "high",
            "nutrition_lookup_key": best_label,
        }

    if best_prob >= MEDIUM_CONF:
        return {
            "label":               best_label,
            "confidence":          round(best_prob, 4),
            "confidence_tier":     "medium",
            "nutrition_lookup_key": best_label,
            "top_3_candidates":    candidates,
        }

    return {
        "food_unrecognized":  True,
        "confidence":         round(best_prob, 4),
        "confidence_tier":    "low",
        "top_3_candidates":   candidates,
    }
