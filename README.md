# VitalScan — Food Recognition AI Module
**AIT 500 · Group 2** | Haotian Guan · Donja Hohendorff · Gayane Manukyan  
Westcliff University | Prof. Desmond Ademiluyi

---

## Project overview

Accepts two input types — a **food photograph** or a **product barcode** — and returns a structured nutrition JSON conforming to the shared contract consumed by Groups 3 and 4.

```json
{
  "name":      "Pizza",
  "sodium_mg": 598.0,
  "sugar_g":   3.0,
  "carbs_g":   33.0,
  "calories":  266.0,
  "fat_g":     10.0,
  "confidence": 0.82,
  "confidence_tier": "high"
}
```

---

## Directory structure

```
food-recognition-ai/
├── models/
│   ├── train.py        MobileNetV3-Large two-phase fine-tuning
│   ├── predict.py      Inference + three-tier confidence thresholding
│   └── evaluate.py     Top-1/5 accuracy, confusion matrix, worst-category report
├── data/
│   ├── nutrition_map.json   101 Food-101 labels → per-100 g nutrition (USDA-backed)
│   └── pipeline.py          Nutrition lookups (label map + Open Food Facts barcode API)
├── api/
│   ├── schemas.py      Pydantic models (shared contract + scan response)
│   └── main.py         FastAPI app — POST /scan/photo · POST /scan/barcode
├── clip_comparison/
│   └── clip_eval.py    CLIP ViT-B/32 zero-shot vs MobileNetV3 comparison
├── checkpoints/        .pt files written by train.py (git-ignored)
├── reports/            Accuracy reports + confusion matrix PNG
└── requirements.txt
```

---

## Setup

```bash
git clone <repo-url>
cd food-recognition-ai
pip install -r requirements.txt
```

---

## 1 · Training (Donja — run on Colab T4)

**Colab quick-start:**
```python
# Cell 1 — mount Drive and install deps
from google.colab import drive
drive.mount('/content/drive')
%pip install -q torch torchvision datasets tqdm

# Cell 2 — clone repo (or upload files)
%cd /content/drive/MyDrive/food-recognition-ai

# Cell 3 — train
!python -m models.train
```

**What it does:**
| Phase | Epochs | Layers trained | LR |
|-------|--------|----------------|----|
| 1     | 5      | Classifier head only | 1e-3 |
| 2     | 10     | Full network | 1e-4 |

Checkpoints are written to `checkpoints/`. The best epoch per phase overwrites `checkpoints/best_model.pt` automatically. Training history is saved to `checkpoints/training_history.json`.

**Expected runtime on T4:** ~6–8 hours total. Enable Colab's "Save to Drive" to survive session timeouts.

**Accuracy targets:** top-1 > 75%, top-5 > 92% on the Food-101 validation split.

---

## 2 · Evaluation (Haotian)

```bash
python -m models.evaluate checkpoints/best_model.pt
```

Outputs written to `reports/`:
- `classification_report.txt` — per-class precision / recall / F1
- `evaluation_summary.json` — top-1, top-5, worst-10 categories
- `confusion_matrix.png` — 101×101 heatmap

---

## 3 · API (Gayane)

### Run locally

```bash
uvicorn api.main:app --reload --port 8000
```

Auto-generated docs at **http://localhost:8000/docs**

### Endpoints

| Method | Path | Input | Description |
|--------|------|-------|-------------|
| GET | `/health` | — | Model readiness check |
| POST | `/scan/photo` | `multipart/form-data` — field `file` | Recognise food from image |
| POST | `/scan/barcode` | `{"barcode": "..."}` | Nutrition lookup by EAN/UPC |

### Confidence tiers (non-negotiable)

| Tier | Probability | Response |
|------|-------------|----------|
| `high` | > 0.60 | Nutrition fields populated |
| `medium` | 0.30 – 0.60 | Nutrition fields + `top_3_candidates` list |
| `low` | < 0.30 | `food_unrecognized: true`, nutrition fields null |

### Example requests

```bash
# Photo scan
curl -X POST http://localhost:8000/scan/photo \
  -F "file=@pizza.jpg"

# Barcode scan
curl -X POST http://localhost:8000/scan/barcode \
  -H "Content-Type: application/json" \
  -d '{"barcode": "0049000028911"}'
```

### Deployment (Week 4 — before integration deadline)

**Render (recommended):**
1. Push repo to GitHub
2. New Web Service → connect repo → Build: `pip install -r requirements.txt`
3. Start: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
4. Set `MODEL_CHECKPOINT` env var to the path of your uploaded checkpoint

**Railway / ngrok** are valid fallbacks if Render free-tier is unavailable.

---

## 4 · CLIP stretch comparison (Haotian)

```bash
# Evaluate CLIP on 5 000 val images (fast) and compare with MobileNetV3 results
python -m clip_comparison.clip_eval --subset 5000
```

Run `models/evaluate.py` first so the MobileNetV3 numbers are available. Results written to `clip_comparison/clip_results.json`.

---

## Key constraints

- **Shared contract field names are non-negotiable:** `name`, `sodium_mg`, `sugar_g`, `carbs_g`, `calories`, `fat_g`
- **Missing nutrition values must be `null`**, never fabricated
- **Food-101 cultural bias:** the dataset originates from North American and European online platforms; the writeup must explicitly acknowledge this limitation

---

## Technology stack

| Layer | Choice |
|-------|--------|
| Language | Python 3.10+ |
| Model | MobileNetV3-Large (torchvision) |
| Training framework | PyTorch + torchvision |
| Training data | Food-101 via HuggingFace Datasets |
| Nutrition (raw foods) | USDA FoodData Central (pre-curated JSON) |
| Nutrition (packaged) | Open Food Facts API (barcode path) |
| API framework | FastAPI + Pydantic v2 |
| Stretch goal | OpenAI CLIP ViT-B/32 via HuggingFace Transformers |
