"""
Stretch goal — CLIP zero-shot evaluation on Food-101.

Compares openai/clip-vit-base-patch32 (zero-shot) against the fine-tuned
MobileNetV3 checkpoint and writes a side-by-side JSON report.

Usage:
  python -m clip_comparison.clip_eval
  python -m clip_comparison.clip_eval --checkpoint checkpoints/best_model.pt --subset 5000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
BATCH_SIZE    = 64
OUT_PATH      = Path("clip_comparison") / "clip_results.json"


# ── Text prompt builder ────────────────────────────────────────────────────────
def build_prompts(class_names: list[str]) -> list[str]:
    """
    "a photo of apple pie" style prompts — standard CLIP zero-shot pattern.
    Underscores → spaces for readability.
    """
    return [f"a photo of {label.replace('_', ' ')}" for label in class_names]


# ── CLIP zero-shot ─────────────────────────────────────────────────────────────
@torch.no_grad()
def run_clip(class_names: list[str], subset_size: Optional[int] = None) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {CLIP_MODEL_ID} on {device} …")

    model     = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    model.eval()

    # Pre-compute text embeddings (101 classes, done once)
    prompts     = build_prompts(class_names)
    text_inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
    text_feats  = model.get_text_features(**text_inputs)
    text_feats  = text_feats / text_feats.norm(dim=-1, keepdim=True)   # (101, 512)

    print("Loading Food-101 validation split …")
    ds = load_dataset("ethz/food101", split="validation")
    if subset_size:
        ds = ds.select(range(min(subset_size, len(ds))))
    print(f"  Evaluating on {len(ds)} images")

    correct1 = correct5 = total = 0
    per_class_correct  = np.zeros(len(class_names), dtype=int)
    per_class_total    = np.zeros(len(class_names), dtype=int)

    for i in tqdm(range(0, len(ds), BATCH_SIZE), desc="CLIP inference"):
        batch   = ds.select(range(i, min(i + BATCH_SIZE, len(ds))))
        images  = [item["image"].convert("RGB") for item in batch]
        labels  = [item["label"]               for item in batch]

        img_inputs  = processor(images=images, return_tensors="pt").to(device)
        img_feats   = model.get_image_features(**img_inputs)
        img_feats   = img_feats / img_feats.norm(dim=-1, keepdim=True)   # (B, 512)

        sim         = img_feats @ text_feats.T                           # (B, 101)
        top1_preds  = sim.argmax(dim=1).cpu().tolist()
        _, top5_idx = sim.topk(5, dim=1)
        top5_idx    = top5_idx.cpu().tolist()

        for j, true_lbl in enumerate(labels):
            correct1 += int(top1_preds[j] == true_lbl)
            correct5 += int(true_lbl in top5_idx[j])
            per_class_total[true_lbl]   += 1
            per_class_correct[true_lbl] += int(top1_preds[j] == true_lbl)
        total += len(labels)

    acc1 = correct1 / total
    acc5 = correct5 / total

    # Worst 10 classes for CLIP
    per_class_acc = {
        class_names[i]: (per_class_correct[i] / per_class_total[i]
                         if per_class_total[i] > 0 else 0.0)
        for i in range(len(class_names))
    }
    worst10_clip = dict(sorted(per_class_acc.items(), key=lambda x: x[1])[:10])

    print(f"\nCLIP Zero-Shot  (n={total})")
    print(f"  Top-1 : {acc1:.4f}  ({acc1*100:.2f}%)")
    print(f"  Top-5 : {acc5:.4f}  ({acc5*100:.2f}%)")

    return {
        "model":           CLIP_MODEL_ID,
        "n_samples":       total,
        "clip_acc1":       float(acc1),
        "clip_acc5":       float(acc5),
        "worst10_clip":    worst10_clip,
    }


# ── MobileNetV3 accuracy (read from evaluation_summary.json if present) ────────
def _load_mobilenet_results() -> Optional[dict]:
    summary = Path("reports") / "evaluation_summary.json"
    if not summary.exists():
        return None
    with open(summary) as f:
        return json.load(f)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="CLIP zero-shot vs MobileNetV3 comparison")
    parser.add_argument("--subset", type=int, default=5000,
                        help="Number of val images to evaluate (default 5000; set 0 for all)")
    args = parser.parse_args()
    subset = args.subset or None

    # Class names from dataset
    ds_meta     = load_dataset("ethz/food101", split="train[:1%]")
    class_names = ds_meta.features["label"].names

    clip_results = run_clip(class_names, subset_size=subset)

    # Merge with MobileNetV3 results if available
    mobilenet = _load_mobilenet_results()
    report = {**clip_results}
    if mobilenet:
        report["mobilenetv3_acc1"]     = mobilenet.get("acc1")
        report["mobilenetv3_acc5"]     = mobilenet.get("acc5")
        report["mobilenetv3_checkpoint"] = mobilenet.get("checkpoint")
        report["delta_acc1"]           = (
            (mobilenet["acc1"] - clip_results["clip_acc1"])
            if mobilenet.get("acc1") else None
        )
        print(f"\nMobileNetV3 Fine-tuned")
        print(f"  Top-1 : {mobilenet['acc1']:.4f}  ({mobilenet['acc1']*100:.2f}%)")
        print(f"  Top-5 : {mobilenet['acc5']:.4f}  ({mobilenet['acc5']*100:.2f}%)")
        delta = mobilenet["acc1"] - clip_results["clip_acc1"]
        print(f"\nFine-tuning gain over CLIP zero-shot: {delta:+.4f} ({delta*100:+.2f}pp)")
    else:
        print("\nNo MobileNetV3 evaluation_summary.json found. "
              "Run models/evaluate.py first to include the comparison.")

    OUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nComparison report saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
