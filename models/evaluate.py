"""
Evaluation harness for the Food-101 fine-tuned MobileNetV3 checkpoint.

Produces:
  reports/classification_report.txt  — per-class precision/recall/F1
  reports/evaluation_summary.json    — top-1, top-5, worst categories
  reports/confusion_matrix.png       — 101×101 heatmap

Usage:
  python -m models.evaluate checkpoints/best_model.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torchvision.models as models
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.train import Food101Dataset, val_transforms

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)


# ── Model loader ──────────────────────────────────────────────────────────────
def load_model(checkpoint_path: Path, num_classes: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = models.mobilenet_v3_large(weights=None)
    in_features = m.classifier[3].in_features
    m.classifier[3] = nn.Linear(in_features, num_classes)
    ckpt = torch.load(checkpoint_path, map_location=device)
    m.load_state_dict(ckpt["model_state_dict"])
    return m.to(device).eval(), device


# ── Core evaluation ───────────────────────────────────────────────────────────
@torch.no_grad()
def run_evaluation(checkpoint_path: str | Path) -> dict:
    checkpoint_path = Path(checkpoint_path)
    print("Loading Food-101 validation split …")
    ds          = load_dataset("ethz/food101")
    class_names = ds["train"].features["label"].names

    val_ds     = Food101Dataset(ds["validation"], transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    model, device = load_model(checkpoint_path, len(class_names))

    all_labels: list[int] = []
    all_preds:  list[int] = []
    all_top5:   list[list[int]] = []

    for images, labels in tqdm(val_loader, desc="Evaluating"):
        images = images.to(device)
        out    = model(images)
        probs  = torch.softmax(out, dim=1)
        top1   = probs.argmax(dim=1).cpu().numpy()
        _, t5  = probs.topk(5, dim=1)
        t5     = t5.cpu().numpy()

        all_labels.extend(labels.numpy())
        all_preds.extend(top1)
        all_top5.extend(t5.tolist())

    labels_arr = np.array(all_labels)
    preds_arr  = np.array(all_preds)

    acc1 = (preds_arr == labels_arr).mean()
    acc5 = np.mean([labels_arr[i] in all_top5[i] for i in range(len(labels_arr))])

    print(f"\nTop-1 Accuracy : {acc1:.4f}  ({acc1*100:.2f}%)")
    print(f"Top-5 Accuracy : {acc5:.4f}  ({acc5*100:.2f}%)")
    print(f"Target: top-1 > 0.75, top-5 > 0.92")

    # Per-class report
    report_dict = classification_report(
        labels_arr, preds_arr, target_names=class_names, output_dict=True, zero_division=0
    )
    class_f1 = {cls: report_dict[cls]["f1-score"] for cls in class_names if cls in report_dict}
    worst_10 = sorted(class_f1.items(), key=lambda x: x[1])[:10]

    print("\nWorst 10 categories (by F1-score):")
    for cls, f1 in worst_10:
        print(f"  {cls:<35s}  F1={f1:.4f}")

    # Text report
    report_txt = REPORTS_DIR / "classification_report.txt"
    with open(report_txt, "w", encoding="utf-8") as f:
        f.write(f"Checkpoint : {checkpoint_path}\n")
        f.write(f"Top-1 Acc  : {acc1:.4f}\n")
        f.write(f"Top-5 Acc  : {acc5:.4f}\n\n")
        f.write(classification_report(labels_arr, preds_arr,
                                      target_names=class_names, zero_division=0))
    print(f"\nClassification report → {report_txt}")

    # Confusion matrix PNG
    _plot_confusion_matrix(labels_arr, preds_arr, class_names)

    # JSON summary
    summary = {
        "checkpoint": str(checkpoint_path),
        "acc1": float(acc1),
        "acc5": float(acc5),
        "meets_top1_target": bool(acc1 > 0.75),
        "meets_top5_target": bool(acc5 > 0.92),
        "worst_categories": dict(worst_10),
    }
    summary_path = REPORTS_DIR / "evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Evaluation summary  → {summary_path}")
    return summary


# ── Confusion matrix plot ─────────────────────────────────────────────────────
def _plot_confusion_matrix(labels: np.ndarray, preds: np.ndarray, class_names: list[str]):
    cm  = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(30, 28))
    sns.heatmap(
        cm, annot=False, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True",      fontsize=10)
    ax.set_title("Food-101 Confusion Matrix — MobileNetV3-Large", fontsize=12)
    plt.xticks(rotation=90, fontsize=4)
    plt.yticks(rotation=0,  fontsize=4)
    plt.tight_layout()
    out = REPORTS_DIR / "confusion_matrix.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Confusion matrix    → {out}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Food-101 checkpoint")
    parser.add_argument("checkpoint", help="Path to .pt checkpoint file")
    args = parser.parse_args()
    run_evaluation(args.checkpoint)
