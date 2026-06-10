"""
MobileNetV3-Large fine-tuning on Food-101.
Two-phase strategy:
  Phase 1 — 5 epochs, head-only, lr=1e-3
  Phase 2 — 10 epochs, full network, lr=1e-4

Run on Colab T4 GPU (batch_size=32, ~6–8 h total).
Checkpoints are written to ./checkpoints/ after each phase's best val-acc1 epoch.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as T
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── Hyper-parameters ──────────────────────────────────────────────────────────
BATCH_SIZE     = 32
IMG_SIZE       = 224
PHASE1_EPOCHS  = 5
PHASE2_EPOCHS  = 10
NUM_CLASSES    = 101

IMAGENET_MEAN  = [0.485, 0.456, 0.406]
IMAGENET_STD   = [0.229, 0.224, 0.225]

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

# ── Transforms ────────────────────────────────────────────────────────────────
train_transforms = T.Compose([
    T.RandomResizedCrop(IMG_SIZE),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_transforms = T.Compose([
    T.Resize(256),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ── Dataset wrapper ───────────────────────────────────────────────────────────
class Food101Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_split, transform=None):
        self.data = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = item["image"].convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, item["label"]


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    m = models.mobilenet_v3_large(weights="IMAGENET1K_V2")
    in_features = m.classifier[3].in_features      # 1280
    m.classifier[3] = nn.Linear(in_features, num_classes)
    return m


def freeze_backbone(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True


def unfreeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


# ── Training / evaluation loops ───────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = correct = total = 0
    for images, labels in tqdm(loader, leave=False, desc="train"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(images)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct    += (out.argmax(1) == labels).sum().item()
        total      += images.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = correct1 = correct5 = total = 0
    for images, labels in tqdm(loader, leave=False, desc="val  "):
        images, labels = images.to(device), labels.to(device)
        out  = model(images)
        loss = criterion(out, labels)
        total_loss += loss.item() * images.size(0)
        _, top5 = out.topk(5, dim=1)
        correct1 += (out.argmax(1) == labels).sum().item()
        correct5 += top5.eq(labels.unsqueeze(1)).any(1).sum().item()
        total    += images.size(0)
    return total_loss / total, correct1 / total, correct5 / total


# ── Checkpointing ─────────────────────────────────────────────────────────────
def save_checkpoint(model, optimizer, epoch, acc1, acc5, phase, class_names) -> Path:
    path = CHECKPOINT_DIR / f"mobilenetv3_phase{phase}_ep{epoch:02d}_acc{acc1:.4f}.pt"
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "acc1": acc1,
        "acc5": acc5,
        "class_names": class_names,
    }, path)
    # Keep a stable "best" symlink-equivalent: just copy the filename to best_model.pt
    best = CHECKPOINT_DIR / "best_model.pt"
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "acc1": acc1,
        "acc5": acc5,
        "class_names": class_names,
    }, best)
    print(f"  ✓ Checkpoint saved: {path}  (best_model.pt updated)")
    return path


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading Food-101 from HuggingFace Datasets …")
    ds = load_dataset("ethz/food101")
# ── CPU quick-run: comment out these two lines for full GPU training ──
    ds["train"]      = ds["train"].select(range(2000))
    ds["validation"] = ds["validation"].select(range(500))
    class_names = ds["train"].features["label"].names
    print(f"  {len(class_names)} classes | train={len(ds['train'])} | val={len(ds['validation'])}")

    with open(CHECKPOINT_DIR / "class_names.json", "w") as f:
        json.dump(class_names, f)

    train_ds = Food101Dataset(ds["train"],      transform=train_transforms)
    val_ds   = Food101Dataset(ds["validation"], transform=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    model     = build_model(len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    history   = []

    # ── Phase 1: head-only ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Phase 1 — head-only | {PHASE1_EPOCHS} epochs | lr=1e-3")
    print(f"{'='*60}")
    freeze_backbone(model)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
    )

    best1, best_path1 = 0.0, None
    for ep in range(1, PHASE1_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc        = train_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc1, vl_acc5 = eval_epoch(model, val_loader,   criterion, device)
        elapsed = time.time() - t0
        print(f"  Epoch {ep:2d}/{PHASE1_EPOCHS} | "
              f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val loss={vl_loss:.4f} acc1={vl_acc1:.4f} acc5={vl_acc5:.4f} | "
              f"{elapsed:.0f}s")
        history.append({"phase": 1, "epoch": ep,
                        "tr_loss": tr_loss, "tr_acc": tr_acc,
                        "vl_loss": vl_loss, "vl_acc1": vl_acc1, "vl_acc5": vl_acc5})
        if vl_acc1 > best1:
            best1 = vl_acc1
            best_path1 = save_checkpoint(model, optimizer, ep, vl_acc1, vl_acc5, 1, class_names)

    # ── Phase 2: full network ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Phase 2 — full network | {PHASE2_EPOCHS} epochs | lr=1e-4")
    print(f"{'='*60}")
    unfreeze_all(model)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best2, best_path2 = 0.0, None
    for ep in range(1, PHASE2_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc        = train_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc1, vl_acc5 = eval_epoch(model, val_loader,   criterion, device)
        elapsed = time.time() - t0
        print(f"  Epoch {ep:2d}/{PHASE2_EPOCHS} | "
              f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val loss={vl_loss:.4f} acc1={vl_acc1:.4f} acc5={vl_acc5:.4f} | "
              f"{elapsed:.0f}s")
        history.append({"phase": 2, "epoch": ep,
                        "tr_loss": tr_loss, "tr_acc": tr_acc,
                        "vl_loss": vl_loss, "vl_acc1": vl_acc1, "vl_acc5": vl_acc5})
        if vl_acc1 > best2:
            best2 = vl_acc1
            best_path2 = save_checkpoint(model, optimizer, ep, vl_acc1, vl_acc5, 2, class_names)

    print(f"\nBest Phase 1: {best_path1}  (acc1={best1:.4f})")
    print(f"Best Phase 2: {best_path2}  (acc1={best2:.4f})")

    with open(CHECKPOINT_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("Training complete — history saved to checkpoints/training_history.json")


if __name__ == "__main__":
    main()
