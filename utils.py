# ============================================================
#  utils.py — Loss functions, metrics, and helper utilities
# ============================================================

import os
import random
import numpy as np
import torch
import torch.nn.functional as F

from config import THRESHOLD


# ── Seed ────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"🌱 Seed set: {seed}")


# ── Loss ────────────────────────────────────────────────────

def dice_loss(pred, target, smooth=1.0):
    """Soft Dice loss — works with raw logits via sigmoid."""
    pred   = torch.sigmoid(pred).view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def bce_dice_loss(pred, target):
    """Combined BCE + Dice loss for segmentation."""
    if pred.shape != target.shape:
        pred = F.interpolate(
            pred.unsqueeze(1), size=target.shape[-2:],
            mode='bilinear', align_corners=False
        ).squeeze(1)
    bce  = F.binary_cross_entropy_with_logits(pred, target)
    dice = dice_loss(pred, target)
    return bce + dice


# ── Metrics ─────────────────────────────────────────────────

def compute_metrics(pred, target, threshold=THRESHOLD):
    """
    Compute IoU and Dice score from raw logits and binary target mask.
    Returns: (iou, dice) as Python floats
    """
    pred = torch.sigmoid(pred)
    if pred.shape != target.shape:
        pred = F.interpolate(
            pred.unsqueeze(1), size=target.shape[-2:],
            mode='bilinear', align_corners=False
        ).squeeze(1)

    pred_bin     = (pred > threshold).float()
    intersection = (pred_bin * target).sum()
    union        = (pred_bin + target).clamp(0, 1).sum()
    iou          = (intersection + 1e-6) / (union + 1e-6)
    dice         = (2 * intersection + 1e-6) / (pred_bin.sum() + target.sum() + 1e-6)
    return iou.item(), dice.item()


# ── VRAM ────────────────────────────────────────────────────

def print_vram():
    """Print current GPU memory usage."""
    if not torch.cuda.is_available():
        print("No GPU available")
        return
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved  = torch.cuda.memory_reserved()  / 1024**3
    total     = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"🖥️  VRAM — Allocated: {allocated:.1f}GB | "
          f"Reserved: {reserved:.1f}GB | "
          f"Free: {total - reserved:.1f}GB / {total:.1f}GB")


# ── Directory helpers ────────────────────────────────────────

def makedirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)