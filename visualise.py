# ============================================================
#  visualize.py — Plot training curves and prediction examples
# ============================================================

import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from config import PREDICTIONS_DIR, CHECKPOINTS_DIR


def plot_training_curves(seeds, save_path=None):
    """
    Load history.json for each seed and plot Loss / IoU / Dice curves.
    """
    colors = ['steelblue', 'darkorange', 'seagreen', 'crimson', 'mediumpurple']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Training History — All Seeds', fontsize=14, fontweight='bold')

    for i, seed in enumerate(seeds):
        hist_path = os.path.join(CHECKPOINTS_DIR, f"seed_{seed}", "history.json")
        if not os.path.exists(hist_path):
            print(f"⚠️  No history found for seed {seed} — skipping")
            continue

        with open(hist_path) as f:
            h = json.load(f)

        epochs = range(1, len(h['train_loss']) + 1)
        color  = colors[i % len(colors)]
        label  = f"Seed {seed}"

        axes[0].plot(epochs, h['train_loss'], label=f"{label} Train",
                     color=color, linestyle='--', alpha=0.7)
        axes[0].plot(epochs, h['val_loss'],   label=f"{label} Val",
                     color=color, linestyle='-')
        axes[1].plot(epochs, h['val_iou'],    label=label,
                     color=color, marker='o', markersize=3)
        axes[2].plot(epochs, h['val_dice'],   label=label,
                     color=color, marker='o', markersize=3)

    for ax, title, ylabel in zip(
        axes,
        ['Loss (BCE + Dice)', 'Validation mIoU', 'Validation Dice'],
        ['Loss', 'IoU', 'Dice']
    ):
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    out = save_path or os.path.join(CHECKPOINTS_DIR, "all_seeds_curves.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Training curves saved → {out}")


def visualize_random_examples(imgs_dir, masks_dir, pred_dir,
                               meta, label, n=4, save_path=None):
    """
    Display n random examples: Original | Ground Truth | Prediction.
    Uses saved metadata to match images with their saved masks.
    """
    sampled = random.sample(meta, min(n, len(meta)))

    fig, axes = plt.subplots(n, 3, figsize=(13, n * 4))
    fig.suptitle(f'Random Examples — {label.upper()}',
                 fontsize=14, fontweight='bold')

    plotted = 0
    for entry in sampled:
        img_file  = entry['img_file']
        prompt    = entry['prompt']
        mask_name = entry['mask_name']

        img_path  = os.path.join(imgs_dir, img_file)
        mask_path = os.path.join(masks_dir, os.path.splitext(img_file)[0] + '.png')
        pred_path = os.path.join(pred_dir,  mask_name)

        if not all(os.path.exists(p) for p in [img_path, mask_path, pred_path]):
            continue

        img  = Image.open(img_path).convert('RGB')
        gt   = Image.open(mask_path)
        pred = Image.open(pred_path)

        axes[plotted][0].imshow(img)
        axes[plotted][0].set_title(f'Original\nPrompt: "{prompt}"', fontsize=9)
        axes[plotted][1].imshow(gt,   cmap='gray')
        axes[plotted][1].set_title('Ground Truth', fontsize=9)
        axes[plotted][2].imshow(pred, cmap='gray')
        axes[plotted][2].set_title('Prediction', fontsize=9)

        for ax in axes[plotted]:
            ax.axis('off')

        plotted += 1

    # Hide unused rows
    for j in range(plotted, n):
        for ax in axes[j]:
            ax.axis('off')

    plt.tight_layout()

    out = save_path or os.path.join(PREDICTIONS_DIR, f"{label}_random_examples.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Saved {plotted} examples → {out}")