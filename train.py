#  train.py — Training loop for one seed


import os
import json
import time
import torch
import torchvision.transforms as T
from tqdm import tqdm

from config  import EPOCHS, LR, WEIGHT_DECAY, CHECKPOINTS_DIR
from utils   import bce_dice_loss, compute_metrics, set_seed, print_vram
from model   import load_model
from dataset import get_dataloaders

from config import (
    D1_TRAIN_IMGS, D1_TRAIN_MASKS, D1_VALID_IMGS, D1_VALID_MASKS,
    D2_TRAIN_IMGS, D2_TRAIN_MASKS, D2_VALID_IMGS, D2_VALID_MASKS,
    BATCH_SIZE, IMAGE_SIZE
)


def train_one_seed(seed, device, epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE):
    """
    Full training run for a single seed.
    Saves best and last checkpoints + history.json after every epoch.
    Automatically resumes if a previous run was interrupted.

    Returns:
        history  : dict with train_loss, val_loss, val_iou, val_dice, epoch_time
        best_iou : float
    """
    set_seed(seed)

    # Paths for this seed
    ckpt_dir  = os.path.join(CHECKPOINTS_DIR, f"seed_{seed}")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, "clipseg_best.pt")
    last_path = os.path.join(ckpt_dir, "clipseg_last.pt")
    hist_path = os.path.join(ckpt_dir, "history.json")

    print(f"\n{'='*60}")
    print(f"Training — SEED {seed} | Epochs: {epochs} | LR: {lr} | Batch: {batch_size}")
    print(f"{'='*60}")

    # DataLoaders
    train_loader, valid_loader = get_dataloaders(
        D1_TRAIN_IMGS, D1_TRAIN_MASKS,
        D1_VALID_IMGS, D1_VALID_MASKS,
        D2_TRAIN_IMGS, D2_TRAIN_MASKS,
        D2_VALID_IMGS, D2_VALID_MASKS,
        batch_size=batch_size, image_size=IMAGE_SIZE
    )

    # Load fresh model
    checkpoint = last_path if os.path.exists(last_path) else None
    processor, model = load_model(device, checkpoint_path=checkpoint)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Resume history if exists
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            history = json.load(f)
        start_epoch = len(history['train_loss'])
        best_iou    = max(history['val_iou']) if history['val_iou'] else 0.0
        print(f"Resuming from epoch {start_epoch + 1} (best IoU: {best_iou:.4f})")
    else:
        history = {
            'train_loss': [], 'val_loss': [],
            'val_iou'   : [], 'val_dice': [],
            'epoch_time': []
        }
        start_epoch = 0
        best_iou    = 0.0

    # Training loop
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        print_vram()

        # Train
        model.train()
        train_loss = 0.0
        for images, prompts, masks, _ in tqdm(
                train_loader, desc=f"Seed {seed} | Epoch {epoch+1}/{epochs} [Train]"):

            images, masks = images.to(device), masks.to(device)
            inputs = processor(
                text=list(prompts),
                images=[T.ToPILImage()(img.cpu()) for img in images],
                return_tensors="pt", padding=True
            ).to(device)

            loss = bce_dice_loss(model(**inputs).logits, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = val_iou = val_dice = 0.0
        with torch.no_grad():
            for images, prompts, masks, _ in tqdm(
                    valid_loader, desc=f"Seed {seed} | Epoch {epoch+1}/{epochs} [Valid]"):

                images, masks = images.to(device), masks.to(device)
                inputs = processor(
                    text=list(prompts),
                    images=[T.ToPILImage()(img.cpu()) for img in images],
                    return_tensors="pt", padding=True
                ).to(device)

                logits     = model(**inputs).logits
                iou, dice  = compute_metrics(logits, masks)
                val_loss  += bce_dice_loss(logits, masks).item()
                val_iou   += iou
                val_dice  += dice

        val_loss /= len(valid_loader)
        val_iou  /= len(valid_loader)
        val_dice /= len(valid_loader)
        scheduler.step()

        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), best_path)
            print(f"Best model saved! IoU: {best_iou:.4f}")

        
        torch.save(model.state_dict(), last_path)

        # Saves history after every epoch
        epoch_time = time.time() - t0
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)
        history['val_dice'].append(val_dice)
        history['epoch_time'].append(epoch_time)

        with open(hist_path, 'w') as f:
            json.dump(history, f, indent=2)

        print(f"Epoch {epoch+1:02d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val IoU: {val_iou:.4f} | "
              f"Val Dice: {val_dice:.4f} | "
              f"Time: {epoch_time:.1f}s")

    print(f"\n Seed {seed} done! Best IoU: {best_iou:.4f}")
    return history, best_iou