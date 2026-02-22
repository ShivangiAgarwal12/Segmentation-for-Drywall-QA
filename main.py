#  main.py — Entry point for Drywall QA Segmentation
#
#  Usage (Google Colab):
#    Run each section by setting the RUN_* flags below


import os
import json
import torch

# Config 
from config import (
    SEEDS, EPOCHS, BATCH_SIZE, LR, CHECKPOINTS_DIR, PREDICTIONS_DIR,
    D1_VALID_IMGS, D1_VALID_MASKS,
    D2_VALID_IMGS, D2_VALID_MASKS,
)

#  Modules
from utils      import set_seed, makedirs
from model      import load_model
from dataset    import get_dataloaders
from train      import train_one_seed
from evaluate   import evaluate, print_results
from inference  import save_predictions
from visualize  import plot_training_curves, visualize_random_examples

# Flags — set True/False to control what runs 
RUN_TRAINING   = True
RUN_EVALUATION = True
RUN_INFERENCE  = True
RUN_VISUALIZE  = True



DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")
makedirs(CHECKPOINTS_DIR, PREDICTIONS_DIR)



#  TRAIN ACROSS ALL SEEDS

all_results = {}

if RUN_TRAINING:
    for seed in SEEDS:
        history, best_iou = train_one_seed(
            seed=seed,
            device=DEVICE,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LR
        )
        all_results[seed] = {'history': history, 'best_iou': best_iou}

    # Print summary across seeds
    print("\n" + "=" * 50)
    print("ALL SEEDS — TRAINING SUMMARY")
    print("=" * 50)
    for seed, res in all_results.items():
        print(f"  Seed {seed:>4} → Best IoU: {res['best_iou']:.4f}")

    best_seed = max(all_results, key=lambda s: all_results[s]['best_iou'])
    print(f"\nBest seed: {best_seed}  (IoU: {all_results[best_seed]['best_iou']:.4f})")
else:
    # Find best seed from saved checkpoints
    best_seed = None
    best_iou  = 0.0
    for seed in SEEDS:
        hist_path = os.path.join(CHECKPOINTS_DIR, f"seed_{seed}", "history.json")
        if os.path.exists(hist_path):
            with open(hist_path) as f:
                h = json.load(f)
            seed_best = max(h['val_iou']) if h['val_iou'] else 0.0
            print(f"  Seed {seed} — Best IoU from history: {seed_best:.4f}")
            if seed_best > best_iou:
                best_iou  = seed_best
                best_seed = seed
    print(f"\n Best seed found: {best_seed}  (IoU: {best_iou:.4f})")



#  EVALUATE BEST SEED

if RUN_EVALUATION and best_seed is not None:
    best_ckpt = os.path.join(CHECKPOINTS_DIR, f"seed_{best_seed}", "clipseg_best.pt")
    processor, model = load_model(DEVICE, checkpoint_path=best_ckpt)

    # Build validation dataloader
    from config import (D1_TRAIN_IMGS, D1_TRAIN_MASKS,
                        D2_TRAIN_IMGS, D2_TRAIN_MASKS)
    _, valid_loader = get_dataloaders(
        D1_TRAIN_IMGS, D1_TRAIN_MASKS,
        D1_VALID_IMGS, D1_VALID_MASKS,
        D2_TRAIN_IMGS, D2_TRAIN_MASKS,
        D2_VALID_IMGS, D2_VALID_MASKS,
        batch_size=BATCH_SIZE
    )

    results  = evaluate(model, processor, valid_loader, DEVICE)
    metrics  = print_results(results, seed=best_seed)



#  SAVE PREDICTION MASKS

if RUN_INFERENCE and best_seed is not None:
    if not RUN_EVALUATION:
        # Load model if evaluation was skipped
        best_ckpt = os.path.join(CHECKPOINTS_DIR, f"seed_{best_seed}", "clipseg_best.pt")
        processor, model = load_model(DEVICE, checkpoint_path=best_ckpt)

    taping_meta = save_predictions(
        model, processor,
        imgs_dir   = D1_VALID_IMGS,
        masks_dir  = D1_VALID_MASKS,
        label      = 'taping',
        output_dir = os.path.join(PREDICTIONS_DIR, 'taping'),
        device     = DEVICE
    )

    crack_meta = save_predictions(
        model, processor,
        imgs_dir   = D2_VALID_IMGS,
        masks_dir  = D2_VALID_MASKS,
        label      = 'crack',
        output_dir = os.path.join(PREDICTIONS_DIR, 'cracks'),
        device     = DEVICE
    )


#  VISUALIZE

if RUN_VISUALIZE:
    # Training curves for all seeds
    plot_training_curves(seeds=SEEDS)

    # Load metadata if inference was skipped
    if not RUN_INFERENCE:
        with open(os.path.join(PREDICTIONS_DIR, 'taping', 'metadata.json')) as f:
            taping_meta = json.load(f)
        with open(os.path.join(PREDICTIONS_DIR, 'cracks', 'metadata.json')) as f:
            crack_meta = json.load(f)

    # 4 random taping examples
    visualize_random_examples(
        imgs_dir  = D1_VALID_IMGS,
        masks_dir = D1_VALID_MASKS,
        pred_dir  = os.path.join(PREDICTIONS_DIR, 'taping'),
        meta      = taping_meta,
        label     = 'taping',
        n         = 4
    )

    # 4 random crack examples
    visualize_random_examples(
        imgs_dir  = D2_VALID_IMGS,
        masks_dir = D2_VALID_MASKS,
        pred_dir  = os.path.join(PREDICTIONS_DIR, 'cracks'),
        meta      = crack_meta,
        label     = 'crack',
        n         = 4
    )

print("\n All done!")