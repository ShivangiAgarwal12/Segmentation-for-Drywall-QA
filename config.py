#  config.py — All hyperparameters and paths in one place
#  Change anything here; the rest of the code reads from this


import os

# Reproducibility
SEEDS = [42, 123, 7]          # Seeds to train across

# Model
MODEL_NAME = "CIDAS/clipseg-rd64-refined"
IMAGE_SIZE = 352               # Input resolution fed to CLIPSeg

# Training
EPOCHS      = 10
BATCH_SIZE  = 16
LR          = 1e-4
WEIGHT_DECAY= 1e-4
THRESHOLD   = 0.5              # Sigmoid threshold for binary mask

# Prompts 
PROMPTS = {
    'taping': [
        "segment taping area",
        "segment joint tape",
        "segment drywall seam",
        "segment wall joint"
    ],
    'crack': [
        "segment crack",
        "segment wall crack",
        "segment surface crack",
        "segment fracture"
    ]
}

# Paths
BASE = os.environ.get("DRYWALL_BASE", "/content/drive/MyDrive/drywall-qa")

# Dataset 1 — Taping
D1_TRAIN_IMGS  = f"{BASE}/data/dataset1_taping_raw/train"
D1_TRAIN_MASKS = f"{BASE}/data/dataset1_taping/masks/train"
D1_VALID_IMGS  = f"{BASE}/data/dataset1_taping_raw/valid"
D1_VALID_MASKS = f"{BASE}/data/dataset1_taping/masks/valid"

# Dataset 2 — Cracks
D2_TRAIN_IMGS  = f"{BASE}/data/dataset2_cracks/images/train"
D2_TRAIN_MASKS = f"{BASE}/data/dataset2_cracks/masks/train_final"
D2_VALID_IMGS  = f"{BASE}/data/dataset2_cracks/images/valid"
D2_VALID_MASKS = f"{BASE}/data/dataset2_cracks/masks/valid"

# Output directories
CHECKPOINTS_DIR = f"{BASE}/checkpoints"
PREDICTIONS_DIR = f"{BASE}/predictions"