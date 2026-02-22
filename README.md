# Prompted Segmentation for Drywall QA

## Overview
A text-conditioned segmentation model fine-tuned on drywall inspection images.
Given an image and a natural-language prompt, the model produces a binary mask
highlighting the region of interest.

Supported prompts:
- `segment taping area` → highlights drywall joints and seams
- `segment crack`       → highlights wall cracks and fractures

---

## Model
- **Architecture:** CLIPSeg (CIDAS/clipseg-rd64-refined)
- **Base model:** CLIP ViT-B/16 encoder + lightweight CNN decoder
- **Strategy:** Frozen CLIP encoder, fine-tuned decoder only
- **Loss:** BCE + Dice combined loss
- **Optimizer:** AdamW (lr=1e-4, weight_decay=1e-4)
- **Scheduler:** CosineAnnealingLR
- **Epochs:** 10
- **Batch size:** 8
- **Image size:** 352×352
- **Device:** NVIDIA T4 GPU (Google Colab)

---

## Reproducibility
All random seeds set to **42** at the start of training:
```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
```

---

## Datasets

| Dataset | Source | Task | Split |
|---|---|---|---|
| Drywall Join Detect | roboflow: objectdetect-pu6rn | Taping Area | Train: 936, Valid: 250 |
| Crack Detection | roboflow: university-bswxt | Cracks | Train: 991, Valid: 248 |

**Note:** Both datasets were originally object-detection format (bounding boxes).
Bounding boxes were converted to binary masks using filled rectangle regions.
Dataset 2 had no validation split — manually split 80/20 using sklearn with seed=42.

---

## Results

| Class   | mIoU   | Dice   | Samples |
|---------|--------|--------|---------|
| Taping  | 0.5218 | 0.6727 | 250     |
| Crack   | 0.5676 | 0.7139 | 248     |
| Overall | 0.5446 | 0.6932 | 498     |

**Best validation IoU:** 0.5468

---

## Output Masks
- Format: PNG, single-channel, values {0, 255}
- Same spatial size as source image
- Naming: `{image_id}__{prompt}.png`
- Example: `IMG_001__segment_crack.png`

---

## Project Structure
```
drywall-qa/
├── data/
│   ├── dataset1_taping_raw/       # Raw Roboflow download
│   ├── dataset1_taping/           # Converted masks
│   ├── dataset2_cracks_raw/       # Raw Roboflow download
│   └── dataset2_cracks/           # Converted masks + split
├── checkpoints/
│   ├── clipseg_best.pt            # Best model weights
│   └── training_curves.png        # Loss/IoU/Dice plots
└── predictions/
    ├── taping/                    # Taping prediction masks
    ├── cracks/                    # Crack prediction masks
    ├── taping_examples.png        # Visual examples
    └── crack_examples.png         # Visual examples
```

---

## Requirements
```
torch
torchvision
transformers
roboflow
opencv-python-headless
albumentations
scikit-learn
matplotlib
tqdm
pillow
```

---

## Runtime & Footprint
| Metric | Value |
|---|---|
| Training time | ~35 minutes (T4 GPU) |
| Avg inference time | ~0.18 sec/image |
| Model size (clipseg-rd64-refined) | ~230 MB |
| Checkpoint size (decoder only) | ~45 MB |

---

## Failure Cases
- Thin hairline cracks are sometimes missed (low contrast)
- Taping areas with no visible seam texture are under-segmented
- Bounding box masks as GT introduce noise — pixel-level annotations would improve results

