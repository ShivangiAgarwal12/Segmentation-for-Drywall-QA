# Prompted Segmentation for Drywall QA

Text-conditioned image segmentation model that, given an image and a natural-language
prompt, produces a binary mask highlighting the region of interest.

**Supported prompts:**
- `"segment taping area"` → highlights drywall joints and seams
- `"segment crack"` → highlights wall cracks and surface fractures

---

## Results

| Class | mIoU | Dice | Samples |
|-------|------|------|---------|
| Taping | 0.5287 | 0.6789 | 250 |
| Crack | 0.5771 | 0.7216 | 248 |
| **Overall** | **0.5528** | **0.7002** | **498** |

> Best seed: **42** — trained across seeds 42, 123, 7

---

## Model

**CLIPSeg** (`CIDAS/clipseg-rd64-refined`) — a text-conditioned segmentation model
built on top of CLIP. The CLIP encoder is frozen and only the decoder is fine-tuned
on the drywall datasets.

| Component | Detail |
|-----------|--------|
| Architecture | CLIPSeg (ViT-B/16 + Transformer Decoder) |
| Base checkpoint | CIDAS/clipseg-rd64-refined |
| Trainable params | Decoder only (~45 MB) |
| Total model size | ~230 MB |
| Loss | BCE + Dice |
| Optimizer | AdamW (lr=1e-4, wd=1e-4) |
| Scheduler | CosineAnnealingLR |
| Image size | 352 × 352 |
| Epochs | 10 |
| Batch size | 16 |
| Seeds | 42, 123, 7 |

---

## Datasets

| Dataset | Source | Train | Valid | Total |
|---------|--------|-------|-------|-------|
| Drywall Join Detect | [Roboflow](https://universe.roboflow.com/objectdetect-pu6rn/drywall-join-detect) | 936 | 250 | 1,186 |
| Crack Detection | [Roboflow](https://universe.roboflow.com/university-bswxt/crack-bphdr) | 991 | 248 | 1,239 |
| **Combined** | | **1,927** | **498** | **2,425** |

> Both datasets were originally object-detection format (bounding boxes).
> Boxes were converted to filled binary masks.
> Dataset 2 had no validation split — manually split 80/20 with `seed=42`.

---

## Prompt Augmentation

Each training sample is randomly paired with one prompt from its class list
to prevent overfitting to a single phrasing:

```python
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
```

---

## Output Masks

- Format: PNG, single-channel, values `{0, 255}`
- Same spatial size as source image
- Naming: `{image_id}__{prompt_slug}.png`
- Example: `0001__segment_crack.png`

---

## Project Structure

```
drywall-qa/
├── main.py            # Entry point — controls all steps via flags
├── config.py          # All hyperparameters and paths in one place
├── dataset.py         # PyTorch Dataset class + DataLoader builder
├── model.py           # CLIPSeg loading and checkpoint handling
├── train.py           # Training loop (one seed, auto-resumes)
├── evaluate.py        # Per-class mIoU and Dice evaluation
├── inference.py       # Save prediction masks with random prompts
├── visualize.py       # Training curves + visual examples
├── requirements.txt
└── .gitignore
```

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your base path in `config.py`
```python
BASE = "/content/drive/MyDrive/drywall-qa"  # Google Colab
# BASE = "./data"                            # Local
```

### 3. Download datasets
```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")

# Dataset 1 — Taping
rf.workspace("objectdetect-pu6rn").project("drywall-join-detect").version(1).download(
    "coco", location=f"{BASE}/data/dataset1_taping_raw"
)

# Dataset 2 — Cracks
rf.workspace("university-bswxt").project("crack-bphdr").version(2).download(
    "coco", location=f"{BASE}/data/dataset2_cracks_raw"
)
```

### 4. Run everything
```python
# In main.py — set which steps to run
RUN_TRAINING   = True
RUN_EVALUATION = True
RUN_INFERENCE  = True
RUN_VISUALIZE  = True

# Then run
python main.py
```

---

## Running in Google Colab

This project was trained on **Google Colab (NVIDIA T4 GPU)**. To use the scripts
inside Colab:

```python
import sys
sys.path.append('/content/drive/MyDrive/drywall-qa')

from config import *
from train import train_one_seed
from evaluate import evaluate, print_results
# etc.
```

Or run the full pipeline:
```python
exec(open('/content/drive/MyDrive/drywall-qa/main.py').read())
```

---

## Resuming After a Crash

Training auto-resumes from the last saved epoch. If your session dies mid-training,
just re-run — the code detects the saved `history.json` and `clipseg_last.pt` and
picks up where it left off.

To skip completed seeds, update `SEEDS` in `config.py`:
```python
# Seed 42 done, resume from 123
SEEDS = [123, 7]
```

---

## Reproducibility

All random seeds are set at the start of every training run:

```python
SEED = 42  # also tested: 123, 7
random.seed(SEED)
numpy.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
```

Each seed saves to its own folder: `checkpoints/seed_{seed}/`

---

## Runtime

| Metric | Value |
|--------|-------|
| Training device | NVIDIA T4 (Google Colab) |
| Training time per seed | ~35–40 minutes |
| Total training (3 seeds) | ~110 minutes |
| Avg inference time/image | ~0.18 seconds |
| Checkpoint size | ~45 MB (decoder only) |

---

## Failure Cases

- **Hairline cracks missed** — CLIPSeg's internal 64×64 resolution loses very fine structures
- **Taping under-segmented** — subtle joints with no visible texture are hard to localise
- **Bounding box noise** — both datasets had boxes not pixel masks, introducing label noise
- **High-res downsampling** — Dataset 2 is 2560×1440 originally; resizing loses fine detail

---

## References

- Lüddecke & Ecker (2022). *Image Segmentation Using Text and Image Prompts*. CVPR.
- Radford et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision*. ICML.
- [CLIPSeg on Hugging Face](https://huggingface.co/CIDAS/clipseg-rd64-refined)
- [Drywall Join Detect — Roboflow](https://universe.roboflow.com/objectdetect-pu6rn/drywall-join-detect)
- [Crack Detection — Roboflow](https://universe.roboflow.com/university-bswxt/crack-bphdr)