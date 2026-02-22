# ============================================================
#  inference.py — Save prediction masks with random prompts
# ============================================================

import os
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

from config import PROMPTS, PREDICTIONS_DIR, THRESHOLD


def save_predictions(model, processor, imgs_dir, masks_dir,
                     label, output_dir, device, threshold=THRESHOLD):
    """
    Run inference on all images in imgs_dir.
    - Randomly picks a prompt from PROMPTS[label] for each image
    - Saves binary masks as {numeric_id}__{prompt_slug}.png
    - Saves metadata.json with prompt used per image

    Returns: list of metadata dicts
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    img_files = sorted([f for f in os.listdir(imgs_dir)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    saved_meta = []

    for idx, img_file in enumerate(tqdm(img_files, desc=f"Saving {label} masks")):
        img_id    = os.path.splitext(img_file)[0]
        mask_path = os.path.join(masks_dir, img_id + '.png')
        if not os.path.exists(mask_path):
            continue

        # Randomly pick a prompt
        prompt      = random.choice(PROMPTS[label])
        prompt_slug = prompt.replace(" ", "_")

        image     = Image.open(os.path.join(imgs_dir, img_file)).convert('RGB')
        orig_size = image.size  # (W, H)

        inputs = processor(
            text=[prompt],
            images=[image],
            return_tensors="pt",
            padding=True
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits  # [1, H, W]

        # Resize prediction to original image size
        pred = torch.sigmoid(logits[0])
        pred = F.interpolate(
            pred.unsqueeze(0).unsqueeze(0),
            size=(orig_size[1], orig_size[0]),
            mode='bilinear', align_corners=False
        ).squeeze()

        binary_mask = (pred.cpu().numpy() > threshold).astype(np.uint8) * 255

        # Save with zero-padded numeric ID
        numeric_id = str(idx + 1).zfill(4)
        out_name   = f"{numeric_id}__{prompt_slug}.png"
        Image.fromarray(binary_mask).save(os.path.join(output_dir, out_name))

        saved_meta.append({
            'idx'      : numeric_id,
            'img_file' : img_file,
            'prompt'   : prompt,
            'mask_name': out_name
        })

    # Save metadata
    meta_path = os.path.join(output_dir, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(saved_meta, f, indent=2)

    print(f"✅ Saved {len(saved_meta)} masks → {output_dir}")
    print(f"📋 Metadata → {meta_path}")
    return saved_meta