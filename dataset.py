# ============================================================
#  dataset.py — PyTorch Dataset for Drywall QA
# ============================================================

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image

from config import PROMPTS, IMAGE_SIZE


class DrywallDataset(Dataset):
    """
    Loads images and binary masks for two classes:
      - 'taping' (drywall seam / joint areas)
      - 'crack'  (wall cracks)

    Each __getitem__ returns:
      image  : FloatTensor [3, H, W]
      prompt : str  (randomly sampled from PROMPTS[label])
      mask   : FloatTensor [H, W]  values in [0, 1]
      label  : str  ('taping' or 'crack')
    """

    def __init__(self, d1_imgs, d1_masks, d2_imgs, d2_masks,
                 image_size=IMAGE_SIZE):
        self.image_size = image_size
        self.samples    = []

        for imgs_dir, masks_dir, label in [
            (d1_imgs, d1_masks, 'taping'),
            (d2_imgs, d2_masks, 'crack')
        ]:
            if not os.path.exists(imgs_dir):
                print(f"⚠️  Missing: {imgs_dir}")
                continue
            for img_file in os.listdir(imgs_dir):
                if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                mask_path = os.path.join(
                    masks_dir, os.path.splitext(img_file)[0] + '.png'
                )
                if os.path.exists(mask_path):
                    self.samples.append({
                        'image': os.path.join(imgs_dir, img_file),
                        'mask' : mask_path,
                        'label': label
                    })

        taping = sum(1 for s in self.samples if s['label'] == 'taping')
        crack  = sum(1 for s in self.samples if s['label'] == 'crack')
        print(f"✅ Dataset loaded — taping: {taping} | crack: {crack} | total: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s      = self.samples[idx]
        image  = Image.open(s['image']).convert('RGB')
        mask   = Image.open(s['mask']).convert('L')

        image  = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask   = mask.resize((self.image_size, self.image_size), Image.NEAREST)

        image  = T.ToTensor()(image)
        mask   = torch.tensor(np.array(mask), dtype=torch.float32) / 255.0
        prompt = random.choice(PROMPTS[s['label']])

        return image, prompt, mask, s['label']


def get_dataloaders(d1_train_imgs, d1_train_masks,
                    d1_valid_imgs, d1_valid_masks,
                    d2_train_imgs, d2_train_masks,
                    d2_valid_imgs, d2_valid_masks,
                    batch_size, image_size=IMAGE_SIZE):
    """Build and return train and validation DataLoaders."""

    train_ds = DrywallDataset(d1_train_imgs, d1_train_masks,
                              d2_train_imgs, d2_train_masks,
                              image_size=image_size)

    valid_ds = DrywallDataset(d1_valid_imgs, d1_valid_masks,
                              d2_valid_imgs, d2_valid_masks,
                              image_size=image_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    print(f"✅ Train batches: {len(train_loader)} | Valid batches: {len(valid_loader)}")
    return train_loader, valid_loader