#  evaluate.py — Per-class evaluation on validation set


import numpy as np
import torch
import torchvision.transforms as T
from tqdm import tqdm

from utils import compute_metrics


def evaluate(model, processor, dataloader, device):
    """
    Runs evaluation on a dataloader.
    Returns per-class and overall mIoU and Dice scores.
    """
    model.eval()
    results = {
        'taping': {'iou': [], 'dice': []},
        'crack' : {'iou': [], 'dice': []}
    }

    with torch.no_grad():
        for images, prompts, masks, labels in tqdm(dataloader, desc="Evaluating"):
            images, masks = images.to(device), masks.to(device)

            inputs = processor(
                text=list(prompts),
                images=[T.ToPILImage()(img.cpu()) for img in images],
                return_tensors="pt", padding=True
            ).to(device)

            logits = model(**inputs).logits

            for i in range(len(images)):
                iou, dice = compute_metrics(
                    logits[i].unsqueeze(0),
                    masks[i].unsqueeze(0)
                )
                results[labels[i]]['iou'].append(iou)
                results[labels[i]]['dice'].append(dice)

    return results


def print_results(results, seed=None):
    header = f"EVALUATION RESULTS" + (f"  (Seed: {seed})" if seed else "")
    print("\n" + "=" * 55)
    print(f"   {header}")
    print("=" * 55)
    print(f"{'Class':<15} {'mIoU':>10} {'Dice':>10} {'Samples':>10}")
    print("-" * 55)

    all_ious, all_dices = [], []
    for label, m in results.items():
        miou  = np.mean(m['iou'])
        mdice = np.mean(m['dice'])
        all_ious.extend(m['iou'])
        all_dices.extend(m['dice'])
        print(f"{label:<15} {miou:>10.4f} {mdice:>10.4f} {len(m['iou']):>10}")

    print("-" * 55)
    print(f"{'Overall':<15} {np.mean(all_ious):>10.4f} "
          f"{np.mean(all_dices):>10.4f} {len(all_ious):>10}")
    print("=" * 55)

    return {
        'taping_iou' : np.mean(results['taping']['iou']),
        'taping_dice': np.mean(results['taping']['dice']),
        'crack_iou'  : np.mean(results['crack']['iou']),
        'crack_dice' : np.mean(results['crack']['dice']),
        'overall_iou': np.mean(all_ious),
        'overall_dice': np.mean(all_dices),
    }