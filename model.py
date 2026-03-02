

import torch
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

from config import MODEL_NAME


def load_model(device, checkpoint_path=None):
    """
    Load CLIPSeg model and processor.
    - Freezes CLIP encoder (only decoder is trained)
    - Optionally loads weights from a saved checkpoint

    Args:
        device          : torch device ('cuda' or 'cpu')
        checkpoint_path : path to .pt checkpoint file (optional)

    Returns:
        processor, model
    """
    processor = CLIPSegProcessor.from_pretrained(MODEL_NAME)
    model     = CLIPSegForImageSegmentation.from_pretrained(MODEL_NAME)
    model     = model.to(device)

    # Freeze CLIP encoder — only fine-tune the decoder
    for name, param in model.named_parameters():
        if 'clip' in name.lower():
            param.requires_grad = False

    # Load checkpoint if provided
    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print(f"✅ Loaded checkpoint: {checkpoint_path}")

    # Print parameter counts
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Model loaded — "
          f"Total: {total:,} | "
          f"Trainable: {trainable:,} | "
          f"Frozen: {total - trainable:,}")

    return processor, model