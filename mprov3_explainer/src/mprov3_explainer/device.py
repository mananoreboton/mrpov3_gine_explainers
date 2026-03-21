"""Device selection: MPS (Apple Silicon / M3 Pro), CUDA, or CPU."""

import torch


def get_device() -> torch.device:
    """Prefer MPS (Metal) on Apple Silicon, then CUDA, then CPU."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}", flush=True)
    return device
