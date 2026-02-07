import matplotlib.pyplot as plt
from collections import defaultdict
import random
import numpy as np
from typing import Dict, List, Tuple
import pandas as pd
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def split_per_user(
    ratings: pd.DataFrame,
) -> Dict[int, List[Tuple[int, float]]]:
    """
    """
    by_user: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    for row in ratings.itertuples(index=False):
        by_user[int(row.u)].append((int(row.i), float(row.rating)))
    
    return by_user


def plot_history(history: List[Tuple[int, float, float]]) -> None:
    """
    """
    if not history:
        print("No history to plot (train() hasn't been run yet).")
        return

    xs = [h[0] for h in history]
    tr = [h[1] for h in history]
    te = [h[2] for h in history]

    plt.figure()
    plt.plot(xs, tr, label="train")
    plt.plot(xs, te, label="test")
    plt.xlabel("round")
    plt.ylabel("RMSE")
    plt.legend()
    plt.tight_layout()
    plt.show()


def clip_l2(
    x: torch.Tensor,
    clip_norm: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    """
    l2 = torch.linalg.vector_norm(x.reshape(-1), ord=2)
    scale = min(1.0, (clip_norm / (l2 + eps)).item())
    return x * scale


def add_gaussian_noise(
    x: torch.Tensor,
    noise_multiplier: float,
    clip_norm: float,
) -> torch.Tensor:
    """
    """
    noise_std = noise_multiplier * clip_norm
    noise = torch.randn_like(x) * noise_std
    return x + noise
