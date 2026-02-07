from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

@dataclass
class DPConfig:
    clip_norm: float # L2 clipping bound (user-level = one client)
    noise_multiplier: float # Gaussian noise multiplier (std = noise_multiplier * clip_norm)
    delta: float