from dataclasses import dataclass


@dataclass
class ClientConfig:
    k: int
    lr: float
    local_epochs: int
    batch_size: int
    reg: float  # L2 regularization strength
