from dataclasses import dataclass


@dataclass
class BasicConfig:
    """
    """
    n_users: int
    n_items: int
    k: int
    lr: float
    reg: float
    batch_size: int
    epochs: int