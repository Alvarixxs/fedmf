from dataclasses import dataclass


@dataclass
class ServerConfig:
    n_items: int
    k: int
    sample_rate: float
    rounds: int