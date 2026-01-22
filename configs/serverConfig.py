from dataclasses import dataclass


@dataclass
class ServerConfig:
    n_items: int
    k: int
    client_frac: float
    rounds: int