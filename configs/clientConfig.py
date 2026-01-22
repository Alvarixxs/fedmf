from dataclasses import dataclass


@dataclass
class ClientConfig:
    k: int
    lr: float
    local_epochs: int
    batch_size: int
    reg: float  # L2 regularization strength
    weight_by_client_data: bool # Whether to weight uploads by |D_u| (client data size