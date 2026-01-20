from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import defaultdict

import torch


@dataclass
class ServerConfig:
    n_items: int
    k: int


class Server:
    """
    Holds global item parameters (Q, bi) and aggregates client item updates.
    """

    def __init__(self, config: ServerConfig, device: torch.device):
        self.cfg = config
        self.device = device

        self.Q = (0.01 * torch.randn(self.cfg.n_items, self.cfg.k, device=self.device))
        self.bi = torch.zeros(self.cfg.n_items, device=self.device)

    @torch.no_grad()
    def get_item_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.Q, self.bi

    def aggregate_item_updates(
        self,
        updates_list: List[Dict[int, Tuple[torch.Tensor, torch.Tensor, float]]],
    ) -> None:
        """
        FedAvg-style aggregation on items:
        for each item i, average (q_i, b_i) over clients that sent an update for i.
        """
        q_sum = defaultdict(lambda: torch.zeros(self.cfg.k, device=self.device))
        b_sum = defaultdict(lambda: torch.tensor(0.0, device=self.device))
        w_sum = defaultdict(float)

        for upd in updates_list:
            for item_id, (q_new, b_new, w) in upd.items():
                q_sum[item_id] += w * q_new
                b_sum[item_id] += w * b_new
                w_sum[item_id] += w

        with torch.no_grad():
            for item_id, w in w_sum.items():
                self.Q[item_id] = q_sum[item_id] / w
                self.bi[item_id] = b_sum[item_id] / w
