from __future__ import annotations
import math
import random
from typing import Dict, List, Tuple
from collections import defaultdict

import torch

from components.client import Client
from configs.serverConfig import ServerConfig


class Server:
    """
    """

    def __init__(self, cfg: ServerConfig, device: torch.device, clients: List[Client]) -> None:
        """
        """
        self.cfg = cfg
        self.device = device

        self.clients = clients

        self.Q = (0.01 * torch.randn(self.cfg.n_items, self.cfg.k, device=self.device))
        self.bi = torch.zeros(self.cfg.n_items, device=self.device)
        self.mu = self._compute_global_mean()

    def _compute_global_mean(self) -> float:
        """
        """
        total_sm = 0.0
        total_cnt = 0

        for client in self.clients:
            sm, cnt = client.compute_sum()
            total_sm += sm
            total_cnt += cnt

        return total_sm / max(total_cnt, 1)
    
    # -----------------------------
    # Federated training
    # -----------------------------
    def _sample_clients(self) -> List[Client]:
        """
        """
        n_sample = max(1, int(self.cfg.client_frac * len(self.clients)))
        return random.sample(self.clients, n_sample)

    def aggregate_item_updates(
        self,
        updates_list: List[Dict[int, Tuple[torch.Tensor, torch.Tensor, float]]],
    ) -> None:
        """
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

    def train(self) -> List[Tuple[int, float, float]]:
        """
        """
        history = []

        for rnd in range(1,  self.cfg.rounds + 1):
            selected = self._sample_clients()
            updates_list = []

            for client in selected:
                uploads = client.local_train(
                    mu=self.mu,
                    Q_items=self.Q,
                    bi_items=self.bi,
                )
                updates_list.append(uploads)

            self.aggregate_item_updates(updates_list)

            train_rmse = self.rmse(split="train")
            test_rmse = self.rmse(split="test")
            history.append((rnd, train_rmse, test_rmse))

            print(
                f"Round {rnd:3d}: "
                f"train RMSE = {train_rmse:.4f}, "
                f"test RMSE = {test_rmse:.4f}"
            )

        return history

    # -----------------------------
    # Evaluation
    # -----------------------------
    @torch.no_grad()
    def rmse(self, split: str) -> float:
        """
        """
        total_se = 0.0
        total_cnt = 0

        for client in self.clients:
            se, cnt = client.sum_squared_error(split, self.mu, self.Q, self.bi)
            total_se += se
            total_cnt += cnt

        return math.sqrt(total_se / max(1, total_cnt))