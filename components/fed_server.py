from __future__ import annotations
import math
import random
from typing import Dict, List, Tuple

import torch

from components.fed_client import fedClient
from configs.serverConfig import ServerConfig


class fedServer:
    """
    """

    def __init__(self, cfg: ServerConfig, device: torch.device, clients: List[fedClient]) -> None:
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
            sm, cnt = client.compute_sum_train()
            total_sm += sm
            total_cnt += cnt

        return total_sm / max(total_cnt, 1)
    
    # -----------------------------
    # Federated training
    # -----------------------------
    def _sample_clients(self) -> List[fedClient]:
        selected = [c for c in self.clients if random.random() < self.cfg.sample_rate]
        return selected

    def aggregate_item_updates(
        self,
        updates_list: List[Dict[int, Tuple[torch.Tensor, torch.Tensor]]],
    ) -> None:
        """
        """
        m = len(updates_list)
        if m == 0:
            return  # nothing released/updated; don't step accountant

        dQ_sum = torch.zeros_like(self.Q)   # [n_items, k]
        db_sum = torch.zeros_like(self.bi)  # [n_items]

        # Accumulate sparse client uploads into dense tensors
        for upd in updates_list:
            for item_id, (dq, db) in upd.items():
                dQ_sum[item_id] += dq
                db_sum[item_id] += db

        # Average across selected clients
        dQ_avg = dQ_sum / m
        db_avg = db_sum / m

        with torch.no_grad():
            self.Q += dQ_avg
            self.bi += db_avg

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

            print(f"Round {rnd:3d}: train RMSE = {train_rmse:.4f}, test RMSE = {test_rmse:.4f}")

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