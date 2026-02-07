from __future__ import annotations
import math
import random
from typing import Dict, List, Tuple
from collections import defaultdict

import torch

from opacus.accountants import RDPAccountant

from components.fed_client import fedClient
from configs.dpConfig import DPConfig
from configs.serverConfig import ServerConfig
from utils.utils import add_gaussian_noise, clip_l2


class fedDPServer:
    """
    """

    def __init__(self, cfg: ServerConfig, device: torch.device, dp_cfg: DPConfig, clients: List[fedClient]) -> None:
        """
        """
        self.cfg = cfg
        self.device = device
        self.dp_cfg = dp_cfg

        self._accountant = RDPAccountant()

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

        for upd in updates_list:
            item_ids = list(upd.keys())
            dQs = torch.stack([upd[i][0] for i in item_ids])   # [s, k]
            dbs = torch.stack([upd[i][1] for i in item_ids])   # [s]

            # ---- Per-client clipping ----
            flat = torch.cat([dQs.reshape(-1), dbs.reshape(-1)], dim=0)
            flat = clip_l2(flat, self.dp_cfg.clip_norm)

            m_Q = dQs.numel()
            dQs = flat[:m_Q].reshape_as(dQs)
            dbs = flat[m_Q:].reshape_as(dbs)

            # Accumulate
            for t, item_id in enumerate(item_ids):
                dQ_sum[item_id] += dQs[t]
                db_sum[item_id] += dbs[t]

        # Average across selected clients
        dQ_avg = dQ_sum / m
        db_avg = db_sum / m

        sens = self.dp_cfg.clip_norm / m

        dQ_avg = add_gaussian_noise(dQ_avg, self.dp_cfg.noise_multiplier, sens)
        db_avg = add_gaussian_noise(db_avg, self.dp_cfg.noise_multiplier, sens)

        with torch.no_grad():
            self.Q += dQ_avg
            self.bi += db_avg

        self._accountant.step(
            noise_multiplier=self.dp_cfg.noise_multiplier,
            sample_rate=self.cfg.sample_rate,
        )

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
            eps = self.privacy_report()
            history.append((rnd, train_rmse, test_rmse, eps))

            analysis = f"Round {rnd:3d}: train RMSE = {train_rmse:.4f}, test RMSE = {test_rmse:.4f}, ε = {eps:.2f}"

            print(analysis)

        return history
    
    # -----------------------------
    # Privacy reporting (local DP)
    # -----------------------------
    def privacy_report(self) -> float:
        """
        """        
        eps, _ = self._accountant.get_privacy_spent(
            delta=self.dp_cfg.delta,
        )
        return eps
        
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