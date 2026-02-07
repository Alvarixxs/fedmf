from __future__ import annotations
import math
import random
from typing import Dict, List, Tuple
from collections import defaultdict

import torch

from opacus.accountants import RDPAccountant

from components.client import Client
from configs.dpConfig import DPConfig
from configs.serverConfig import ServerConfig
from utils.utils import add_gaussian_noise


class Server:
    """
    """

    def __init__(self, cfg: ServerConfig, device: torch.device, dp_cfg: DPConfig, clients: List[Client]) -> None:
        """
        """
        self.cfg = cfg
        self.device = device
        self.dp_cfg = dp_cfg

        if self.dp_cfg.mode == "central":
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
            sm, cnt = client.compute_sum()
            total_sm += sm
            total_cnt += cnt

        return total_sm / max(total_cnt, 1)
    
    # -----------------------------
    # Federated training
    # -----------------------------
    def _sample_clients(self) -> List[Client]:
        selected = [c for c in self.clients if random.random() < self.cfg.sample_rate]
        return selected

    def aggregate_item_updates(
        self,
        updates_list: List[Dict[int, Tuple[torch.Tensor, torch.Tensor]]],
    ) -> None:
        """
        Correct central DP (Gaussian mechanism on ONE fixed-dimensional aggregate per round):
        1) Each client upload is already clipped on the client to L2 <= C (your client code does this).
        2) Server forms a dense aggregate update over ALL parameters (Q, bi).
        3) Server averages over m selected clients.
        4) If central DP: add Gaussian noise once to the dense average with std = sigma * C / m.
        5) Apply update, then step accountant once.
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

        if self.dp_cfg.mode == "central":
            # Sensitivity of the average is C/m because each client's *full upload vector*
            # is clipped to L2 <= C on the client, and embedding it into dense tensors
            # does not increase its L2 norm.
            sens = self.dp_cfg.clip_norm / m

            dQ_avg = add_gaussian_noise(dQ_avg, self.dp_cfg.noise_multiplier, sens)
            db_avg = add_gaussian_noise(db_avg, self.dp_cfg.noise_multiplier, sens)

        with torch.no_grad():
            self.Q += dQ_avg
            self.bi += db_avg

        if self.dp_cfg.mode == "central":
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
            history.append((rnd, train_rmse, test_rmse))

            analysis = f"Round {rnd:3d}: train RMSE = {train_rmse:.4f}, test RMSE = {test_rmse:.4f}, dp mode = {self.dp_cfg.mode}"
            if self.dp_cfg.mode in ["local", "central"]:
                eps = self.privacy_report()
                analysis += f", ε = {eps:.2f}"

            print(analysis)

        return history
    
        # -----------------------------
    # Privacy reporting (local DP)
    # -----------------------------
    def privacy_report(self) -> float:
        """
        """        
        if self.dp_cfg.mode == "central":
            eps, _ = self._accountant.get_privacy_spent(
                delta=self.dp_cfg.delta,
            )
            return eps
        
        elif self.dp_cfg.mode == "local":
            epsilons = []
            for c in self.clients:
                eps = c.get_epsilon()
                epsilons.append(eps)

            return max(epsilons)
        
        return float("inf")

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