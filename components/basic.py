from __future__ import annotations
import math
import random
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

from configs.basicConfig import BasicConfig


class MatrixFactorization:
    """
    """

    def __init__(
        self,
        cfg: BasicConfig,
        device: torch.device,
        data: List[Tuple[int, int, float]],  # (u, i, r)
        test_frac: float,
    ):
        self.cfg = cfg
        self.device = device

        self.train_data, self.test_data = self._split_data(data, test_frac)

        # Global mean
        self.mu = self._compute_global_mean()

        # Parameters
        self.P = 0.01 * torch.randn(cfg.n_users, cfg.k, device=device)
        self.Q = 0.01 * torch.randn(cfg.n_items, cfg.k, device=device)
        self.bu = torch.zeros(cfg.n_users, device=device)
        self.bi = torch.zeros(cfg.n_items, device=device)

        # Enable gradients
        self.P.requires_grad_(True)
        self.Q.requires_grad_(True)
        self.bu.requires_grad_(True)
        self.bi.requires_grad_(True)

    # -----------------------------
    # Data handling
    # -----------------------------
    def _split_data(
        self,
        data: List[Tuple[int, int, float]],
        test_frac: float,
    ):
        random.shuffle(data)
        n = len(data)
        n_test = int(test_frac * n)
        return data[n_test:], data[:n_test]

    def _compute_global_mean(self) -> float:
        return sum(r for (_, _, r) in self.train_data) / len(self.train_data)

    # -----------------------------
    # Training
    # -----------------------------
    def train(self):
        history = []

        users = torch.tensor([u for (u, _, _) in self.train_data], dtype=torch.long)
        items = torch.tensor([i for (_, i, _) in self.train_data], dtype=torch.long)
        ratings = torch.tensor([r for (_, _, r) in self.train_data], dtype=torch.float32)

        dataset = TensorDataset(users, items, ratings)
        loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True)

        opt = torch.optim.SGD([self.P, self.Q, self.bu, self.bi], lr=self.cfg.lr)
        mu_t = torch.tensor(self.mu, device=self.device)

        for epoch in range(1, self.cfg.epochs + 1):
            for u, i, r in loader:
                u = u.to(self.device)
                i = i.to(self.device)
                r = r.to(self.device)

                pred = (
                    mu_t
                    + self.bu[u]
                    + self.bi[i]
                    + (self.P[u] * self.Q[i]).sum(dim=1)
                )

                err = pred - r
                mse = (err ** 2).mean()

                l2 = (
                    self.P[u].pow(2).sum()
                    + self.Q[i].pow(2).sum()
                    + self.bu[u].pow(2).sum()
                    + self.bi[i].pow(2).sum()
                )

                loss = mse + self.cfg.reg * l2

                opt.zero_grad()
                loss.backward()
                opt.step()

            train_rmse = self.rmse(split="train")
            test_rmse = self.rmse(split="test")
            history.append((epoch, train_rmse, test_rmse))

            print(
                f"Epoch {epoch:3d} | "
                f"train RMSE = {train_rmse:.4f} | "
                f"test RMSE = {test_rmse:.4f}"
            )

        return history

    # -----------------------------
    # Evaluation
    # -----------------------------
    @torch.no_grad()
    def rmse(self, split: str) -> float:
        data = self.train_data if split == "train" else self.test_data

        se = 0.0
        for (u, i, r) in data:
            pred = (
                self.mu
                + self.bu[u]
                + self.bi[i]
                + torch.dot(self.P[u], self.Q[i])
            )
            se += (pred.item() - r) ** 2

        return math.sqrt(se / len(data))
