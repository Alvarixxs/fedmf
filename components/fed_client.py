from __future__ import annotations
import random
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

from configs.clientConfig import ClientConfig


class fedClient:
    """
    """

    def __init__(
        self,
        user_id: int,
        cfg: ClientConfig,
        device: torch.device,
        user_data: List[Tuple[int, float]],
        test_frac: float,
    ) -> None:
        """
        """
        self.user_id = user_id
        self.cfg = cfg
        self.device = device

        self.train_data, self.test_data = self._split_data(user_data, test_frac)

        # Local user parameters (persist)
        self.p_u = (0.01 * torch.randn(self.cfg.k, device=self.device))
        self.b_u = torch.tensor(0.0, device=self.device)

    # -----------------------------
    # Data handling (client-owned)
    # -----------------------------
    def _split_data(
        self,
        user_data: List[Tuple[int, float]],
        test_frac: float,
    ) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
        """
        """
        random.shuffle(user_data)
        n = len(user_data)

        if n <= 1:
            return user_data, []

        n_test = max(1, int(test_frac * n))
        test = user_data[:n_test]
        train = user_data[n_test:]

        return train, test
    
    def compute_sum_train(self) -> Tuple[float, int]:
        """
        """
        return sum(r for (_, r) in self.train_data), len(self.train_data)

    # -----------------------------
    # Local training (uses ONLY train split)
    # -----------------------------
    def local_train(
        self,
        mu: float,
        Q_items: torch.Tensor,
        bi_items: torch.Tensor,
    ) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """ 
        """
        if len(self.train_data) == 0:
            return {}

        # Items this client touched
        items = sorted(set(i for (i, _) in self.train_data))
        item_pos = {item_id: t for t, item_id in enumerate(items)}

        # Local copies of touched item params
        Q_u = Q_items[items].clone().detach().requires_grad_(True)     # [m, k]
        bi_u = bi_items[items].clone().detach().requires_grad_(True)   # [m]

        # Local user params (learnable this round, then persisted)
        p_u = self.p_u.clone().detach().requires_grad_(True)
        b_u = self.b_u.clone().detach().requires_grad_(True)

        # Build local dataset
        ii = torch.tensor([item_pos[i] for (i, _) in self.train_data],
                          device=self.device, dtype=torch.long)
        rr = torch.tensor([r for (_, r) in self.train_data],
                          device=self.device, dtype=torch.float32)

        loader = DataLoader(
            TensorDataset(ii, rr),
            batch_size=min(self.cfg.batch_size, len(rr)),
            shuffle=True
        )

        opt = torch.optim.SGD([p_u, b_u, Q_u, bi_u], lr=self.cfg.lr)

        mu_t = torch.tensor(mu, device=self.device, dtype=torch.float32)

        for _ in range(self.cfg.local_epochs):
            for (i_batch, r_batch) in loader:
                q_batch = Q_u[i_batch]        # [B, k]
                bi_batch = bi_u[i_batch]      # [B]

                pred = mu_t + b_u + bi_batch + (q_batch @ p_u)  # [B]
                err = pred - r_batch
                mse = (err ** 2).mean()

                # L2 regularization
                l2 = (
                    p_u.pow(2).sum()
                    + Q_u.pow(2).sum()
                    + b_u.pow(2)
                    + bi_u.pow(2).sum()
                )

                loss = mse + self.cfg.reg * l2

                opt.zero_grad()
                loss.backward()
                opt.step()

        # Persist updated user params locally
        self.p_u = p_u.detach()
        self.b_u = b_u.detach()

        with torch.no_grad():
            base_Q = Q_items[items]
            base_bi = bi_items[items]
            delta_Q = (Q_u.detach() - base_Q)          # [m, k]
            delta_bi = (bi_u.detach() - base_bi)       # [m]

        # Upload updated item params for touched items
        uploads: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        with torch.no_grad():
            for t, item_id in enumerate(items):
                uploads[item_id] = (delta_Q[t].detach().clone(), delta_bi[t].detach().clone())

        return uploads

    # -----------------------------
    # Prediction / evaluation
    # -----------------------------
    @torch.no_grad()
    def _predict_one(self, mu: float, bi: torch.Tensor, q_i: torch.Tensor) -> float:
        """
        """
        mu_t = torch.tensor(mu, device=self.device, dtype=torch.float32)
        pred = mu_t + self.b_u + bi + torch.dot(self.p_u, q_i)

        return float(pred)

    @torch.no_grad()
    def sum_squared_error(self, split: str, mu: float, Q_items: torch.Tensor, bi_items: torch.Tensor) -> Tuple[float, int]:
        """
        """
        data = self.train_data if split == "train" else self.test_data

        se = 0.0
        for (i, r) in data:
            pred = self._predict_one(mu, bi_items[i], Q_items[i])
            se += (pred - r) ** 2

        return se, len(data)