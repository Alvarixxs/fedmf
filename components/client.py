from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class ClientConfig:
    k: int
    lr: float
    local_epochs: int
    batch_size: int
    reg: float  # L2 regularization strength


class Client:
    """
    One client = one user.
    Keeps local user parameters (p_u, b_u) across rounds.
    Trains on its own ratings and returns updates for touched items (q_i, b_i).
    """

    def __init__(self, user_id: int, config: ClientConfig, device: torch.device):
        self.user_id = user_id
        self.cfg = config
        self.device = device

        # Local user parameters (persist)
        self.p_u = (0.01 * torch.randn(self.cfg.k, device=self.device))
        self.b_u = torch.tensor(0.0, device=self.device)

    def local_train(
        self,
        user_train_data: List[Tuple[int, float]],
        mu: float,
        Q_items: torch.Tensor,
        bi_items: torch.Tensor,
    ) -> Dict[int, Tuple[torch.Tensor, torch.Tensor, float]]:
        """
        Parameters
        ----------
        user_train_data: list of (global_item_id, rating)
        mu: global mean rating
        Q_items: server's full Q [n_items, k]
        bi_items: server's full bi [n_items]

        Returns
        -------
        uploads: dict global_item_id -> (q_i_new, b_i_new, weight)
        """
        if len(user_train_data) == 0:
            return {}

        # Items this client touched
        items = sorted(set(i for (i, _) in user_train_data))
        item_pos = {item_id: t for t, item_id in enumerate(items)}

        # Local copies of touched item params
        Q_u = Q_items[items].clone().detach().requires_grad_(True)     # [m, k]
        bi_u = bi_items[items].clone().detach().requires_grad_(True)   # [m]

        # Local user params (learnable this round, then persisted)
        p_u = self.p_u.clone().detach().requires_grad_(True)
        b_u = self.b_u.clone().detach().requires_grad_(True)

        # Build local dataset
        ii = torch.tensor([item_pos[i] for (i, _) in user_train_data],
                          device=self.device, dtype=torch.long)
        rr = torch.tensor([r for (_, r) in user_train_data],
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

        # Count how many ratings per item for weighting
        w = len(user_train_data)

        # Upload updated item params for touched items
        uploads: Dict[int, Tuple[torch.Tensor, torch.Tensor, float]] = {}
        with torch.no_grad():
            for t, item_id in enumerate(items):
                uploads[item_id] = (Q_u[t].detach().clone(), bi_u[t].detach().clone(), w)

        return uploads

    @torch.no_grad()
    def predict_one(self, mu: float, bi: torch.Tensor, q_i: torch.Tensor) -> float:
        """
        Predict rating for this user and one item given (bi_i, q_i).
        """
        mu_t = torch.tensor(mu, device=self.device, dtype=torch.float32)
        pred = mu_t + self.b_u + bi + torch.dot(self.p_u, q_i)
        return float(pred)
