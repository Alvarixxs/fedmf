from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd  # <-- needed for split

from components.client import Client, ClientConfig
from components.server import Server, ServerConfig


@dataclass
class FederatedMFTrainer:
    # Problem sizes (mu can be set later after splitting)
    n_users: int
    n_items: int
    mu: float = 0.0

    # Hyperparams
    k: int = 32
    rounds: int = 30
    client_fraction: float = 0.2
    local_epochs: int = 2
    batch_size: int = 64
    lr: float = 0.05
    reg: float = 1e-3
    seed: int = 0
    weight_by_client_data: bool = True

    # Runtime / state
    device: torch.device = field(init=False)
    server: Server = field(init=False)
    clients: List[Client] = field(init=False)
    user_ids: List[int] = field(init=False)
    history: List[Tuple[int, float, float]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._seed_all(self.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.server = Server(ServerConfig(n_items=self.n_items, k=self.k), device=self.device)

        client_cfg = ClientConfig(
            k=self.k,
            lr=self.lr,
            local_epochs=self.local_epochs,
            batch_size=self.batch_size,
            reg=self.reg,
        )
        self.clients = [Client(user_id=u, config=client_cfg, device=self.device) for u in range(self.n_users)]
        self.user_ids = list(range(self.n_users))

    @staticmethod
    def _seed_all(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    # -----------------------------
    # Per-user split (federated clients)  ✅ moved into trainer
    # -----------------------------
    @staticmethod
    def split_per_user(
        ratings: pd.DataFrame,
        n_users: int,
        test_frac: float = 0.2,
        seed: int = 0,
    ) -> Tuple[Dict[int, List[Tuple[int, float]]], Dict[int, List[Tuple[int, float]]], float]:
        rng = random.Random(seed)

        by_user: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        for row in ratings.itertuples(index=False):
            by_user[int(row.u)].append((int(row.i), float(row.rating)))

        train_by_user: Dict[int, List[Tuple[int, float]]] = {}
        test_by_user: Dict[int, List[Tuple[int, float]]] = {}

        for u in range(n_users):
            items = by_user.get(u, []).copy()
            rng.shuffle(items)
            n = len(items)

            if n <= 1:
                train_by_user[u] = items
                test_by_user[u] = []
                continue

            n_test = max(1, int(test_frac * n))
            test_by_user[u] = items[:n_test]
            train_by_user[u] = items[n_test:]

        # Global mean mu on TRAIN only (avoid leakage)
        all_train = [r for u in train_by_user for (_, r) in train_by_user[u]]
        mu = float(np.mean(all_train)) if len(all_train) else 0.0

        return train_by_user, test_by_user, mu

    def prepare_data_split(
        self,
        ratings: pd.DataFrame,
        *,
        test_frac: float = 0.2,
        seed: Optional[int] = None,
        set_mu: bool = True,
    ) -> Tuple[Dict[int, List[Tuple[int, float]]], Dict[int, List[Tuple[int, float]]], float]:
        """Convenience wrapper using this trainer's n_users + seed; optionally sets self.mu."""
        s = self.seed if seed is None else seed
        train_by_user, test_by_user, mu = self.split_per_user(
            ratings=ratings,
            n_users=self.n_users,
            test_frac=test_frac,
            seed=s,
        )
        if set_mu:
            self.mu = mu
        return train_by_user, test_by_user, mu

    # -----------------------------
    # Evaluation
    # -----------------------------
    @torch.no_grad()
    def rmse(self, split_by_user: Dict[int, List[Tuple[int, float]]]) -> float:
        Q, bi = self.server.get_item_params()

        se = 0.0
        cnt = 0
        for client in self.clients:
            u = client.user_id
            data = split_by_user.get(u, [])
            if not data:
                continue
            for (i, r) in data:
                pred = client.predict_one(mu=self.mu, bi=bi[i], q_i=Q[i])
                se += (pred - r) ** 2
                cnt += 1

        return math.sqrt(se / max(1, cnt))

    # -----------------------------
    # Federated training
    # -----------------------------
    def _sample_clients(self) -> List[int]:
        m = max(1, int(self.client_fraction * self.n_users))
        return random.sample(self.user_ids, m)

    def train(
        self,
        train_by_user: Dict[int, List[Tuple[int, float]]],
        test_by_user: Dict[int, List[Tuple[int, float]]],
        *,
        plot: bool = True,
        log_every: int = 5,
    ) -> List[Tuple[int, float, float]]:
        self.history.clear()

        for rnd in range(1, self.rounds + 1):
            selected = self._sample_clients()

            Q, bi = self.server.get_item_params()
            updates_list = []

            for u in selected:
                uploads = self.clients[u].local_train(
                    user_train_data=train_by_user.get(u, []),
                    mu=self.mu,
                    Q_items=Q,
                    bi_items=bi,
                )

                if self.weight_by_client_data and uploads:
                    w = float(len(train_by_user.get(u, [])))
                    uploads = {iid: (q, b, w) for iid, (q, b, _oldw) in uploads.items()}

                updates_list.append(uploads)

            self.server.aggregate_item_updates(updates_list)

            train_rmse = self.rmse(train_by_user)
            test_rmse = self.rmse(test_by_user)
            self.history.append((rnd, train_rmse, test_rmse))

            if rnd == 1 or (log_every and rnd % log_every == 0):
                print(f"Round {rnd:03d} | train RMSE {train_rmse:.4f} | test RMSE {test_rmse:.4f}")

        if plot:
            self.plot_history()

        return self.history

    def plot_history(self) -> None:
        if not self.history:
            print("No history to plot (train() hasn't been run yet).")
            return

        xs = [h[0] for h in self.history]
        tr = [h[1] for h in self.history]
        te = [h[2] for h in self.history]

        plt.figure()
        plt.plot(xs, tr, label="train")
        plt.plot(xs, te, label="test")
        plt.xlabel("round")
        plt.ylabel("RMSE")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def get_server_params(self):
        return self.server.get_item_params()
