import argparse

import torch

from configs.dpConfig import DPConfig
from configs.serverConfig import ServerConfig
from dataset.movielens import download_movielens_latest_small, load_and_preprocess
from components.client import Client
from components.server import Server
from configs.clientConfig import ClientConfig
from utils.utils import plot_history, split_per_user, set_seed


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test_frac", type=float, default=0.2)

    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--sample_rate", type=float, default=0.3)
    ap.add_argument("--local_epochs", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--reg", type=float, default=1e-3)

    ap.add_argument("--clip_norm", type=float, default=1.0)
    ap.add_argument("--noise_multiplier", type=float, default=1.0)
    ap.add_argument("--delta", type=float, default=1e-6)

    ap.add_argument("--no_plot", action="store_true", help="Disable plotting RMSE curves.")

    args = ap.parse_args()
    set_seed(args.seed)

    # Data
    ratings_csv = download_movielens_latest_small(data_dir="data")
    ratings_df, n_users, n_items = load_and_preprocess(ratings_csv)
    print(f"MovieLens: users={n_users}, items={n_items}, interactions={len(ratings_df)}")

    by_user = split_per_user(ratings_df)

    dp_cfg=DPConfig(
        clip_norm=args.clip_norm,
        noise_multiplier=args.noise_multiplier,
        delta=args.delta,
    )

    clients = [
        Client(
            user_id=u, 
            cfg=ClientConfig(
                k=args.k, 
                lr=args.lr, 
                local_epochs=args.local_epochs, 
                batch_size=args.batch_size, 
                reg=args.reg,
            ), 
            device=torch.device("cpu"),
            user_data=by_user.get(u, []),
            test_frac=args.test_frac,
            dp_cfg=dp_cfg,
            )
        for u in range(n_users)
        ]
    print(f"Created {len(clients)} clients.")

    server = Server(
        cfg=ServerConfig(
            n_items=n_items,
            k=args.k,
            sample_rate=args.sample_rate,
            rounds=args.rounds,
        ),
        device=torch.device("cpu"),
        dp_cfg=dp_cfg,
        clients=clients,
    )
    print("Server initialized.")

    history = server.train()

    if not args.no_plot:
        plot_history(history)


if __name__ == "__main__":
    main()
