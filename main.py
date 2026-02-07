import argparse

import torch

from components.fed_client import fedClient
from components.fed_server import fedServer
from configs.dpConfig import DPConfig
from configs.serverConfig import ServerConfig
from dataset.movielens import download_movielens_latest_small, load_and_preprocess
from components.fedDP_server import fedDPServer
from configs.clientConfig import ClientConfig
from utils.utils import plot_history, split_per_user, set_seed


def main():
    ap = argparse.ArgumentParser()

    # Training settings
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test_frac", type=float, default=0.2)

    # Model settings
    ap.add_argument("--model", type=str, default="FedDP_MF", choices=["MF", "Fed_MF", "FedDP_MF"])

    # MF settings
    ap.add_argument("--k", type=int, default=32)

    # Training settings
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--local_epochs", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--reg", type=float, default=1e-3)

    # Federated learning settings
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--sample_rate", type=float, default=0.1)

    # Differential Privacy settings
    ap.add_argument("--clip_norm", type=float, default=1.0)
    ap.add_argument("--noise_multiplier", type=float, default=2.5)
    ap.add_argument("--delta", type=float, default=1e-6)

    # Other settings
    ap.add_argument("--no_plot", action="store_true", help="Disable plotting RMSE curves.")

    args = ap.parse_args()
    set_seed(args.seed)

    # Data
    ratings_csv = download_movielens_latest_small(data_dir="data")
    ratings_df, n_users, n_items = load_and_preprocess(ratings_csv)
    print(f"MovieLens: users={n_users}, items={n_items}, interactions={len(ratings_df)}")

    # Model and training setup
    if args.model == "MF":
        print("Training non-federated MF...")
        # Create a single client with all data for non-federated baseline

    elif args.model in ["Fed_MF", "FedDP_MF"]:
        by_user = split_per_user(ratings_df)

        client_cfg = ClientConfig(
            k=args.k, 
            lr=args.lr, 
            local_epochs=args.local_epochs, 
            batch_size=args.batch_size, 
            reg=args.reg,
        )

        server_cfg = ServerConfig(
            n_items=n_items,
            k=args.k,
            rounds=args.rounds,
            sample_rate=args.sample_rate,
        )

        clients = [
            fedClient(
                user_id=u, 
                cfg=client_cfg, 
                device=torch.device("cpu"),
                user_data=by_user.get(u, []),
                test_frac=args.test_frac,
            )
            for u in range(n_users)
        ]
        print(f"Created {len(clients)} clients.")

        if args.model == "Fed_MF":            
            server = fedServer(
                cfg=server_cfg,
                device=torch.device("cpu"),
                clients=clients,
            )
            print("Server initialized.")
            print("Training federated MF (no DP)...")

        elif args.model == "FedDP_MF":            
            dp_cfg=DPConfig(
                clip_norm=args.clip_norm,
                noise_multiplier=args.noise_multiplier,
                delta=args.delta,
            )

            server = fedDPServer(
                cfg=server_cfg,
                device=torch.device("cpu"),
                dp_cfg=dp_cfg,
                clients=clients,
            )
            print("Server initialized.")
            print("Training federated MF with DP...")


    history = server.train()

    if not args.no_plot:
        plot_history(history)


if __name__ == "__main__":
    main()
