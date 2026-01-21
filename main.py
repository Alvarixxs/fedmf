import argparse

from dataset.movielens import download_movielens_latest_small, load_and_preprocess
from components.trainer import Trainer


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test_frac", type=float, default=0.2)

    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--client_fraction", type=float, default=0.2)
    ap.add_argument("--local_epochs", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--reg", type=float, default=1e-3)

    ap.add_argument(
        "--no_weight_by_client_data",
        action="store_true",
        help="Disable weighting uploads by |D_u| (client data size).",
    )

    ap.add_argument("--no_plot", action="store_true", help="Disable plotting RMSE curves.")
    ap.add_argument("--log_every", type=int, default=5, help="Print metrics every N rounds (0 disables periodic logs).")

    args = ap.parse_args()

    # Data
    ratings_csv = download_movielens_latest_small(data_dir="data")
    ratings_df, n_users, n_items = load_and_preprocess(ratings_csv)
    print(f"MovieLens: users={n_users}, items={n_items}, interactions={len(ratings_df)}")

    # Trainer (mu will be set after split)
    trainer = Trainer(
        n_users=n_users,
        n_items=n_items,
        mu=0.0,
        k=args.k,
        rounds=args.rounds,
        client_fraction=args.client_fraction,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        reg=args.reg,
        seed=args.seed,
        weight_by_client_data=not args.no_weight_by_client_data,
    )

    # Split using the trainer's method (and set trainer.mu from TRAIN only)
    train_by_user, test_by_user, mu = trainer.prepare_data_split(
        ratings_df,
        test_frac=args.test_frac,
        seed=args.seed,
        set_mu=True,
    )
    print(f"Train-only global mean mu = {mu:.4f}")

    # Train
    trainer.train(
        train_by_user=train_by_user,
        test_by_user=test_by_user,
        plot=not args.no_plot,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
