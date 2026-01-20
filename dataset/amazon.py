"""
Federated Matrix Factorization (MF) on Amazon Reviews (UCSD / McAuley Amazon Review Data 2018),
using the same client/server modules as the MovieLens project.

Assumes you already have:
  - client.py   (Client, ClientConfig)
  - server.py   (Server, ServerConfig)

Run:
  python amazon_trainer.py --category Video_Games --subset small

Notes:
- Uses ratings-only per-category CSVs (recommended for MF baselines).
- Builds (u, i, rating) table, per-user split, trains FedMF and plots RMSE curve.
"""

from __future__ import annotations

import os
from typing import List

import requests
import pandas as pd


# -----------------------------
# Amazon dataset download
# -----------------------------
def download_amazon_category_csv(
    category: str,
    *,
    data_dir: str = "data",
    subset: str = "small",   # "small" recommended
    timeout: int = 120,
) -> str:
    """
    Downloads Amazon ratings-only CSV for a given category.
    """
    os.makedirs(data_dir, exist_ok=True)

    cat = category.replace(" ", "_")
    out_path = os.path.join(data_dir, f"amazon_{subset}_{cat}.csv")
    if os.path.exists(out_path):
        return out_path

    if subset == "small":
        base = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall"
    elif subset == "full":
        base = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFiles"
    else:
        raise ValueError("subset must be 'small' or 'full'")

    url = f"{base}/{cat}.csv"
    headers = {"User-Agent": "Mozilla/5.0"}

    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()

    with open(out_path, "wb") as f:
        f.write(r.content)

    return out_path


# -----------------------------
# Amazon CSV -> (u, i, rating)
# -----------------------------
def load_amazon_category_ratings(
    csv_path: str,
    *,
    min_user_interactions: int = 5,
    min_item_interactions: int = 5,
    max_users: int | None = 50000,
    max_items: int | None = 50000,
) -> tuple[pd.DataFrame, int, int]:
    """
    Reads Amazon ratings CSV and returns:
      ratings_df with columns ['u','i','rating'] (contiguous int indices)
      n_users, n_items

    Handles both headered and headerless formats defensively.
    """
    # Try headered CSV first
    df = pd.read_csv(csv_path)
    cols_lower = [c.lower() for c in df.columns]

    def pick_col(cands: List[str]) -> str | None:
        for c in df.columns:
            if c.lower() in cands:
                return c
        return None

    user_col = pick_col(["reviewerid", "user_id", "userid"])
    item_col = pick_col(["asin", "parent_asin"])
    rating_col = pick_col(["overall", "rating"])

    if user_col and item_col and rating_col:
        df = df[[user_col, item_col, rating_col]].rename(
            columns={user_col: "user", item_col: "item", rating_col: "rating"}
        )
    else:
        # Fallback: headerless CSV, assume first three columns are user,item,rating
        df = pd.read_csv(csv_path, header=None)
        if df.shape[1] < 3:
            raise ValueError("Amazon CSV has fewer than 3 columns; cannot parse (user,item,rating).")
        df = df.iloc[:, :3]
        df.columns = ["user", "item", "rating"]

    df = df.dropna(subset=["user", "item", "rating"])
    df["user"] = df["user"].astype(str)
    df["item"] = df["item"].astype(str)
    df["rating"] = df["rating"].astype(float)

    # Filter sparse users/items
    user_counts = df["user"].value_counts()
    item_counts = df["item"].value_counts()
    keep_users = user_counts[user_counts >= min_user_interactions].index
    keep_items = item_counts[item_counts >= min_item_interactions].index
    df = df[df["user"].isin(keep_users) & df["item"].isin(keep_items)]

    # Optional caps for speed
    if max_users is not None:
        top_users = df["user"].value_counts().head(max_users).index
        df = df[df["user"].isin(top_users)]
    if max_items is not None:
        top_items = df["item"].value_counts().head(max_items).index
        df = df[df["item"].isin(top_items)]

    # Map to contiguous indices
    user_ids = df["user"].unique()
    item_ids = df["item"].unique()
    user2idx = {u: i for i, u in enumerate(user_ids)}
    item2idx = {it: i for i, it in enumerate(item_ids)}

    ratings_df = pd.DataFrame({
        "u": df["user"].map(user2idx).astype(int),
        "i": df["item"].map(item2idx).astype(int),
        "rating": df["rating"].astype(float),
    })

    return ratings_df, len(user2idx), len(item2idx)



