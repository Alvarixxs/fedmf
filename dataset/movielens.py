from __future__ import annotations
import os
import zipfile
from typing import Tuple

import requests
import pandas as pd



# -----------------------------
# Data utilities
# -----------------------------
def download_movielens_latest_small(data_dir: str = "data") -> str:
    """
    Downloads and extracts ml-latest-small into data_dir.
    Returns path to ratings.csv.
    """
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "ml-latest-small.zip")
    extract_dir = os.path.join(data_dir, "ml-latest-small")
    ratings_path = os.path.join(extract_dir, "ratings.csv")

    if os.path.exists(ratings_path):
        return ratings_path

    url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with open(zip_path, "wb") as f:
        f.write(r.content)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(data_dir)

    if not os.path.exists(ratings_path):
        raise FileNotFoundError(f"Expected {ratings_path} after extraction.")
    return ratings_path


def load_and_preprocess(ratings_csv: str) -> Tuple[pd.DataFrame, int, int]:
    ratings = pd.read_csv(ratings_csv)

    user_ids = ratings["userId"].unique()
    item_ids = ratings["movieId"].unique()

    user2idx = {u: i for i, u in enumerate(user_ids)}
    item2idx = {m: i for i, m in enumerate(item_ids)}

    ratings["u"] = ratings["userId"].map(user2idx)
    ratings["i"] = ratings["movieId"].map(item2idx)

    n_users = len(user2idx)
    n_items = len(item2idx)
    return ratings, n_users, n_items