"""
preprocess.py
-------------
Combined data loading, cleaning, and preprocessing pipeline.

Absorbs all logic from DataProcesser.py and adds:
  - Integer index maps (user_idx, item_idx) needed by DropoutNet
  - simulate_cold_start() for K ∈ {0, 2, 5, 10, 20}
  - save/load helpers so Spark only runs once

Data flow:
  my_amazon_books_sample.parquet        (1% review data from DataLoader.py)
  my_amazon_books_meta_sample.parquet   (full meta from DataLoader.py)
        ↓
  filter users with >= 14 ratings
  rank-window split by k ratio (train / test)
  clean metadata text (title + description)
        ↓
  build user_idx / item_idx integer maps
  save to data/processed/
"""

import logging
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col, trim
from pyspark.sql.window import Window

log = logging.getLogger(__name__)

PROC_DIR = Path("data/processed")

# Cold-start K% levels used throughout the project
COLD_START_LEVELS = [0, 5, 10, 20, 30, 50, 75, 90]

# Minimum ratings per user to be eligible for cold-start simulation
MIN_ALPHA = 30

# Raw parquet file names (produced by DataLoader.py)
REVIEW_PARQUET = "my_amazon_books_sample.parquet"
META_PARQUET   = "my_amazon_books_meta_sample.parquet"


# ── Spark session (same config as friend's DataProcesser.py) ──────────────────

def get_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("Amazon_book_recommendation")
        .config("spark.driver.memory", "6g")
        .config("spark.executor.memory", "6g")
        .config("spark.default.parallelism", "32")
        .config("spark.sql.shuffle.partitions", "32")
        .config("spark.driver.maxResultSize", "1g")
        .config("spark.dynamicAllocation.enabled", "false")
        .config("spark.memory.fraction", "0.6")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .getOrCreate()
    )


# ── Step 1: Load & filter reviews (from friend's _als_sample_data_) ───────────

def load_and_split(spark: SparkSession, k: float):
    """
    Load review data, filter active users, and split into train/test.

    Directly absorbed from friend's DataProcessing._als_sample_data_():
      - Filters users with >= 14 ratings (90th percentile threshold)
      - Assigns random rank per user with fixed seed=42
      - Test set  = rows where rank <= cnt * k  (first k% of each user)
      - Train set = remaining rows

    Parameters
    ----------
    k : float
        Fraction of each user's history to use as test set (e.g. 0.2 = 20%)
    """
    log.info(f"Loading reviews from {REVIEW_PARQUET} (k={k}) ...")

    df_review = (
        spark.read.parquet(REVIEW_PARQUET)
        .select("user_id", "parent_asin", "rating")
        .cache()
    )

    # Filter active users (>= 14 ratings) — from friend's code
    user_count = df_review.groupBy("user_id").agg(
        F.count("*").alias("rating_cnt_by_user")
    )
    users_eligible = user_count.filter(
        F.col("rating_cnt_by_user") >= 14
    ).select("user_id")

    df_filtered = df_review.join(users_eligible, on="user_id", how="inner")

    # Recount after filtering
    user_count  = df_filtered.groupBy("user_id").agg(F.count("*").alias("cnt"))
    df_filtered = df_filtered.join(user_count, on="user_id")

    # Assign random rank per user with fixed seed — from friend's code
    window    = Window.partitionBy("user_id").orderBy(F.rand(42))
    df_ranked = df_filtered.withColumn("rank", F.row_number().over(window))
    df_ranked = df_ranked.withColumn("threshold", F.col("cnt") * k)

    test_df  = df_ranked.filter(F.col("rank") <= F.col("threshold"))
    train_df = df_ranked.filter(F.col("rank") >  F.col("threshold"))

    log.info(f"  Train: {train_df.count():,}  |  Test: {test_df.count():,}")
    return train_df, test_df


# ── Step 2: Load & clean metadata (from friend's _embedding_data_clean_) ──────

def load_and_clean_meta(spark: SparkSession, train_spark, test_spark,
                        max_chars: int = 500):
    """
    Load metadata, join to items seen in train+test, clean description text.

    Directly absorbed from friend's:
      - DataProcessing._embedding_data_()        : load + join meta
      - DataProcessing._embedding_data_clean_()  : clean text, build embedding_text
    """
    log.info(f"Loading metadata from {META_PARQUET} ...")

    df_meta = spark.read.parquet(META_PARQUET)

    # Join meta to items that actually appear in our data — from friend's code
    user_data = train_spark.unionByName(test_spark)
    meta_df   = df_meta.join(
        user_data.select("parent_asin").distinct(),
        on="parent_asin",
        how="inner",
    ).select("title", "parent_asin", "subtitle", "description", "author")

    # Clean description text — from friend's _embedding_data_clean_()
    text_col = F.concat_ws(" ", F.col("description"))

    common_patterns = [
        r"\bReview\b",
        r"\bAbout the Author\b",
        r"\bFrom the Author\b",
        r"\bFrom the Back Cover\b",
        r"\bProduct Description\b",
        r"\bPraise for\b",
    ]

    cleaned = (
        meta_df
        .withColumn("title_text",       F.coalesce(F.col("title").cast("string"), F.lit("")))
        .withColumn("description_text", F.coalesce(text_col, F.lit("")))
        .withColumn("description_text", F.regexp_replace("description_text", r"[\xa0\r\n\t]+", " "))
        .withColumn("description_text", F.regexp_replace("description_text", r"\s+", " "))
        .withColumn("description_text", F.trim(F.col("description_text")))
    )

    for pattern in common_patterns:
        cleaned = cleaned.withColumn(
            "description_text",
            F.regexp_replace(F.col("description_text"), pattern, "")
        )

    cleaned = (
        cleaned
        .withColumn("description_text",  F.regexp_replace("description_text", r"\s+", " "))
        .withColumn("description_text",  F.trim(F.col("description_text")))
        .withColumn("description_clean", F.substring(F.col("description_text"), 1, max_chars))
        .withColumn(
            "text",   # renamed from friend's "embedding_text" for our pipeline
            F.trim(F.concat_ws(" ", F.col("title_text"), F.col("description_clean")))
        )
        .filter(F.col("text").isNotNull())
        .filter(F.col("text") != "")
        .select("parent_asin", "text")
    )

    log.info(f"  Meta rows after cleaning: {cleaned.count():,}")
    return cleaned


# ── Step 3: Collect to pandas + build integer indices ─────────────────────────

def to_pandas_and_index(train_spark, test_spark, meta_spark):
    """
    Collect Spark DataFrames to pandas and add integer user_idx / item_idx.

    Index maps are built from the training vocab only.
    Test rows whose user/item wasn't seen in training are dropped
    (same as ALS.py coldStartStrategy='drop').
    """
    log.info("Collecting to pandas and building index maps ...")

    train = train_spark.select("user_id", "parent_asin", "rating").toPandas()
    test  = test_spark.select("user_id",  "parent_asin", "rating").toPandas()
    meta  = meta_spark.toPandas()

    # Build index maps from training vocab only
    users    = sorted(train["user_id"].unique())
    items    = sorted(train["parent_asin"].unique())
    user2idx = {u: i for i, u in enumerate(users)}
    item2idx = {it: i for i, it in enumerate(items)}

    def _apply_index(df):
        df = df.copy()
        df["user_idx"] = df["user_id"].map(user2idx)
        df["item_idx"] = df["parent_asin"].map(item2idx)
        df = df.dropna(subset=["user_idx", "item_idx"])
        return df.astype({"user_idx": int, "item_idx": int}).reset_index(drop=True)

    train = _apply_index(train)
    test  = _apply_index(test)

    # Align metadata to training vocab
    meta["item_idx"] = meta["parent_asin"].map(item2idx)
    meta = meta.dropna(subset=["item_idx"]).astype({"item_idx": int})

    log.info(f"  {len(user2idx):,} users  |  {len(item2idx):,} items")
    log.info(f"  Train: {len(train):,} rows  |  Test: {len(test):,} rows")
    return train, test, meta, user2idx, item2idx


# ── Step 4: Cold-start simulation ─────────────────────────────────────────────

def simulate_cold_start(test_df: pd.DataFrame,
                        k_pct: int,
                        min_alpha: int = MIN_ALPHA,
                        seed: int = 42) -> pd.DataFrame:
    """
    For a given K%, tag each test row as seed or ground-truth.

      - Only users with >= min_alpha test ratings are included
      - is_seed = True : exposed to the model as observed input
      - is_gt   = True : held out as ground truth for evaluation
      - K=0            : fully cold, no seed at all

    Parameters
    ----------
    test_df   : pandas DataFrame with [user_idx, item_idx, rating]
    k_pct     : 0 | 2 | 5 | 10 | 20  (% of user history as seed)
    min_alpha : minimum test ratings per user to be included
    seed      : random seed for reproducibility
    """
    rng = np.random.default_rng(seed)

    user_counts = test_df.groupby("user_idx")["rating"].count()
    eligible    = user_counts[user_counts >= min_alpha].index
    df          = test_df[test_df["user_idx"].isin(eligible)].copy()

    df["is_seed"] = False
    df["is_gt"]   = False

    for uid, group in df.groupby("user_idx"):
        n        = len(group)
        n_seed   = math.floor(n * k_pct / 100) if k_pct > 0 else 0
        shuffled = rng.permutation(group.index.tolist())
        df.loc[shuffled[:n_seed], "is_seed"] = True
        df.loc[shuffled[n_seed:], "is_gt"]   = True

    log.info(f"  K={k_pct}%: {len(eligible):,} eligible users, "
             f"{df['is_seed'].sum():,} seed rows, "
             f"{df['is_gt'].sum():,} gt rows")
    return df.reset_index(drop=True)


# ── Save / Load ────────────────────────────────────────────────────────────────

def save_processed(train, test, meta, user2idx, item2idx,
                   proc_dir: Path = PROC_DIR):
    proc_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(proc_dir / "train_interactions.parquet", index=False)
    test.to_parquet(proc_dir  / "test_interactions.parquet",  index=False)
    meta.to_parquet(proc_dir  / "items.parquet",              index=False)
    with open(proc_dir / "user2idx.pkl", "wb") as f:
        pickle.dump(user2idx, f)
    with open(proc_dir / "item2idx.pkl", "wb") as f:
        pickle.dump(item2idx, f)
    log.info(f"Saved processed data to {proc_dir}/")


def load_processed(proc_dir: Path = PROC_DIR):
    """Return (train_df, test_df, items_df, user2idx, item2idx)."""
    train = pd.read_parquet(proc_dir / "train_interactions.parquet")
    test  = pd.read_parquet(proc_dir / "test_interactions.parquet")
    items = pd.read_parquet(proc_dir / "items.parquet")
    with open(proc_dir / "user2idx.pkl", "rb") as f:
        user2idx = pickle.load(f)
    with open(proc_dir / "item2idx.pkl", "rb") as f:
        item2idx = pickle.load(f)
    return train, test, items, user2idx, item2idx


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_preprocessing(k: float = 0.2,
                      proc_dir: Path = PROC_DIR) -> dict:
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    k : float
        Test split ratio. e.g. 0.2 means 20% of each user's
        ratings go to the test set (same as friend's k parameter).
    """
    spark = get_spark()
    spark.catalog.clearCache()

    # 1. Load reviews, filter users, split train/test
    train_spark, test_spark = load_and_split(spark, k)

    # 2. Load and clean metadata
    meta_spark = load_and_clean_meta(spark, train_spark, test_spark)

    # 3. Collect to pandas + build integer indices
    train, test, meta, user2idx, item2idx = to_pandas_and_index(
        train_spark, test_spark, meta_spark
    )

    # 4. Save
    save_processed(train, test, meta, user2idx, item2idx, proc_dir)

    stats = {
        "n_users": len(user2idx),
        "n_items": len(item2idx),
        "n_train": len(train),
        "n_test":  len(test),
    }
    log.info(f"Preprocessing complete: {stats}")
    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    stats = run_preprocessing(k=0.2)
    print(stats)