"""
evaluate.py
-----------
Cold-start evaluation using the simulation protocol from the paper.

For each K ∈ {0, 2, 5, 10, 20}:
  1. Call simulate_cold_start() to tag test rows as seed / ground-truth
  2. For each eligible test user:
       - Feed seed interactions to the model
       - Request top-10 recommendations
       - Compute Recall@10 and NDCG@10 against ground-truth items
  3. Average metrics across users and save to results/metrics.csv

Works with all three models:
  ALSModel         (src/ALS.py)
  SemanticModel    (src/SementicEmbedding.py)
  DropoutNetModel  (src/model_dropoutnet.py)
"""

import logging
import math
from pathlib import Path
from typing import Callable
import numpy as np
import pandas as pd
from tqdm import tqdm
from model_dropoutnet import fold_in_user
from preprocess import simulate_cold_start, COLD_START_LEVELS

log = logging.getLogger(__name__)

TOP_K       = 10
RESULTS_DIR = Path("results")
MIN_ALPHA   = 5    # minimum test rows per user to be included in evaluation


# ── Metric helpers ─────────────────────────────────────────────────────────────

def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    dcg  = sum(1.0 / math.log2(rank + 2)
               for rank, item in enumerate(recommended[:k]) if item in relevant)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


# ── Single-model evaluation ────────────────────────────────────────────────────

def evaluate_model(model_name: str,
                   recommend_fn: Callable,
                   test_df: pd.DataFrame,
                   cold_start_levels: list[int] = COLD_START_LEVELS,
                   top_k: int = TOP_K,
                   max_users: int | None = None,
                   als_model=None) -> pd.DataFrame:
    """
    Run cold-start evaluation for a single model across all K levels.

    Parameters
    ----------
    model_name        : label for output CSV (e.g. "ALS", "Semantic", "DropoutNet")
    recommend_fn      : model's recommend() method
    test_df           : raw test DataFrame (user_idx, item_idx, rating columns)
    cold_start_levels : K% values to evaluate
    top_k             : recommendation list length
    max_users         : cap on test users (for fast dev runs)
    als_model         : ALSModel instance; needed for DropoutNet fold-in, else None

    Returns
    -------
    DataFrame with columns: model, k_pct, recall, ndcg, n_users
    """
    records = []

    for k_pct in cold_start_levels:
        # simulate_cold_start tags each row with is_seed / is_gt
        sim_df = simulate_cold_start(test_df, k_pct=k_pct, min_alpha=MIN_ALPHA)

        users = sim_df["user_idx"].unique()
        if max_users is not None:
            rng   = np.random.default_rng(42)
            users = rng.choice(users, size=min(max_users, len(users)), replace=False)

        recalls, ndcgs = [], []

        for uid in tqdm(users, desc=f"{model_name} K={k_pct}%", leave=False):
            user_sim = sim_df[sim_df["user_idx"] == uid]

            seed_rows = user_sim[user_sim["is_seed"]]
            gt_rows   = user_sim[user_sim["is_gt"]]

            seed_indices = seed_rows["item_idx"].tolist()
            seed_ratings = seed_rows["rating"].tolist()
            gt_set       = set(gt_rows[gt_rows["rating"] >= 4]["item_idx"].tolist())

            if not gt_set:
                continue

            # Build keyword arguments for each model type
            kwargs = {}
            if als_model is not None:
                # DropoutNet needs an ALS fold-in user vector
                kwargs["als_user_vec"] = (
                    fold_in_user(seed_indices, seed_ratings, als_model.item_factors)
                    if seed_indices else None
                )

            recs = recommend_fn(seed_indices, seed_ratings, top_k=top_k, **kwargs)

            recalls.append(recall_at_k(recs, gt_set, top_k))
            ndcgs.append(ndcg_at_k(recs, gt_set, top_k))

        records.append({
            "model":   model_name,
            "k_pct":   k_pct,
            "recall":  float(np.mean(recalls)) if recalls else 0.0,
            "ndcg":    float(np.mean(ndcgs))   if ndcgs   else 0.0,
            "n_users": len(recalls),
        })
        log.info(
            f"  {model_name}  K={k_pct:2d}%  "
            f"Recall@{top_k}={records[-1]['recall']:.4f}  "
            f"NDCG@{top_k}={records[-1]['ndcg']:.4f}  "
            f"(n={records[-1]['n_users']})"
        )

    return pd.DataFrame(records)


# ── Full evaluation across all three models ────────────────────────────────────

def run_full_evaluation(als_model,
                        semantic_model,
                        dropoutnet_model,
                        test_df: pd.DataFrame,
                        cold_start_levels: list[int] = COLD_START_LEVELS,
                        top_k: int = TOP_K,
                        max_users: int | None = None,
                        results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    """
    Evaluate all three models and save results/metrics.csv.

    Parameters
    ----------
    als_model        : fitted ALSModel (src/model_als.py)
    semantic_model   : fitted SemanticModel (src/model_semantic.py)
    dropoutnet_model : fitted DropoutNetModel (src/model_dropoutnet.py)
    test_df          : pandas test DataFrame from load_processed()
    max_users        : cap test users per K level (useful for dev runs)
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    log.info("=== Evaluating ALS ===")
    all_results.append(evaluate_model(
        "ALS", als_model.recommend, test_df,
        cold_start_levels=cold_start_levels,
        top_k=top_k, max_users=max_users,
    ))

    log.info("=== Evaluating Semantic Embeddings ===")
    all_results.append(evaluate_model(
        "Semantic", semantic_model.recommend, test_df,
        cold_start_levels=cold_start_levels,
        top_k=top_k, max_users=max_users,
    ))

    log.info("=== Evaluating DropoutNet ===")
    all_results.append(evaluate_model(
        "DropoutNet", dropoutnet_model.recommend, test_df,
        cold_start_levels=cold_start_levels,
        top_k=top_k, max_users=max_users,
        als_model=als_model,   # needed for fold-in user vectors
    ))

    df = pd.concat(all_results, ignore_index=True)
    out_path = results_dir / "metrics.csv"
    df.to_csv(out_path, index=False)
    log.info(f"Results saved to {out_path}")
    return df