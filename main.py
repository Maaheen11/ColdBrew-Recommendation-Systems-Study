"""
main.py
-------
Full pipeline combining friend's DataProcesser/ALS/SementicEmbedding
with our DropoutNet, cold-start simulation, and evaluation.

Steps
-----
  1. preprocess  — uses friend's DataProcessing class to load + clean data,
                   then adds integer indices and saves to data/processed/
  2. embeddings  — Sentence-BERT item embeddings (memory-mapped file)
  3. train       — ALS (friend's PySpark), Semantic (friend's SBERT),
                   DropoutNet (our two-tower MLP using both)
  4. evaluate    — cold-start simulation K ∈ {0,2,5,10,20}, Recall@10 / NDCG@10
  5. analyze     — degradation curves, crossover detection, plots

Quick-start
-----------
  python main.py              # full run (uses k=0.2 test split)
  python main.py --skip-preprocess --skip-embeddings --skip-train  # just re-evaluate
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src/ to path so all modules can import each other cleanly
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.preprocess       import run_preprocessing, load_processed, COLD_START_LEVELS
from src.embeddings       import build_item_embeddings, load_item_embeddings
from src.ALS        import ALSModel
from src.SementicEmbedding   import SemanticModel
from src.model_dropoutnet import DropoutNetModel
from src.evaluate         import run_full_evaluation
from src.analyze          import run_analysis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROC_DIR    = Path("data/processed")
RESULTS_DIR = Path("results")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Cold-start benchmarking pipeline")
    p.add_argument("--k", type=float, default=0.2,
                   help="Test split ratio for friend's DataProcessing (default 0.2)")
    p.add_argument("--skip-preprocess",  action="store_true")
    p.add_argument("--skip-embeddings",  action="store_true")
    p.add_argument("--skip-train",       action="store_true")
    p.add_argument("--skip-evaluate",    action="store_true")
    p.add_argument("--skip-analyze",     action="store_true")
    p.add_argument("--top-k",    type=int, default=10)
    p.add_argument("--max-users", type=int, default=None,
                   help="Cap test users per K level (speeds up evaluation)")
    return p.parse_args()


# ── Step 1: Preprocessing ──────────────────────────────────────────────────────

def step_preprocess(k: float):
    """
    Uses friend's DataProcessing(k) to load and clean data,
    then adds integer indices and saves to data/processed/.
    """
    stats = run_preprocessing(k=k, proc_dir=PROC_DIR)
    log.info(f"Preprocessing stats: {stats}")
    return stats


# ── Step 2: Embeddings ─────────────────────────────────────────────────────────

def step_embeddings():
    """
    Encodes item text (from friend's _embedding_data_clean_) with SBERT.
    Writes to a memory-mapped file — never loads full matrix into RAM.
    """
    build_item_embeddings(proc_dir=PROC_DIR, force=False)


# ── Step 3: Training ───────────────────────────────────────────────────────────

def step_train():
    """
    Train all three models:
      - ALS:        friend's PySpark ALS → we extract numpy factors
      - Semantic:   friend's SBERT embeddings → we add cold-start recommend()
      - DropoutNet: our two-tower MLP using ALS factors + SBERT embeddings
    """
    from pyspark.sql import SparkSession
    from src.preprocess import get_spark

    train_df, test_df, items_df, user2idx, item2idx = load_processed(PROC_DIR)
    n_users = len(user2idx)
    n_items = len(item2idx)

    # Get or start Spark session
    spark = SparkSession.getActiveSession() or get_spark()

    # Convert pandas back to Spark for friend's ALS and Semantic classes
    train_spark = spark.createDataFrame(
        train_df[["user_id", "parent_asin", "rating"]]
    )
    test_spark = spark.createDataFrame(
        test_df[["user_id", "parent_asin", "rating"]]
    )
    # meta needs embedding_text column (friend's SementicEmbedding column name)
    meta_spark = spark.createDataFrame(
        items_df[["parent_asin", "text"]].rename(columns={"text": "embedding_text"})
    )

    # ── ALS ────────────────────────────────────────────────────────────────────
    als_path = PROC_DIR / "als_model.pkl"
    if als_path.exists():
        log.info("ALS model found — loading from disk.")
        als_model = ALSModel.load(als_path)
    else:
        als_model = ALSModel(rank=30, max_iter=10, reg_param=0.09)
        als_model.train(
            train_df=train_df,
            train_spark=train_spark,
            n_users=n_users,
            n_items=n_items,
            user2idx=user2idx,
            item2idx=item2idx,
        )
        als_model.save(als_path)

    # ── Semantic ────────────────────────────────────────────────────────────────
    sem_path = PROC_DIR / "semantic_model.pkl"
    if sem_path.exists():
        log.info("Semantic model found — loading from disk.")
        sem_model = SemanticModel.load(sem_path)
    else:
        sem_model = SemanticModel(top_n=10)
        sem_model.train(
            train_df=train_df,
            n_items=n_items,
            train_spark=train_spark,
            meta_spark=meta_spark,
            proc_dir=PROC_DIR,
        )
        sem_model.save(sem_path)

    # ── DropoutNet ──────────────────────────────────────────────────────────────
    dnet_path = PROC_DIR / "dropoutnet_model.pt"
    if dnet_path.exists():
        log.info("DropoutNet found — loading from disk.")
        dnet_model = DropoutNetModel.load(dnet_path)
    else:
        item_emb = load_item_embeddings(PROC_DIR)

        # Use numpy factors extracted from friend's PySpark ALS
        user_factors = als_model.user_factors   # (n_users, rank)
        item_factors = als_model.item_factors   # (n_items, rank)

        dnet_model = DropoutNetModel(epochs=50, batch_size=1024, hidden_dim=512)
        dnet_model.train(
            train_df=train_df,
            user_factors=user_factors,
            item_factors=item_factors,
            item_content=item_emb,
            proc_dir=PROC_DIR,
        )
        dnet_model.save(dnet_path)

    return als_model, sem_model, dnet_model


# ── Step 4: Evaluation ─────────────────────────────────────────────────────────

def step_evaluate(als_model, sem_model, dnet_model,
                  top_k: int, max_users: int | None):
    _, test_df, _, _, _ = load_processed(PROC_DIR)

    return run_full_evaluation(
        als_model        = als_model,
        semantic_model   = sem_model,
        dropoutnet_model = dnet_model,
        test_df          = test_df,
        cold_start_levels= COLD_START_LEVELS,
        top_k            = top_k,
        max_users        = max_users,
        results_dir      = RESULTS_DIR,
    )


# ── Step 5: Analysis ───────────────────────────────────────────────────────────

def step_analyze(top_k: int):
    run_analysis(results_dir=RESULTS_DIR, top_k=top_k)


# ── Dataset statistics ─────────────────────────────────────────────────────────

def print_data_stats():
    train_df, test_df, items_df, user2idx, item2idx = load_processed(PROC_DIR)

    n_users      = len(user2idx)
    n_items      = len(item2idx)
    n_train      = len(train_df)
    n_test       = len(test_df)
    n_total      = n_train + n_test
    density      = n_total / (n_users * n_items) * 100
    avg_train_u  = n_train / n_users
    avg_test_u   = n_test  / n_users
    avg_train_i  = n_train / n_items
    n_items_meta = len(items_df)

    print("\n" + "═" * 55)
    print("  DATASET STATISTICS")
    print("═" * 55)
    print(f"  Users                    : {n_users:>10,}")
    print(f"  Items (interaction vocab): {n_items:>10,}")
    print(f"  Items (with metadata)    : {n_items_meta:>10,}")
    print(f"  Total interactions       : {n_total:>10,}")
    print(f"    Train                  : {n_train:>10,}  ({n_train/n_total*100:.1f}%)")
    print(f"    Test                   : {n_test:>10,}  ({n_test/n_total*100:.1f}%)")
    print(f"  Interaction density      : {density:>10.4f}%")
    print(f"  Avg train interactions/user : {avg_train_u:>7.1f}")
    print(f"  Avg test  interactions/user : {avg_test_u:>7.1f}")
    print(f"  Avg train interactions/item : {avg_train_i:>7.1f}")
    print(f"  Rating scale             : 1 – 5")
    print(f"  Relevance threshold      : ≥ 4 stars")
    print("═" * 55 + "\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if not args.skip_preprocess:
        log.info("━━━ STEP 1: Preprocessing ━━━")
        step_preprocess(k=args.k)
    else:
        log.info("Skipping preprocessing.")

    print_data_stats()

    if not args.skip_embeddings:
        log.info("━━━ STEP 2: Building SBERT item embeddings ━━━")
        step_embeddings()
    else:
        log.info("Skipping embeddings.")

    if not args.skip_train:
        log.info("━━━ STEP 3: Training models ━━━")
        als_model, sem_model, dnet_model = step_train()
    else:
        log.info("Loading saved models ...")
        als_model  = ALSModel.load(PROC_DIR / "als_model.pkl")
        sem_model  = SemanticModel.load(PROC_DIR / "semantic_model.pkl")
        dnet_model = DropoutNetModel.load(PROC_DIR / "dropoutnet_model.pt")

    if not args.skip_evaluate:
        log.info("━━━ STEP 4: Cold-start evaluation ━━━")
        step_evaluate(als_model, sem_model, dnet_model,
                      top_k=args.top_k, max_users=args.max_users)
    else:
        log.info("Skipping evaluation.")

    if not args.skip_analyze:
        log.info("━━━ STEP 5: Analysis & plots ━━━")
        step_analyze(top_k=args.top_k)
    else:
        log.info("Skipping analysis.")

    log.info("Pipeline complete. Check results/")


if __name__ == "__main__":
    main()