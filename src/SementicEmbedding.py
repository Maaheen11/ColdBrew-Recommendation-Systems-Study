"""
model_semantic.py
-----------------
Semantic embedding-based recommendation model.

Combines friend's SementicEmbedding.py logic with the cold-start
recommendation interface needed by evaluate.py.

Contains everything in one class:
  - train()                : SBERT encode all items + build user profiles
                             (friend's generate_item_embedding + user_history)
  - recommand_for_user()   : recommend from full user profile by user_id
                             (friend's original method — kept as-is)
  - embedding_eval()       : precision, recall, hit-rate
                             (friend's original evaluation — kept as-is)
  - recommend()            : cold-start interface using K seed item indices
                             for evaluate.py                (our addition)
  - save() / load()        : persist between runs           (our addition)
"""

 

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

PROC_DIR = Path("data/processed")


class SemanticModel:

    def __init__(self,
                 model_name: str = "all-MiniLM-L6-v2",
                 top_n: int = 10):
        self.model_name = model_name
        self.top_n      = top_n

        # Set after train()
        self.sbert_model       = None   # SentenceTransformer instance
        self.item_embedding_df = None   # pandas: parent_asin, embedding, item_idx
        self.asin_to_embedding = None   # dict: parent_asin -> embedding vector
        self.user_profile      = None   # dict: user_id -> mean embedding vector
        self.popularity        = None   # numpy (n_items,) for K=0 fallback
        self.n_items           = None
        self.item_matrix       = None   # numpy (n_items, emb_dim) precomputed for fast scoring
        self.item_norms        = None   # numpy (n_items,) precomputed L2 norms

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, train_df: pd.DataFrame,
              n_items: int,
              train_spark=None,
              meta_spark=None,
              proc_dir: Path = PROC_DIR):
        """
        Encode all items with SBERT and build user profiles from training history.

        Parameters
        ----------
        train_df    : pandas DataFrame [user_id, parent_asin, rating, user_idx, item_idx]
        n_items     : total item vocabulary size
        train_spark : Spark DataFrame of train_df (optional — created if not given)
        meta_spark  : Spark DataFrame with [parent_asin, embedding_text] (optional)
        proc_dir    : where items.parquet is saved
        """
        from pyspark.sql import SparkSession
        spark = SparkSession.getActiveSession()

        if train_spark is None:
            train_spark = spark.createDataFrame(
                train_df[["user_id", "parent_asin", "rating"]]
            )

        if meta_spark is None:
            items_df   = pd.read_parquet(proc_dir / "items.parquet")
            meta_spark = spark.createDataFrame(
                items_df[["parent_asin", "text"]].rename(
                    columns={"text": "embedding_text"}
                )
            )

        self.n_items    = n_items
        self.sbert_model = SentenceTransformer(self.model_name)

        # Step 1: generate item embeddings (friend's generate_item_embedding)
        self._generate_item_embedding(meta_spark, proc_dir)

        # Step 2: build user profiles from full training history
        # (friend's user_history)
        self._build_user_profiles(train_spark)

        # Step 3: popularity fallback for K=0 cold-start
        counts          = train_df["item_idx"].value_counts()
        self.popularity = np.zeros(n_items, dtype=np.float32)
        for idx, cnt in counts.items():
            self.popularity[int(idx)] = cnt

        log.info("SemanticModel training complete.")

    def _generate_item_embedding(self, meta_spark, proc_dir: Path):
        """
        Encode all items using SBERT.
        Absorbed from friend's generate_item_embedding().
        """
        log.info("Generating item embeddings with SBERT ...")

        # Collect meta to pandas (friend's approach)
        meta_df = meta_spark.toPandas()

        # Encode with SBERT (friend's encode call)
        embeddings = self.sbert_model.encode(
            meta_df["embedding_text"].tolist(),
            show_progress_bar=True,
        )

        meta_df["embedding"] = list(embeddings)

        # Store lookup dict (friend's asin_to_embedding)
        self.asin_to_embedding = dict(
            zip(meta_df["parent_asin"], meta_df["embedding"])
        )

        # Add item_idx for cold-start recommend()
        items_pd = pd.read_parquet(proc_dir / "items.parquet")[
            ["parent_asin", "item_idx"]
        ]
        self.item_embedding_df = meta_df.merge(
            items_pd, on="parent_asin", how="left"
        )

        log.info(f"  Encoded {len(meta_df):,} items.")
        self._build_item_matrix()

    def _build_user_profiles(self, train_spark):
        """
        Build mean embedding profile per user from training history.
        Absorbed from friend's user_history().
        """
        log.info("Building user profiles from training history ...")

        user_history_df = (
            train_spark.groupBy("user_id")
            .agg(F.collect_list("parent_asin").alias("parent_asin"))
            .toPandas()
        )

        user_profile = {}
        for _, row in user_history_df.iterrows():
            user_id  = row["user_id"]
            item_list = row["parent_asin"]

            # Mean of item embeddings (friend's approach)
            item_vectors = [
                self.asin_to_embedding[item]
                for item in item_list
                if item in self.asin_to_embedding
            ]

            if len(item_vectors) > 0:
                user_profile[user_id] = np.mean(item_vectors, axis=0)

        self.user_profile = user_profile
        log.info(f"  Built profiles for {len(user_profile):,} users.")

    # ── Recommend for a known user (friend's recommand_for_user — kept as-is) ──

    def recommand_for_user(self, user_id: str, train_spark=None) -> list[str]:
        """
        Recommend top-N items for a known user using their full profile.
        Friend's original method — returns parent_asin strings.

        Used by embedding_eval() and standalone runs.
        """
        if user_id not in self.user_profile:
            return []

        user_vector = self.user_profile[user_id]

        # Exclude items already seen in training (friend's approach)
        if train_spark is not None:
            seen_items = (
                train_spark.filter(F.col("user_id") == user_id)
                .select("parent_asin")
                .distinct()
                .toPandas()["parent_asin"]
                .tolist()
            )
        else:
            seen_items = []

        seen_items = set(seen_items)
        candidates = self.item_embedding_df[
            ~self.item_embedding_df["parent_asin"].isin(seen_items)
        ].copy()

        if candidates.empty:
            return []

        item_matrix = np.vstack(candidates["embedding"].values)

        # Cosine similarity (friend's approach)
        user_norm  = np.linalg.norm(user_vector)
        item_norms = np.linalg.norm(item_matrix, axis=1)
        sims       = np.dot(item_matrix, user_vector) / (item_norms * user_norm + 1e-12)

        candidates = candidates.copy()
        candidates["score"] = sims
        recs = candidates.sort_values(by="score", ascending=False).head(self.top_n)

        return recs["parent_asin"].tolist()

    # ── Evaluation (friend's embedding_eval — kept as-is) ─────────────────────

    def embedding_eval(self, test_spark, train_spark=None) -> dict:
        """
        Evaluate using precision, recall, and hit-rate.
        Friend's original evaluation method.
        """
        test_truth_df = (
            test_spark.groupBy("user_id")
            .agg(F.collect_set("parent_asin").alias("parent_asin"))
            .toPandas()
        )

        precision_list, recall_list, hit_list = [], [], []

        for _, row in test_truth_df.iterrows():
            user_id   = row["user_id"]
            item_list = set(row["parent_asin"])

            recs     = self.recommand_for_user(user_id, train_spark)
            recs_set = set(recs)

            if len(recs) == 0:
                continue

            hits      = len(recs_set & item_list)
            precision = hits / self.top_n
            recall    = hits / len(item_list) if len(item_list) > 0 else 0
            hit_rate  = 1 if hits > 0 else 0

            precision_list.append(precision)
            recall_list.append(recall)
            hit_list.append(hit_rate)

        result = {
            "precision_at_k": float(np.mean(precision_list)) if precision_list else 0.0,
            "recall_at_k":    float(np.mean(recall_list))    if recall_list    else 0.0,
            "hit_rate_at_k":  float(np.mean(hit_list))       if hit_list       else 0.0,
        }
        return result

    # ── Precompute item matrix for fast scoring ───────────────────────────────

    def _build_item_matrix(self):
        """
        Build a contiguous numpy matrix (n_items, emb_dim) indexed by item_idx.
        Called once after encoding so recommend() never vstack()s per user.
        """
        valid = self.item_embedding_df.dropna(subset=["item_idx"])
        emb_dim = len(valid.iloc[0]["embedding"])
        self.item_matrix = np.zeros((self.n_items, emb_dim), dtype=np.float32)
        for _, row in valid.iterrows():
            self.item_matrix[int(row["item_idx"])] = row["embedding"]
        self.item_norms = np.linalg.norm(self.item_matrix, axis=1)
        log.info(f"  Item matrix precomputed: {self.item_matrix.shape}")

    # ── Cold-start recommend() for evaluate.py ────────────────────────────────

    def recommend(self, seed_item_indices: list[int],
                  seed_ratings: list[float],
                  top_k: int = 10,
                  exclude_seen: bool = True) -> list[int]:
        """
        Return top-K item_idx for a cold-start user defined by seed items.

        K=0 : no seed → return most popular items
        K>0 : build a rating-weighted embedding profile from seed items,
              score all items via a single matrix multiply (fast)
        """
        assert self.item_matrix is not None, "Call train() first."

        if not seed_item_indices:
            return self._popularity_top_k(top_k, exclude_seen, [])

        # Build rating-weighted profile from seed items using precomputed matrix
        seed_vecs = self.item_matrix[seed_item_indices]   # (K, emb_dim)
        weights   = np.array(seed_ratings, dtype=np.float32)
        profile   = (seed_vecs * weights[:, None]).sum(axis=0)
        norm      = np.linalg.norm(profile)
        if norm > 0:
            profile /= norm

        # Score all items in one matrix multiply — no per-call vstack
        sims = self.item_matrix @ profile / (self.item_norms + 1e-12)

        if exclude_seen and seed_item_indices:
            sims[seed_item_indices] = -np.inf

        top_idx = np.argpartition(sims, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]
        return top_idx.tolist()

    def _popularity_top_k(self, top_k: int,
                          exclude_seen: bool,
                          seen: list[int]) -> list[int]:
        scores = self.popularity.copy()
        if exclude_seen and seen:
            scores[seen] = -np.inf
        top_idx = np.argpartition(scores, -top_k)[-top_k:]
        return top_idx[np.argsort(scores[top_idx])[::-1]].tolist()

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Don't pickle the SentenceTransformer — it reloads fast from disk
        state = {
            "item_embedding_df": self.item_embedding_df,
            "asin_to_embedding": self.asin_to_embedding,
            "user_profile":      self.user_profile,
            "popularity":        self.popularity,
            "n_items":           self.n_items,
            "model_name":        self.model_name,
            "top_n":             self.top_n,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        log.info(f"SemanticModel saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "SemanticModel":
        with open(path, "rb") as f:
            state = pickle.load(f)
        m = cls(model_name=state["model_name"], top_n=state["top_n"])
        m.item_embedding_df = state["item_embedding_df"]
        m.asin_to_embedding = state["asin_to_embedding"]
        m.user_profile      = state["user_profile"]
        m.popularity        = state["popularity"]
        m.n_items           = state["n_items"]
        m._build_item_matrix()
        log.info(f"SemanticModel loaded from {path}")
        return m


# ── Standalone run (mirrors friend's __main__ block) ──────────────────────────

if __name__ == "__main__":
    import logging
    from preprocess import run_preprocessing, load_processed
    from pyspark.sql import SparkSession

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    k      = 0.2
    top_n  = 10

    run_preprocessing(k=k)
    train_df, test_df, items_df, user2idx, item2idx = load_processed()

    spark       = SparkSession.getActiveSession()
    train_spark = spark.createDataFrame(train_df[["user_id", "parent_asin", "rating"]])
    test_spark  = spark.createDataFrame(test_df[["user_id",  "parent_asin", "rating"]])

    model = SemanticModel(top_n=top_n)
    model.train(train_df, n_items=len(item2idx),
                train_spark=train_spark)

    result = model.embedding_eval(test_spark, train_spark)
    print(f"embedding result: {result}")