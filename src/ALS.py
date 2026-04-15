"""
model_als.py
------------
ALS collaborative filtering model.

Combines friend's ALS.py logic with the cold-start recommendation
interface needed by evaluate.py.

Contains everything in one class:
  - train()         : PySpark ALS training (friend's train_asl logic)
  - test()          : generate rating predictions (friend's test_asl logic)
  - rating_eval()   : RMSE, MAE, coverage    (friend's asl_eval logic)
  - ranking_eval()  : Recall@K, NDCG@K       (friend's ranking_eval logic)
  - recommend()     : cold-start top-K interface for evaluate.py  (our addition)
  - fold_in_user()  : approximate user vector from seed items     (our addition)
  - get_numpy_factors() : extract numpy matrices for DropoutNet   (our addition)
  - save() / load() : persist between runs                        (our addition)
"""

import logging
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.sql import functions as F
from pyspark.sql.functions import col

log = logging.getLogger(__name__)

PROC_DIR = Path("data/processed")


class ALSModel:

    def __init__(self,
                 rank: int         = 30,
                 max_iter: int     = 10,
                 reg_param: float  = 0.09,
                 cold_start_strategy: str = "drop"):
        self.rank                 = rank
        self.max_iter             = max_iter
        self.reg_param            = reg_param
        self.cold_start_strategy  = cold_start_strategy

        # Set after train()
        self.als_model_spark = None   # fitted PySpark ALSModel
        self.pipeline_model  = None   # fitted PySpark Pipeline (StringIndexer)
        self.user_factors    = None   # numpy (n_users, rank) for DropoutNet
        self.item_factors    = None   # numpy (n_items, rank) for DropoutNet
        self.popularity      = None   # numpy (n_items,) fallback for K=0

    # ── Training (friend's train_asl logic) ───────────────────────────────────

    def train(self, train_df: pd.DataFrame,
              train_spark=None,
              n_users: int = None,
              n_items: int = None,
              user2idx: dict = None,
              item2idx: dict = None):
        """
        Train PySpark ALS on the training data.

        Parameters
        ----------
        train_df    : pandas DataFrame [user_id, parent_asin, rating, user_idx, item_idx]
        train_spark : Spark DataFrame version — if None, converted from train_df
        n_users     : total user vocab size (from preprocess.py user2idx)
        n_items     : total item vocab size (from preprocess.py item2idx)
        user2idx    : dict mapping user_id -> user_idx (required for correct factor alignment)
        item2idx    : dict mapping parent_asin -> item_idx (required for correct factor alignment)
        """
        from pyspark.sql import SparkSession
        spark = SparkSession.getActiveSession()

        if train_spark is None:
            log.info("Converting pandas train_df to Spark ...")
            train_spark = spark.createDataFrame(
                train_df[["user_id", "parent_asin", "rating"]]
            )

        # StringIndexer — friend's approach to map string IDs to integers
        user_indexer = StringIndexer(
            inputCol="user_id", outputCol="user_index", handleInvalid="skip"
        )
        item_indexer = StringIndexer(
            inputCol="parent_asin", outputCol="item_index", handleInvalid="skip"
        )
        pipeline            = Pipeline(stages=[user_indexer, item_indexer])
        self.pipeline_model = pipeline.fit(train_spark)
        df_train_indexed    = self.pipeline_model.transform(train_spark)

        log.info(f"Training ALS (rank={self.rank}, iter={self.max_iter}, "
                 f"reg={self.reg_param}) ...")

        als = ALS(
            maxIter             = self.max_iter,
            regParam            = self.reg_param,
            rank                = self.rank,
            userCol             = "user_index",
            itemCol             = "item_index",
            ratingCol           = "rating",
            coldStartStrategy   = self.cold_start_strategy,
        )
        self.als_model_spark = als.fit(df_train_indexed)

        # Extract numpy factors for DropoutNet
        _n_users = n_users or int(train_df["user_idx"].max()) + 1
        _n_items = n_items or int(train_df["item_idx"].max()) + 1
        self.user_factors, self.item_factors = self.get_numpy_factors(
            _n_users, _n_items, user2idx=user2idx, item2idx=item2idx
        )

        # Popularity fallback for K=0 cold-start
        counts          = train_df["item_idx"].value_counts()
        self.popularity = np.zeros(_n_items, dtype=np.float32)
        for idx, cnt in counts.items():
            self.popularity[int(idx)] = cnt

        log.info("ALS training complete.")

    # ── Testing (friend's test_asl logic) ─────────────────────────────────────

    def test(self, test_spark):
        """Generate rating predictions on the test Spark DataFrame."""
        df_indexed  = self.pipeline_model.transform(test_spark)
        predictions = self.als_model_spark.transform(df_indexed)
        return predictions

    # ── Rating evaluation (friend's asl_eval logic) ───────────────────────────

    def rating_eval(self, predictions, test_spark):
        """
        Compute RMSE, MAE, and coverage on rating predictions.
        Friend's original evaluation — kept as-is.
        """
        pred     = predictions.filter(col("prediction").isNotNull())
        rmse     = RegressionEvaluator(
            metricName="rmse", labelCol="rating", predictionCol="prediction"
        ).evaluate(pred)
        mae      = RegressionEvaluator(
            metricName="mae", labelCol="rating", predictionCol="prediction"
        ).evaluate(pred)
        coverage = pred.count() / test_spark.count()
        return rmse, mae, coverage

    # ── Ranking evaluation (friend's ranking_eval logic) ──────────────────────

    def ranking_eval(self, test_spark, top_k: int = 10,
                     relevance_threshold: float = 4.0):
        """
        Compute Recall@K and NDCG@K using PySpark recommendForUserSubset.
        Friend's original ranking evaluation — kept as-is.
        """
        indexed_test  = self.pipeline_model.transform(test_spark).select(
            "user_id", "parent_asin", "user_index", "item_index", "rating"
        )
        relevant_test = indexed_test.filter(col("rating") >= relevance_threshold)
        eval_users    = relevant_test.select("user_id", "user_index").distinct()

        if eval_users.count() == 0:
            return 0.0, 0.0

        recommendations = self.als_model_spark.recommendForUserSubset(
            eval_users.select("user_index"), top_k
        )

        rec_items = (
            recommendations
            .join(eval_users, on="user_index", how="inner")
            .select("user_id", F.posexplode("recommendations").alias("rank_idx", "rec"))
            .select(
                "user_id",
                (col("rank_idx") + F.lit(1)).alias("rank"),
                col("rec.item_index").alias("item_index"),
            )
        )

        actual_items = relevant_test.groupBy("user_id").agg(
            F.collect_set("item_index").alias("actual_items")
        )

        predicted_items = rec_items.groupBy("user_id").agg(
            F.expr(
                "transform("
                "sort_array(collect_list(named_struct('rank', rank, 'item_index', item_index))), "
                "x -> x.item_index"
                ")"
            ).alias("predicted_items")
        )

        ranking_input = (
            actual_items
            .join(predicted_items, on="user_id", how="inner")
            .select("predicted_items", "actual_items")
        )

        ranking_rows = ranking_input.collect()
        if not ranking_rows:
            return 0.0, 0.0

        recall_scores, ndcg_scores = [], []
        for row in ranking_rows:
            preds   = row["predicted_items"] or []
            actuals = set(row["actual_items"] or [])
            if not actuals:
                continue

            hits = sum(1 for item in preds[:top_k] if item in actuals)
            recall_scores.append(hits / len(actuals))

            dcg = sum(
                1.0 / math.log2(i + 2)
                for i, item in enumerate(preds[:top_k]) if item in actuals
            )
            idcg = sum(
                1.0 / math.log2(i + 2)
                for i in range(min(len(actuals), top_k))
            )
            ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

        if not recall_scores:
            return 0.0, 0.0

        return sum(recall_scores) / len(recall_scores), \
               sum(ndcg_scores)   / len(ndcg_scores)

    # ── Extract numpy factors for DropoutNet ──────────────────────────────────

    def get_numpy_factors(self, n_users: int, n_items: int,
                          user2idx: dict = None, item2idx: dict = None):
        """
        Extract user and item latent factor matrices from the PySpark ALS model
        as numpy arrays so DropoutNet can use them as preference inputs.

        IMPORTANT: PySpark StringIndexer assigns indices by frequency
        (most frequent = 0), which differs from our alphabetically-sorted
        user_idx/item_idx. user2idx and item2idx are required to correctly
        remap ALS factors to our index space.

        Returns
        -------
        user_factors : np.ndarray (n_users, rank)  aligned to user_idx
        item_factors : np.ndarray (n_items, rank)  aligned to item_idx
        """
        log.info("Extracting numpy factor matrices from PySpark ALS ...")

        user_df = self.als_model_spark.userFactors.orderBy("id").toPandas()
        item_df = self.als_model_spark.itemFactors.orderBy("id").toPandas()

        raw_user = np.vstack(user_df["features"].values).astype(np.float32)
        raw_item = np.vstack(item_df["features"].values).astype(np.float32)

        user_factors = np.zeros((n_users, self.rank), dtype=np.float32)
        item_factors = np.zeros((n_items, self.rank), dtype=np.float32)

        if user2idx is not None and item2idx is not None and self.pipeline_model is not None:
            # StringIndexer labels[i] is the original string that was assigned
            # internal ALS index i. Map each back to our user_idx / item_idx.
            user_labels = self.pipeline_model.stages[0].labels
            item_labels = self.pipeline_model.stages[1].labels

            for spark_idx, uid in enumerate(user_labels):
                our_idx = user2idx.get(uid)
                if our_idx is not None and spark_idx < len(raw_user):
                    user_factors[our_idx] = raw_user[spark_idx]

            for spark_idx, asin in enumerate(item_labels):
                our_idx = item2idx.get(asin)
                if our_idx is not None and spark_idx < len(raw_item):
                    item_factors[our_idx] = raw_item[spark_idx]

            log.info("  Factor matrices remapped via StringIndexer labels.")
        else:
            log.warning(
                "user2idx/item2idx not provided — factor matrices may be misaligned "
                "with item_idx/user_idx. Pass these dicts to train() for correct results."
            )
            def _pad(mat, target, dim):
                if mat.shape[0] < target:
                    pad = np.zeros((target - mat.shape[0], dim), dtype=np.float32)
                    mat = np.vstack([mat, pad])
                return mat[:target]
            user_factors = _pad(raw_user, n_users, self.rank)
            item_factors = _pad(raw_item, n_items, self.rank)

        log.info(f"  user_factors: {user_factors.shape}")
        log.info(f"  item_factors: {item_factors.shape}")
        return user_factors, item_factors

    # ── Fold-in: approximate user vector from K seed items ────────────────────

    def fold_in_user(self, seed_item_indices: list[int],
                     seed_ratings: list[float],
                     regularization: float = 0.01,
                     alpha: float = 40.0) -> np.ndarray:
        """
        Compute an approximate user latent vector by solving one ALS update
        step using only the K seed items (fold-in technique).

        Used by recommend() and by evaluate.py for DropoutNet's als_user_vec.
        Returns a zero vector at K=0 (no history at all).
        """
        if not seed_item_indices:
            return np.zeros(self.rank, dtype=np.float32)

        Y     = self.item_factors[seed_item_indices]              # (K, rank)
        cu    = 1.0 + alpha * np.array(seed_ratings, dtype=np.float32)
        YtCuY = Y.T @ (Y * cu[:, None]) + regularization * np.eye(self.rank)
        YtCup = Y.T @ cu
        try:
            vec = np.linalg.solve(YtCuY, YtCup)
        except np.linalg.LinAlgError:
            vec = np.linalg.lstsq(YtCuY, YtCup, rcond=None)[0]
        return vec.astype(np.float32)

    # ── Cold-start recommend() for evaluate.py ────────────────────────────────

    def recommend(self, seed_item_indices: list[int],
                  seed_ratings: list[float],
                  top_k: int = 10,
                  exclude_seen: bool = True) -> list[int]:
        """
        Return top-K item_idx values for a cold-start user.

        K=0 : no history → returns most popular items
        K>0 : folds in the seed items to approximate the user vector,
              then scores all items via dot product with item_factors
        """
        assert self.item_factors is not None, "Call train() first."

        if not seed_item_indices:
            scores = self.popularity.copy()
        else:
            user_vec = self.fold_in_user(seed_item_indices, seed_ratings)
            scores   = self.item_factors @ user_vec

        if exclude_seen and seed_item_indices:
            scores[seed_item_indices] = -np.inf

        top_idx = np.argpartition(scores, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return top_idx.tolist()

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "user_factors": self.user_factors,
                "item_factors": self.item_factors,
                "popularity":   self.popularity,
                "config": {
                    "rank":      self.rank,
                    "max_iter":  self.max_iter,
                    "reg_param": self.reg_param,
                },
            }, f)
        log.info(f"ALSModel saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "ALSModel":
        with open(path, "rb") as f:
            state = pickle.load(f)
        cfg = state["config"]
        m   = cls(rank=cfg["rank"], max_iter=cfg["max_iter"],
                  reg_param=cfg["reg_param"])
        m.user_factors = state["user_factors"]
        m.item_factors = state["item_factors"]
        m.popularity   = state["popularity"]
        log.info(f"ALSModel loaded from {path}")
        return m


# ── Standalone run (mirrors friend's __main__ block) ──────────────────────────

if __name__ == "__main__":
    import logging
    from preprocess import run_preprocessing, load_processed
    from pyspark.sql import SparkSession

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    # Run for multiple k values like friend's original __main__
    rmse_holder, mae_holder, recall_holder, ndcg_holder = [], [], [], []

    for k in [i / 10 for i in range(1, 10)]:
        run_preprocessing(k=k)
        train_df, test_df, _, user2idx, item2idx = load_processed()

        spark       = SparkSession.getActiveSession()
        train_spark = spark.createDataFrame(train_df[["user_id", "parent_asin", "rating"]])
        test_spark  = spark.createDataFrame(test_df[["user_id",  "parent_asin", "rating"]])

        model = ALSModel()
        model.train(train_df, train_spark,
                    n_users=len(user2idx), n_items=len(item2idx),
                    user2idx=user2idx, item2idx=item2idx)

        predictions        = model.test(test_spark)
        rmse, mae, _       = model.rating_eval(predictions, test_spark)
        recall_k, ndcg_k   = model.ranking_eval(test_spark)

        rmse_holder.append(rmse)
        mae_holder.append(mae)
        recall_holder.append(recall_k)
        ndcg_holder.append(ndcg_k)

    print(f"rmse:      {rmse_holder}")
    print(f"mae:       {mae_holder}")
    print(f"recall@10: {recall_holder}")
    print(f"ndcg@10:   {ndcg_holder}")