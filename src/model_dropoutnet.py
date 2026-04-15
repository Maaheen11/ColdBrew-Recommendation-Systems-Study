"""
model_dropoutnet.py
-------------------
DropoutNet: hybrid neural recommendation model.

Inputs it expects
-----------------
  user_factors  : ALS user latent matrix — extracted from teammate's trained
                  PySpark ALS model via _extract_als_factors()
  item_factors  : ALS item latent matrix (same source)
  item_content  : Sentence-BERT embeddings built by src/embeddings.py

Architecture (Volkovs et al., 2017)
-------------------------------------
  Two-tower MLP:
    user_tower( [user_pref | item_pref_avg] ) -> user_repr  (output_dim,)
    item_tower( [item_pref | item_content]  ) -> item_repr  (output_dim,)
    score = dot(user_repr, item_repr)

  During training, preference vectors are randomly zeroed (pref_dropout)
  to force the network to rely on content when interaction data is missing.

Cold-start inference
--------------------
  K=0 : user_pref = 0  (no history at all)
  K>0 : user_pref = ALS fold-in vector computed from seed items
  item_pref_avg is always the mean of seed items' ALS factors.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROC_DIR  = Path("data/processed")


# ── Helper: extract numpy factor matrices from PySpark ALS model ───────────────

def extract_als_factors(als_model_spark, n_users: int, n_items: int,
                        factors: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Pull user and item latent factor matrices out of a fitted PySpark ALS model
    and return them as numpy arrays aligned to user_idx / item_idx.

    Parameters
    ----------
    als_model_spark : fitted pyspark.ml.recommendation.ALSModel
    n_users, n_items : vocabulary sizes (from user2idx / item2idx)
    factors          : ALS rank (number of latent dimensions)

    Returns
    -------
    user_factors : np.ndarray (n_users, factors)
    item_factors : np.ndarray (n_items, factors)

    Note
    ----
    The PySpark ALS model uses integer indices (user_index, item_index)
    that were created by StringIndexer in ALS.py.  Those indices may differ
    from our user_idx / item_idx.  We align by collecting the factor DataFrames
    and sorting by the integer id column.
    """
    log.info("Extracting ALS factor matrices from PySpark model ...")

    # User factors: DataFrame with columns [id, features]
    user_df = als_model_spark.userFactors.orderBy("id").toPandas()
    item_df = als_model_spark.itemFactors.orderBy("id").toPandas()

    # features column is a list/array of floats
    user_factors = np.vstack(user_df["features"].values).astype(np.float32)
    item_factors = np.vstack(item_df["features"].values).astype(np.float32)

    # Pad to n_users / n_items in case some indices have no factors
    def _pad(mat, target_rows, dim):
        if mat.shape[0] < target_rows:
            pad = np.zeros((target_rows - mat.shape[0], dim), dtype=np.float32)
            mat = np.vstack([mat, pad])
        return mat[:target_rows]

    user_factors = _pad(user_factors, n_users, factors)
    item_factors = _pad(item_factors, n_items, factors)

    log.info(f"  user_factors: {user_factors.shape}, item_factors: {item_factors.shape}")
    return user_factors, item_factors


def fold_in_user(seed_item_indices: list[int],
                 seed_ratings: list[float],
                 item_factors: np.ndarray,
                 regularization: float = 0.01,
                 alpha: float = 40.0) -> np.ndarray:
    """
    Approximate a new user's ALS latent vector by solving one ALS step
    (the same fold-in logic used in model_als.py).

    Returns zero vector when seed is empty (K=0).
    """
    if not seed_item_indices:
        return np.zeros(item_factors.shape[1], dtype=np.float32)

    Y  = item_factors[seed_item_indices]           # (K, F)
    cu = 1.0 + alpha * np.array(seed_ratings, dtype=np.float32)  # (K,)
    YtCuY = Y.T @ (Y * cu[:, None]) + regularization * np.eye(Y.shape[1])
    YtCup = Y.T @ cu
    try:
        vec = np.linalg.solve(YtCuY, YtCup)
    except np.linalg.LinAlgError:
        vec = np.linalg.lstsq(YtCuY, YtCup, rcond=None)[0]
    return vec.astype(np.float32)


# ── Dataset ────────────────────────────────────────────────────────────────────

class DropoutNetDataset(Dataset):
    """
    Positive (user, item) pairs from training interactions.
    Negatives sampled uniformly at item level.
    Preference vectors are randomly zeroed with probability pref_dropout.
    """

    def __init__(self, interactions: np.ndarray,
                 user_factors: np.ndarray,
                 item_factors: np.ndarray,
                 item_content: np.ndarray,
                 pref_dropout: float = 0.5):
        self.interactions = interactions          # (N, 2) [user_idx, item_idx]
        self.user_factors = user_factors.astype(np.float32)
        self.item_factors = item_factors.astype(np.float32)
        self.item_content = item_content.astype(np.float32)
        self.pref_dropout = pref_dropout
        self.n_items      = len(item_factors)

        # Pre-compute per-user mean item preference so the user tower
        # sees the same distribution at training time as at inference time.
        n_users = len(user_factors)
        pref_dim = item_factors.shape[1]
        user_sum = np.zeros((n_users, pref_dim), dtype=np.float32)
        user_cnt = np.zeros(n_users, dtype=np.int32)
        for uid, iid in interactions:
            user_sum[uid] += item_factors[iid]
            user_cnt[uid] += 1
        self.user_item_avg = user_sum / np.maximum(user_cnt, 1).astype(np.float32)[:, None]

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        uid, iid = int(self.interactions[idx, 0]), int(self.interactions[idx, 1])

        u_pref     = self.user_factors[uid].copy()
        i_pref_avg = self.user_item_avg[uid].copy()   # mean history pref for user tower
        i_pref     = self.item_factors[iid].copy()
        # Preference dropout (core DropoutNet idea)
        if np.random.rand() < self.pref_dropout:
            u_pref[:] = 0.0
        if np.random.rand() < self.pref_dropout:
            i_pref_avg[:] = 0.0
        if np.random.rand() < self.pref_dropout:
            i_pref[:] = 0.0
        i_cont = self.item_content[iid]

        # Uniform negative sample — exclude the positive item
        neg_iid = np.random.randint(0, self.n_items)
        while neg_iid == iid:
            neg_iid = np.random.randint(0, self.n_items)
        neg_pref = self.item_factors[neg_iid].copy()
        neg_cont = self.item_content[neg_iid]
        if np.random.rand() < self.pref_dropout:
            neg_pref[:] = 0.0

        return (
            torch.tensor(u_pref),
            torch.tensor(i_pref_avg),
            torch.tensor(i_pref),
            torch.tensor(i_cont),
            torch.tensor(neg_pref),
            torch.tensor(neg_cont),
        )


# ── Neural network ─────────────────────────────────────────────────────────────

class _DropoutNetNN(nn.Module):
    def __init__(self, pref_dim: int, content_dim: int,
                 hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.user_tower = nn.Sequential(
            nn.Linear(pref_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
        )
        self.item_tower = nn.Sequential(
            nn.Linear(pref_dim + content_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

    def encode_user(self, u_pref: torch.Tensor,
                    i_pref_avg: torch.Tensor) -> torch.Tensor:
        return self.user_tower(torch.cat([u_pref, i_pref_avg], dim=-1))

    def encode_item(self, i_pref: torch.Tensor,
                    i_cont: torch.Tensor) -> torch.Tensor:
        return self.item_tower(torch.cat([i_pref, i_cont], dim=-1))

    def forward(self, u_pref, i_pref_avg, pos_i_pref, pos_i_cont, neg_i_pref, neg_i_cont):
        user_repr = self.encode_user(u_pref, i_pref_avg)
        pos_repr  = self.encode_item(pos_i_pref, pos_i_cont)
        neg_repr  = self.encode_item(neg_i_pref, neg_i_cont)
        return (user_repr * pos_repr).sum(-1), (user_repr * neg_repr).sum(-1)


# ── Model wrapper ──────────────────────────────────────────────────────────────

class DropoutNetModel:
    def __init__(self,
                 hidden_dim: int   = 512,
                 output_dim: int   = 128,
                 pref_dropout: float = 0.5,
                 epochs: int       = 20,
                 batch_size: int   = 1024,
                 lr: float         = 1e-3,
                 random_state: int = 42):
        self.hidden_dim   = hidden_dim
        self.output_dim   = output_dim
        self.pref_dropout = pref_dropout
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.lr           = lr
        self.random_state = random_state

        self.nn           = None
        self.item_factors = None   # kept for fold-in at inference
        self.item_content = None
        self._repr_path   = None   # path to memmap item repr file
        self.popularity   = None   # numpy (n_items,) fallback for K=0

    # ── Training ───────────────────────────────────────────────────────────────

    def train(self, train_df: pd.DataFrame,
              user_factors: np.ndarray,
              item_factors: np.ndarray,
              item_content: np.ndarray,
              proc_dir: Path = PROC_DIR):
        """
        Parameters
        ----------
        train_df      : pandas DataFrame with [user_idx, item_idx, rating]
        user_factors  : ALS user matrix (n_users, F) — from extract_als_factors()
        item_factors  : ALS item matrix (n_items, F)
        item_content  : SBERT embeddings (n_items, C) — from embeddings.py
        """
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.item_factors = item_factors.astype(np.float32)
        self.item_content = item_content.astype(np.float32)

        # Popularity fallback for K=0 (same approach as ALS/Semantic)
        n_items = len(item_factors)
        self.popularity = np.zeros(n_items, dtype=np.float32)
        counts = train_df["item_idx"].value_counts()
        for idx, cnt in counts.items():
            self.popularity[int(idx)] = cnt

        pref_dim    = item_factors.shape[1]
        content_dim = item_content.shape[1]

        self.nn = _DropoutNetNN(pref_dim, content_dim,
                                self.hidden_dim, self.output_dim).to(DEVICE)

        interactions = train_df[["user_idx", "item_idx"]].values
        dataset = DropoutNetDataset(interactions, user_factors, item_factors,
                                    item_content, self.pref_dropout)
        loader  = DataLoader(dataset, batch_size=self.batch_size,
                             shuffle=True, num_workers=0)
        optimizer = optim.Adam(self.nn.parameters(), lr=self.lr)

        log.info(f"Training DropoutNet for {self.epochs} epochs on {DEVICE} ...")
        for epoch in range(1, self.epochs + 1):
            self.nn.train()
            total_loss = 0.0
            for batch in loader:
                batch = [t.to(DEVICE) for t in batch]
                u_pref, i_pref_avg, i_pref, i_cont, n_pref, n_cont = batch
                pos_score, neg_score = self.nn(u_pref, i_pref_avg, i_pref, i_cont,
                                               n_pref, n_cont)
                loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if epoch % 5 == 0 or epoch == 1:
                log.info(f"  Epoch {epoch:3d}/{self.epochs}  "
                         f"loss={total_loss/len(loader):.4f}")

        self._precompute_item_repr(proc_dir)
        log.info("DropoutNet training complete.")

    def _precompute_item_repr(self, proc_dir: Path = PROC_DIR):
        """
        Encode all items once and write to a memory-mapped file so the
        full repr matrix (~2 GB for 4.4M items) is never entirely in RAM.
        """
        self.nn.eval()
        n       = len(self.item_factors)
        out_dim = self.output_dim
        repr_path = proc_dir / "dropoutnet_item_repr.npy"
        proc_dir.mkdir(parents=True, exist_ok=True)

        mmap = np.lib.format.open_memmap(
            str(repr_path), mode="w+", dtype=np.float32, shape=(n, out_dim)
        )
        bs = 2048
        with torch.no_grad():
            for start in range(0, n, bs):
                end = min(start + bs, n)
                ip  = torch.tensor(self.item_factors[start:end]).to(DEVICE)
                ic  = torch.tensor(self.item_content[start:end]).to(DEVICE)
                mmap[start:end] = self.nn.encode_item(ip, ic).cpu().numpy()
        mmap.flush()
        self._repr_path = str(repr_path)
        log.info(f"Item repr memmap written: {repr_path}  shape=({n},{out_dim})")

    def _load_item_repr(self) -> np.ndarray:
        """Load item repr as read-only memmap (pages in on demand)."""
        n, d = len(self.item_factors), self.output_dim
        return np.lib.format.open_memmap(
            self._repr_path, mode="r", dtype=np.float32, shape=(n, d)
        )

    # ── Recommendation ─────────────────────────────────────────────────────────

    def recommend(self, seed_item_indices: list[int],
                  seed_ratings: list[float],
                  top_k: int = 10,
                  als_user_vec: np.ndarray | None = None,
                  exclude_seen: bool = True) -> list[int]:
        """
        Return top-K item indices.

        Parameters
        ----------
        seed_item_indices : item_idx of observed interactions (K items)
        seed_ratings      : corresponding rating values
        top_k             : number of items to return
        als_user_vec      : ALS fold-in vector (from fold_in_user); None at K=0
        exclude_seen      : filter out seed items from recommendations
        """
        assert self.nn is not None, "Call train() first."

        # K=0: no history at all — fall back to popularity (same as ALS/Semantic)
        if not seed_item_indices:
            assert self.popularity is not None, "popularity not set — retrain model."
            scores = self.popularity.copy()
            top_idx = np.argpartition(scores, -top_k)[-top_k:]
            top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
            return top_idx.tolist()

        # Build user pref vector
        if als_user_vec is None:
            u_pref = np.zeros(self.item_factors.shape[1], dtype=np.float32)
        else:
            u_pref = als_user_vec.astype(np.float32)

        # Build average seed item pref
        if seed_item_indices:
            i_pref_avg = self.item_factors[seed_item_indices].mean(axis=0)
        else:
            i_pref_avg = np.zeros(self.item_factors.shape[1], dtype=np.float32)

        self.nn.eval()
        with torch.no_grad():
            u  = torch.tensor(u_pref[None]).to(DEVICE)
            ip = torch.tensor(i_pref_avg[None]).to(DEVICE)
            user_repr = self.nn.encode_user(u, ip).cpu().numpy()[0]  # (output_dim,)

        # Score all items via memmap (only accessed rows paged in)
        item_repr = self._load_item_repr()
        n_items   = len(item_repr)
        scores    = np.empty(n_items, dtype=np.float32)
        batch     = 65_536
        for start in range(0, n_items, batch):
            end = min(start + batch, n_items)
            scores[start:end] = item_repr[start:end] @ user_repr

        if exclude_seen and seed_item_indices:
            scores[seed_item_indices] = -np.inf

        top_idx = np.argpartition(scores, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return top_idx.tolist()

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "nn_state":     self.nn.state_dict(),
            "item_factors": self.item_factors,
            "item_content": self.item_content,
            "repr_path":    self._repr_path,
            "popularity":   self.popularity,
            "config": {
                "hidden_dim":   self.hidden_dim,
                "output_dim":   self.output_dim,
                "pref_dropout": self.pref_dropout,
            },
        }, path)
        log.info(f"DropoutNet saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "DropoutNetModel":
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        cfg  = ckpt["config"]
        m    = cls(hidden_dim=cfg["hidden_dim"], output_dim=cfg["output_dim"],
                   pref_dropout=cfg["pref_dropout"])
        m.item_factors = ckpt["item_factors"]
        m.item_content = ckpt["item_content"]
        m._repr_path   = ckpt["repr_path"]
        m.popularity   = ckpt.get("popularity")

        pref_dim    = m.item_factors.shape[1]
        content_dim = m.item_content.shape[1]
        m.nn = _DropoutNetNN(pref_dim, content_dim,
                             cfg["hidden_dim"], cfg["output_dim"]).to(DEVICE)
        m.nn.load_state_dict(ckpt["nn_state"])
        m.nn.eval()
        log.info(f"DropoutNet loaded from {path}")
        return m