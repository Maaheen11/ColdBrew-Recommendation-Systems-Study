"""
embeddings.py
-------------
Build Sentence-BERT item embeddings using the cleaned text from
friend's DataProcesser._embedding_data_clean_() output.

The items.parquet saved by preprocess.py has a "text" column
(renamed from friend's "embedding_text") which is title + cleaned description.
We encode that with Sentence-BERT and write to a memory-mapped file.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PROC_DIR   = Path("data/processed")
MODEL_NAME = "all-MiniLM-L6-v2"   # same model as friend's SementicEmbedding.py
BATCH_SIZE = 256


def _get_model(model_name: str = MODEL_NAME):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("Run: pip install sentence-transformers")
    log.info(f"Loading SentenceTransformer '{model_name}' ...")
    return SentenceTransformer(model_name)


def build_item_embeddings(proc_dir: Path = PROC_DIR,
                          model_name: str = MODEL_NAME,
                          batch_size: int = BATCH_SIZE,
                          force: bool = False) -> np.ndarray:
    """
    Encode all items using Sentence-BERT and save as a memory-mapped file.

    Uses the "text" column from items.parquet, which comes from friend's
    embedding_data_clean() — already cleaned title + description.

    Returns
    -------
    np.memmap of shape (n_items, 384) — rows aligned to item_idx
    """
    out_path  = proc_dir / "item_embeddings.npy"
    meta_path = proc_dir / "item_embeddings_meta.pkl"

    if out_path.exists() and meta_path.exists() and not force:
        log.info(f"Embeddings already built at {out_path}, loading ...")
        return load_item_embeddings(proc_dir)

    items   = pd.read_parquet(proc_dir / "items.parquet")
    n_items = int(items["item_idx"].max()) + 1

    # Build text list aligned to item_idx
    # Uses "text" column (= friend's "embedding_text": cleaned title + description)
    texts = [""] * n_items
    for _, row in items.iterrows():
        idx  = int(row["item_idx"])
        text = str(row.get("text") or "").strip()
        texts[idx] = text

    # Probe embedding dimension
    model   = _get_model(model_name)
    probe   = model.encode(["probe"], convert_to_numpy=True)
    emb_dim = probe.shape[1]
    log.info(f"  n_items={n_items:,}  emb_dim={emb_dim}  "
             f"total size ≈ {n_items * emb_dim * 4 / 1e9:.2f} GB")

    # Write directly to memmap — never holds full matrix in RAM
    mmap = np.lib.format.open_memmap(
        str(out_path), mode="w+", dtype=np.float32, shape=(n_items, emb_dim)
    )

    log.info(f"Encoding {n_items:,} items ...")
    for start in range(0, n_items, batch_size):
        end  = min(start + batch_size, n_items)
        vecs = model.encode(
            texts[start:end],
            convert_to_numpy=True,
            normalize_embeddings=True,   # cosine sim = dot product
            show_progress_bar=False,
        )
        mmap[start:end] = vecs
        if start % (batch_size * 20) == 0:
            log.info(f"  {end:,}/{n_items:,} encoded ...")

    mmap.flush()

    import pickle
    with open(meta_path, "wb") as f:
        pickle.dump({"n_items": n_items, "emb_dim": emb_dim}, f)

    log.info(f"Embeddings saved to {out_path}")
    return mmap


def load_item_embeddings(proc_dir: Path = PROC_DIR) -> np.ndarray:
    """
    Load item embeddings as a read-only memory-map.
    Only pages accessed are loaded into RAM — safe for large datasets.
    """
    path      = proc_dir / "item_embeddings.npy"
    meta_path = proc_dir / "item_embeddings_meta.pkl"

    if not path.exists():
        raise FileNotFoundError(
            f"Item embeddings not found at {path}. "
            "Run build_item_embeddings() first."
        )

    import pickle
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    return np.lib.format.open_memmap(
        str(path), mode="r", dtype=np.float32,
        shape=(meta["n_items"], meta["emb_dim"])
    )


def build_user_profile(seed_item_indices: list[int],
                       seed_ratings: list[float],
                       item_embeddings: np.ndarray) -> np.ndarray:
    """
    Build a rating-weighted average embedding as the user profile.
    Same approach as friend's user_history() but for cold-start seeds.

    Returns zero vector when seed is empty (K=0).
    """
    if not seed_item_indices:
        return np.zeros(item_embeddings.shape[1], dtype=np.float32)

    vecs    = item_embeddings[seed_item_indices]
    weights = np.array(seed_ratings, dtype=np.float32)
    profile = (vecs * weights[:, None]).sum(axis=0)

    norm = np.linalg.norm(profile)
    if norm > 0:
        profile /= norm
    return profile.astype(np.float32)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    build_item_embeddings(force=False)
