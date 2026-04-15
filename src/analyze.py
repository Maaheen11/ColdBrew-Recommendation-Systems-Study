"""
analyze.py
----------
Load results/metrics.csv and produce:
  - results/plots/recall_curve.png      (degradation curves)
  - results/plots/ndcg_curve.png
  - results/plots/crossover_recall.png  (bar chart + gain vs K=0)
  - results/plots/crossover_ndcg.png
  - results/crossover_points.csv        (K levels where rankings switch)
  - console summary table
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

log = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
PLOTS_DIR   = RESULTS_DIR / "plots"

MODEL_COLORS = {
    "ALS":        "#1f77b4",
    "Semantic":   "#2ca02c",
    "DropoutNet": "#d62728",
}
MODEL_MARKERS = {
    "ALS":        "o",
    "Semantic":   "s",
    "DropoutNet": "^",
}


# ── Plot helpers ───────────────────────────────────────────────────────────────

def _style_ax(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_metric_curve(df: pd.DataFrame, metric: str,
                      out_path: Path, top_k: int = 10):
    """Degradation curve: one line per model across K% levels."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for model in df["model"].unique():
        sub = df[df["model"] == model].sort_values("k_pct")
        ax.plot(sub["k_pct"], sub[metric],
                label=model,
                color=MODEL_COLORS.get(model),
                marker=MODEL_MARKERS.get(model, "x"),
                linewidth=2, markersize=7)

    ax.set_xticks(sorted(df["k_pct"].unique()))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%g%%"))
    _style_ax(ax,
              xlabel="K% — user history available (0% = fully cold, 90% = nearly warm)",
              ylabel=f"{metric.upper()}@{top_k}",
              title=f"Cold-start to Warm Transition — {metric.upper()}@{top_k}")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info(f"Saved {out_path}")


def plot_crossover_analysis(df: pd.DataFrame, metric: str,
                             out_path: Path, top_k: int = 10):
    """Bar chart of absolute scores + line chart of gain vs K=0."""
    k_levels = sorted(df["k_pct"].unique())
    models   = list(df["model"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: absolute score per K level
    ax    = axes[0]
    x     = np.arange(len(k_levels))
    width = 0.25
    for i, model in enumerate(models):
        sub    = df[df["model"] == model].set_index("k_pct")
        values = [sub.loc[k, metric] if k in sub.index else 0.0 for k in k_levels]
        ax.bar(x + i * width, values, width,
               label=model, color=MODEL_COLORS.get(model), alpha=0.85)
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"{k}%" for k in k_levels])
    _style_ax(ax,
              xlabel="K% history available (cold → warm)",
              ylabel=f"{metric.upper()}@{top_k}",
              title=f"{metric.upper()}@{top_k} by Cold-Start Level")

    # Right: gain relative to K=0 baseline
    ax2 = axes[1]
    for model in models:
        sub  = df[df["model"] == model].set_index("k_pct").sort_index()
        base = sub.loc[0, metric] if 0 in sub.index else 0.0
        gains = [(sub.loc[k, metric] - base) if k in sub.index else 0.0
                 for k in k_levels]
        ax2.plot(k_levels, gains,
                 label=model,
                 color=MODEL_COLORS.get(model),
                 marker=MODEL_MARKERS.get(model, "x"),
                 linewidth=2, markersize=7)
    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_xticks(k_levels)
    ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter("%g%%"))
    _style_ax(ax2,
              xlabel="K% history available (cold → warm)",
              ylabel=f"Δ {metric.upper()}@{top_k} (gain over K=0)",
              title="Relative Gain vs Fully-Cold Baseline")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info(f"Saved {out_path}")


# ── Crossover detection ────────────────────────────────────────────────────────

def find_crossover_points(df: pd.DataFrame, metric: str = "recall") -> pd.DataFrame:
    """
    For each pair of models, find the first K level where their ranking flips.
    This directly addresses the paper's research question:
    "at which K does each model's advantage begin or end?"
    """
    models   = sorted(df["model"].unique())
    k_levels = sorted(df["k_pct"].unique())
    rows     = []

    for i, ma in enumerate(models):
        for mb in models[i + 1:]:
            sub_a = df[df["model"] == ma].set_index("k_pct")[metric]
            sub_b = df[df["model"] == mb].set_index("k_pct")[metric]

            prev_winner = None
            for k in k_levels:
                a_val = sub_a.get(k, np.nan)
                b_val = sub_b.get(k, np.nan)
                if np.isnan(a_val) or np.isnan(b_val):
                    continue
                winner = ma if a_val >= b_val else mb
                if prev_winner is not None and winner != prev_winner:
                    rows.append({
                        "model_a":                ma,
                        "model_b":                mb,
                        "crossover_k":            k,
                        "metric":                 metric,
                        "winner_after_crossover":  winner,
                    })
                prev_winner = winner

    return pd.DataFrame(rows)


# ── Summary table ──────────────────────────────────────────────────────────────

def print_summary_table(df: pd.DataFrame, top_k: int = 10):
    for metric in ["recall", "ndcg"]:
        pivot = df.pivot(index="k_pct", columns="model", values=metric).round(4)
        print(f"\n{'─' * 55}")
        print(f"  {metric.upper()}@{top_k} across cold-start levels")
        print(f"{'─' * 55}")
        print(pivot.to_string())
    print()


# ── Main ───────────────────────────────────────────────────────────────────────

def run_analysis(results_dir: Path = RESULTS_DIR, top_k: int = 10):
    metrics_path = results_dir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"metrics.csv not found at {metrics_path}. "
            "Run evaluation first (python main.py --skip-preprocess --skip-train)."
        )

    df = pd.read_csv(metrics_path)
    df = df[df["n_users"] > 0]   # drop K=100 rows (no ground truth, all zeros)
    log.info(f"Loaded {len(df)} rows from {metrics_path}")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    plot_metric_curve(df, "recall", PLOTS_DIR / "recall_curve.png", top_k=top_k)
    plot_metric_curve(df, "ndcg",   PLOTS_DIR / "ndcg_curve.png",   top_k=top_k)
    plot_crossover_analysis(df, "recall", PLOTS_DIR / "crossover_recall.png", top_k=top_k)
    plot_crossover_analysis(df, "ndcg",   PLOTS_DIR / "crossover_ndcg.png",   top_k=top_k)

    co_all = pd.concat([
        find_crossover_points(df, metric="recall"),
        find_crossover_points(df, metric="ndcg"),
    ], ignore_index=True)
    co_path = results_dir / "crossover_points.csv"
    co_all.to_csv(co_path, index=False)
    log.info(f"Crossover points saved to {co_path}")

    print_summary_table(df, top_k=top_k)
    if not co_all.empty:
        print("Crossover points detected:")
        print(co_all.to_string(index=False))
    else:
        print("No crossover points detected across the evaluated K levels.")

    return df, co_all


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    run_analysis()