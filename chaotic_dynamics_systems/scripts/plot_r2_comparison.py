"""
plot_r2_comparison.py
=====================
Generates a 4-panel bar chart comparing R² across all methods and systems.
Saves to results/r2_comparison.pdf and results/r2_comparison.png.

Usage:
    python scripts/plot_r2_comparison.py
    python scripts/plot_r2_comparison.py --metric RMSE
    python scripts/plot_r2_comparison.py --out my_figure.pdf
"""

import argparse
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUT_DIR     = os.path.join(os.path.dirname(__file__), "..", "results")

SYSTEMS = {
    "Lorenz":  "lorenz_models_eval_results.csv",
    "Duffing": "duffing_models_eval_results.csv",
    "KS":      "KS_models_eval_results.csv",
    "MG":      "MG_models_eval_results.csv",
}

# Shortened labels for the bar chart x-axis
SHORT_LABELS = {
    "Baseline RNN (orig)":          "RNN",
    "Baseline RNN - MS":            "RNN-MS",
    "PINN large-data (orig)":       "PINN-L",
    "Transfer Learning (orig)":     "TL",
    "Transfer Learning - MS":       "TL-MS",
    "MAML (orig)":                  "MAML",
    "MAML - MS":                    "MAML-MS",
    "PINN-small (orig)":            "PINN-S",
    "PINN-small - MS":              "PINN-S-MS",
    "Hybrid Transfer+PINN (orig)":  "Hybrid",
    "Hybrid Transfer+PINN - MS":    "Hybrid-MS",
    "ESN":                          "ESN",
}

TIER_COLORS = {
    "Tier-1": "#2166ac",  # blue
    "Tier-2": "#d6604d",  # red-orange
}

def load_system(fname: str, metric: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(RESULTS_DIR, fname))[["Method", "Tier", metric]]


def make_figure(metric: str, out_path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=False)
    axes = axes.flatten()

    ascending = metric in ("MSE", "RMSE")

    for ax, (system, fname) in zip(axes, SYSTEMS.items()):
        df = load_system(fname, metric)
        methods = df["Method"].tolist()
        values  = df[metric].tolist()
        tiers   = df["Tier"].tolist()

        labels = [SHORT_LABELS.get(m, m) for m in methods]
        colors = [TIER_COLORS[t] for t in tiers]
        x = np.arange(len(methods))

        bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.5, width=0.7)

        # Highlight the best bar
        best_idx = int(np.argmin(values) if ascending else np.argmax(values))
        bars[best_idx].set_edgecolor("black")
        bars[best_idx].set_linewidth(2.0)

        ax.set_title(system, fontsize=13, fontweight="bold", pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5)
        ax.set_ylabel(metric, fontsize=10)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Value annotations on top of each bar
        for bar, val in zip(bars, values):
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h * 1.005 if not ascending else h + max(values) * 0.005,
                f"{val:.4f}",
                ha="center", va="bottom", fontsize=5.5, rotation=90,
            )

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=TIER_COLORS["Tier-1"], label="Tier-1 (D_large, 100k steps)"),
        Patch(facecolor=TIER_COLORS["Tier-2"], label="Tier-2 (D_small, 10k steps)"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        ncol=2,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
    )

    direction = "lower" if ascending else "higher"
    fig.suptitle(
        f"{metric} by Method and System ({direction} is better)\n"
        "Black border = best per system",
        fontsize=11, y=1.04,
    )

    plt.tight_layout()
    plt.savefig(out_path.replace(".pdf", ".pdf"), bbox_inches="tight", dpi=150)
    plt.savefig(out_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    print(f"Saved: {out_path.replace('.pdf', '.pdf')}")
    print(f"Saved: {out_path.replace('.pdf', '.png')}")


def main():
    parser = argparse.ArgumentParser(description="R² / RMSE comparison bar chart")
    parser.add_argument("--metric", choices=["R2", "RMSE", "MSE"], default="R2")
    parser.add_argument(
        "--out",
        default=None,
        help="Output path (default: results/<metric>_comparison.pdf)",
    )
    args = parser.parse_args()

    out = args.out or os.path.join(OUT_DIR, f"{args.metric.lower()}_comparison.pdf")
    make_figure(args.metric, out)


if __name__ == "__main__":
    main()
