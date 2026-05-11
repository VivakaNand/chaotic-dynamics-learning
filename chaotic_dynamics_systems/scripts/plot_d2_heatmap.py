"""
plot_d2_heatmap.py
==================
Generates a heatmap of D2 errors across all methods and systems.
Saves to results/d2_heatmap.pdf and results/d2_heatmap.png.

Usage:
    python scripts/plot_d2_heatmap.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

D2_FILES = {
    "Lorenz":  ("lorenz_d2_results.csv",  2.06),
    "Duffing": ("duffing_d2_results.csv", 1.0),
    "KS":      ("KS_d2_results.csv",      8.0),
    "MG":      ("MG_d2_results.csv",      3.6),
}

METHOD_ORDER = [
    "Baseline RNN (orig)",
    "Baseline RNN - MS",
    "PINN large-data (orig)",
    "Transfer Learning (orig)",
    "Transfer Learning - MS",
    "MAML (orig)",
    "MAML - MS",
    "PINN-small (orig)",
    "PINN-small - MS",
    "Hybrid Transfer+PINN (orig)",
    "Hybrid Transfer+PINN - MS",
    "ESN",
]

SHORT = {
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


def main():
    frames = {}
    for system, (fname, d2_true) in D2_FILES.items():
        df = pd.read_csv(os.path.join(RESULTS_DIR, fname))[["Model", "D2_err"]]
        df = df.set_index("Model")["D2_err"]
        frames[system] = df

    matrix = pd.DataFrame(frames)
    # Reorder rows
    matrix = matrix.reindex([m for m in METHOD_ORDER if m in matrix.index])
    matrix.index = [SHORT.get(m, m) for m in matrix.index]

    fig, ax = plt.subplots(figsize=(8, 6))
    data = matrix.values.astype(float)

    # Normalise per column so colours are comparable within each system
    col_max = np.nanmax(data, axis=0, keepdims=True)
    data_norm = data / (col_max + 1e-12)

    im = ax.imshow(data_norm, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)

    # Annotate each cell with raw D2_err
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = matrix.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=7.5, color="black")

    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(
        [f"{s}\n(D2={D2_FILES[s][1]})" for s in matrix.columns],
        fontsize=9
    )
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=8)
    ax.set_title(
        "D2 Error  |D2 − D2_true|  (lower = better attractor reconstruction)\n"
        "Color normalised within each system column",
        fontsize=10, pad=10
    )

    plt.colorbar(im, ax=ax, label="Normalised D2 error (0=best, 1=worst)")
    plt.tight_layout()

    out_pdf = os.path.join(RESULTS_DIR, "d2_heatmap.pdf")
    out_png = os.path.join(RESULTS_DIR, "d2_heatmap.png")
    plt.savefig(out_pdf, bbox_inches="tight", dpi=150)
    plt.savefig(out_png, bbox_inches="tight", dpi=150)
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
