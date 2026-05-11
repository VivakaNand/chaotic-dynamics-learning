"""
summarise_results.py
====================
Prints a cross-system leaderboard table from the four eval CSVs.

Usage:
    python scripts/summarise_results.py
    python scripts/summarise_results.py --metric R2
    python scripts/summarise_results.py --metric RMSE --sort asc
"""

import argparse
import os
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

SYSTEMS = {
    "Lorenz":  "lorenz_models_eval_results.csv",
    "Duffing": "duffing_models_eval_results.csv",
    "KS":      "KS_models_eval_results.csv",
    "MG":      "MG_models_eval_results.csv",
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


def load_all(metric: str) -> pd.DataFrame:
    frames = {}
    for system, fname in SYSTEMS.items():
        path = os.path.join(RESULTS_DIR, fname)
        df = pd.read_csv(path)[["Method", metric]].set_index("Method")
        frames[system] = df[metric]
    combined = pd.DataFrame(frames)
    # Reorder rows to follow METHOD_ORDER
    combined = combined.reindex([m for m in METHOD_ORDER if m in combined.index])
    return combined


def print_table(df: pd.DataFrame, metric: str, ascending: bool) -> None:
    print(f"\n{'=' * 65}")
    print(f"  Cross-System Leaderboard — {metric}")
    print(f"{'=' * 65}")

    # Format values
    fmt = "{:.6f}" if metric in ("MSE", "RMSE") else "{:.4f}"
    _map = df.applymap if hasattr(df, "applymap") else df.map
    display = _map(lambda x: fmt.format(x) if pd.notna(x) else "—")

    # Highlight best per system
    for col in df.columns:
        if ascending:
            best_idx = df[col].idxmin()
        else:
            best_idx = df[col].idxmax()
        display.loc[best_idx, col] = f"**{display.loc[best_idx, col]}**"

    print(display.to_string())

    print(f"\nBest per system ({'lower' if ascending else 'higher'} is better):")
    for col in df.columns:
        if ascending:
            best = df[col].idxmin()
        else:
            best = df[col].idxmax()
        val = df.loc[best, col]
        print(f"  {col:10s}: {best:35s}  {metric}={val:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Cross-system results summary")
    parser.add_argument(
        "--metric",
        choices=["R2", "RMSE", "MSE"],
        default="R2",
        help="Metric to display (default: R2)",
    )
    parser.add_argument(
        "--sort",
        choices=["asc", "desc"],
        default=None,
        help="Override sort direction (default: desc for R2, asc for MSE/RMSE)",
    )
    args = parser.parse_args()

    ascending = args.metric in ("MSE", "RMSE") if args.sort is None else (args.sort == "asc")

    df = load_all(args.metric)
    print_table(df, args.metric, ascending)

    # Also print D2 errors
    print(f"\n{'=' * 65}")
    print("  Attractor Reconstruction — D2 Error |D2 − D2_true| (lower is better)")
    print(f"  True D2: Lorenz=2.06, Duffing=1.0, KS=8.0, MG=3.6")
    print(f"{'=' * 65}")

    d2_frames = {}
    d2_files = {
        "Lorenz":  "lorenz_d2_results.csv",
        "Duffing": "duffing_d2_results.csv",
        "KS":      "KS_d2_results.csv",
        "MG":      "MG_d2_results.csv",
    }
    for system, fname in d2_files.items():
        path = os.path.join(RESULTS_DIR, fname)
        df_d2 = pd.read_csv(path)[["Model", "D2_err"]].set_index("Model")
        d2_frames[system] = df_d2["D2_err"]

    d2_combined = pd.DataFrame(d2_frames)
    d2_combined = d2_combined.reindex([m for m in METHOD_ORDER if m in d2_combined.index])
    print(d2_combined.round(3).to_string())


if __name__ == "__main__":
    main()
