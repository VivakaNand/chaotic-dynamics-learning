# Scripts

Standalone Python scripts for reproducing key tables and figures from the thesis results.
All scripts read from `../results/` and write output back to `../results/`.

## Requirements

```bash
pip install pandas numpy matplotlib scipy scikit-learn tabulate
```

No GPU needed — these scripts only read CSV files and generate plots.

## Usage

Run from the repository root:

```bash
# Cross-system leaderboard (R² by default)
python scripts/summarise_results.py

# Leaderboard for a different metric
python scripts/summarise_results.py --metric RMSE
python scripts/summarise_results.py --metric MSE

# 4-panel R² bar chart across all systems
python scripts/plot_r2_comparison.py

# Same for RMSE
python scripts/plot_r2_comparison.py --metric RMSE

# D2 error heatmap
python scripts/plot_d2_heatmap.py
```

## Output files

| Script | Output |
|---|---|
| `summarise_results.py` | Terminal table only |
| `plot_r2_comparison.py` | `results/r2_comparison.pdf`, `results/r2_comparison.png` |
| `plot_d2_heatmap.py` | `results/d2_heatmap.pdf`, `results/d2_heatmap.png` |
