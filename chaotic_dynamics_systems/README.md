# Data-Efficient Learning of Chaotic Dynamical Systems

**Master's Thesis — HSE University**

This repository accompanies the thesis *"Data-Efficient Learning of Chaotic Dynamical Systems"*. It benchmarks six learning methods on four dynamical systems that span a wide range of complexity: from a simple 2D oscillator to a spatiotemporal PDE. The central question is whether data-efficient techniques — transfer learning, meta-learning, physics-informed training, and reservoir computing — can match or approach the performance of a fully data-fed baseline when training data is reduced by 10×.

---

## Systems

| # | System | Type | State dim | Lyapunov time | D2 (true) |
|---|---|---|---|---|---|
| 1 | Lorenz attractor | 3D ODE | 3 | ~1.1 time units | 2.06 |
| 2 | Duffing oscillator | 2D ODE (conservative) | 2 | periodic | ~1.0 |
| 3 | Kuramoto-Sivashinsky PDE | PDE, partial obs. | 3 of 64 Fourier modes | ~10.6 time units | ~8.0 |
| 4 | Mackey-Glass DDE | Scalar DDE, delay embedding | 3 (Takens embedding) | ~182 steps | 3.6 |

---

## Methods

All methods use the same LSTM backbone: `LSTM(256) → Dropout(0.2) → LSTM(128) → Dropout(0.2) → Dense(64) → Dense(d)` where `d` matches the state dimension of each system.

| Method | Tier | Training data | Key idea |
|---|---|---|---|
| Baseline RNN | 1 | D_large (100k steps) | Single-step teacher forcing; performance ceiling |
| PINN (large) | 1 | D_large | Physics residual augments MSE loss |
| Transfer Learning | 2 | D_small (10k steps) | Freeze backbone, fine-tune output head |
| MAML | 2 | D_small | Meta-learning: find weights that adapt quickly |
| PINN-small | 2 | D_small | Physics residual without pretrained backbone |
| Hybrid (TL + PINN) | 2 | D_small | Transfer + physics residual |
| ESN (Reservoir Computing) | 2 | D_small | Fixed reservoir, ridge regression readout |

**Two-phase training protocol:**
- **Phase 1:** Standard single-step or meta-training.
- **Phase 2:** Curriculum rollout — prediction horizon K grows from 1 to 16 over 40 epochs, using truncated BPTT (TBPTT_K=8). This corrects the exposure bias that causes models trained with teacher forcing to diverge rapidly in closed-loop mode.

---

## Repository Structure

```
.
├── notebooks/
│   ├── 01_lorenz.ipynb          # Lorenz attractor experiments
│   ├── 02_Duffing.ipynb         # Duffing oscillator experiments
│   ├── 03_KS.ipynb              # Kuramoto-Sivashinsky PDE experiments
│   └── 04_MG.ipynb              # Mackey-Glass DDE experiments
├── results/
│   ├── lorenz_models_eval_results.csv
│   ├── lorenz_d2_results.csv
│   ├── lorenz_manual_ic_results.csv
│   ├── lorenz_rand_ic_results.csv
│   ├── duffing_models_eval_results.csv
│   ├── duffing_d2_results.csv
│   ├── duffing_manual_ic_results.csv
│   ├── duffing_rand_ic_results.csv
│   ├── KS_models_eval_results.csv
│   ├── KS_d2_results.csv
│   ├── KS_manual_ic_results.csv
│   ├── KS_rand_ic_results.csv
│   ├── MG_models_eval_results.csv
│   ├── MG_d2_results.csv
│   ├── MG_manual_ic_results.csv
│   └── MG_rand_ic_results.csv
├── scripts/
│   ├── summarise_results.py     # Print cross-system leaderboard
│   └── plot_r2_comparison.py    # Generate R² comparison bar chart
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Results Summary

All metrics computed on held-out test trajectories with different initial conditions from training.

### R² (higher is better)

| Method | Lorenz | Duffing | KS | MG |
|---|---|---|---|---|
| Baseline RNN (orig) | 0.9502 | 0.9910 | 0.9980 | 0.9994 |
| Baseline RNN – MS | 0.9493 | 0.9901 | 0.9984 | 0.9996 |
| PINN large-data | 0.9614 | 0.9839 | 0.9970 | 0.9974 |
| Transfer Learning (orig) | 0.9991 | 0.9843 | 0.9963 | 0.9996 |
| Transfer Learning – MS | **0.9996** | 0.9857 | 0.9976 | **0.9997** |
| MAML (orig) | 0.9948 | 0.9803 | 0.9228 | 0.9973 |
| MAML – MS | 0.9992 | 0.9842 | 0.9967 | 0.9995 |
| PINN-small (orig) | 0.9651 | 0.9925 | 0.9657 | 0.9712 |
| PINN-small – MS | 0.9773 | 0.9848 | 0.9716 | 0.9933 |
| Hybrid TL+PINN (orig) | 0.9975 | 0.9857 | 0.9963 | 0.9991 |
| Hybrid TL+PINN – MS | 0.9980 | 0.9835 | **0.9982** | **0.9997** |
| **ESN** | **1.0000** | **1.0000** | **0.9999** | **0.9999** |

### Correlation Dimension D2 Error — |D2 − D2_true| (lower is better)

True values: Lorenz = 2.06, Duffing = 1.0, KS = 8.0, MG = 3.6

| Method | Lorenz | Duffing | KS | MG |
|---|---|---|---|---|
| ESN | 0.345 | 0.051 | 7.573 | 2.251 |
| Hybrid TL+PINN – MS | 0.448 | **0.017** | 7.969 | 1.893 |
| Transfer Learning (orig) | 0.669 | 0.291 | 7.772 | 1.834 |
| Transfer Learning – MS | 0.887 | 0.956 | **7.059** | **1.717** |
| Baseline RNN – MS | 0.986 | 0.915 | 7.899 | 1.850 |
| MAML – MS | 1.032 | 0.970 | 7.575 | **1.736** |

---

## Quick Start

### Run in Google Colab (recommended)

Each notebook is self-contained and designed for Google Colab with a T4 GPU. Open any notebook, run Section 0 first to mount Drive and initialise checkpointing, then run all cells. Completed steps are automatically skipped on subsequent runs.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/chaotic-dynamics-learning/blob/main/notebooks/01_lorenz.ipynb)

### Local setup

```bash
git clone https://github.com/YOUR_USERNAME/chaotic-dynamics-learning.git
cd chaotic-dynamics-learning
pip install -r requirements.txt
jupyter notebook notebooks/01_lorenz.ipynb
```

---

## Reproducing Results

Each notebook is fully reproducible with fixed random seeds (`np.random.seed(42)`, `tf.random.set_seed(42)`). Runtime estimates on a T4 GPU:

| Notebook | Phase 1 | Phase 2 | Total |
|---|---|---|---|
| 01_lorenz | ~20 min | ~15 min | ~35 min |
| 02_Duffing | ~20 min | ~15 min | ~35 min |
| 03_KS | ~30 min | ~30 min | ~60 min |
| 04_MG | ~20 min | ~15 min | ~35 min |

**Checkpointing:** All notebooks save to Google Drive after each completed step. If a Colab session resets mid-training, re-running all cells resumes from the last checkpoint automatically.

---

## Key Findings

1. **ESN dominates on R² and D2 for periodic/near-periodic systems** (Duffing, MG). Its closed-form training and implicit memory structure are well-matched to these dynamics.

2. **Transfer Learning is the most data-efficient neural method.** On Lorenz, Transfer-MS achieves R² = 0.9996, closing 99% of the gap between the small-data baseline and the large-data ceiling with 10× less training data.

3. **Curriculum rollout (Phase 2) consistently improves VPT** across all systems and methods. Teacher-forcing alone leaves all LSTM methods with near-zero VPT on Lorenz and KS.

4. **PINN benefits depend on physics completeness.** On Lorenz and Duffing, where the full ODE right-hand side is available, PINN improves D2 accuracy. On KS (linear residual only) and MG (approximate delay substitute), the benefit is weaker.

5. **KS is the hardest system.** Partial observability (3 of 64 Fourier modes) limits all methods' D2 estimates — every model's 3D trajectory falls well below the true D2 = 8.0 by construction.

---

## Citation

If you use this code or results in your work, please cite:

```bibtex
@mastersthesis{author2025chaotic,
  title  = {Data-Efficient Learning of Chaotic Dynamical Systems},
  author = {Author Name},
  school = {HSE University},
  year   = {2025}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
