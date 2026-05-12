# Data-Efficient Learning of Chaotic Dynamical Systems

**Master's Thesis — HSE University**

This repository contains all experimental code, results, and figures for the thesis *"Data-Efficient Learning of Chaotic Dynamical Systems"*. Six learning methods are benchmarked on four dynamical systems that range from a 2D conservative oscillator to a spatiotemporal PDE. The central question is whether data-efficient techniques — transfer learning, meta-learning, physics-informed training, and reservoir computing — can approach the performance of a fully data-fed baseline when training data is reduced by a factor of ten.

---

## Notebooks

All experiments run on **Google Colab (T4 GPU)**. Each notebook is self-contained: it generates data, trains all models, evaluates every metric, and exports all figures. Click a badge to open directly in Colab.

| # | System | Type | Notebook | Colab |
|---|---|---|---|---|
| 1 | Lorenz attractor | 3D ODE | [`01_lorenz.ipynb`](chaotic_dynamics_systems/01_lorenz.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VivakaNand/chaotic-dynamics-learning/blob/main/chaotic_dynamics_systems/01_lorenz.ipynb) |
| 2 | Duffing oscillator | 2D ODE | [`02_Duffing.ipynb`](chaotic_dynamics_systems/02_Duffing.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VivakaNand/chaotic-dynamics-learning/blob/main/chaotic_dynamics_systems/02_Duffing.ipynb) |
| 3 | Kuramoto-Sivashinsky PDE | PDE, partial obs. | [`03_KS.ipynb`](chaotic_dynamics_systems/03_KS.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VivakaNand/chaotic-dynamics-learning/blob/main/chaotic_dynamics_systems/03_KS.ipynb) |
| 4 | Mackey-Glass DDE | Scalar DDE | [`04_MG.ipynb`](chaotic_dynamics_systems/04_MG.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VivakaNand/chaotic-dynamics-learning/blob/main/chaotic_dynamics_systems/04_MG.ipynb) |

---

## Methods

All neural-network methods share the same LSTM backbone. Performance differences reflect training strategy and data volume, not model capacity.

| Method | Tier | Data | Description |
|---|---|---|---|
| **Baseline RNN** | 1 | D_large (100k steps) | Single-step teacher forcing. Performance ceiling. |
| **PINN (large)** | 1 | D_large | MSE loss augmented with a physics residual from the known governing equation. |
| **Transfer Learning** | 2 | D_small (10k steps) | Backbone pretrained on D_large, output layers fine-tuned on D_small. |
| **MAML** | 2 | D_small | Meta-learns an initialisation that adapts in one gradient step to any trajectory segment. |
| **PINN-small** | 2 | D_small | Physics residual on small data, no pretrained backbone. |
| **Hybrid (TL + PINN)** | 2 | D_small | Transfer learning and physics residual applied together. |
| **ESN** | 2 | D_small | Fixed random reservoir; linear readout trained by ridge regression only. |

**Shared backbone:** `LSTM(256) → Dropout(0.2) → LSTM(128) → Dropout(0.2) → Dense(64, relu) → Dense(d)`

**Two-phase training:**
- **Phase 1** — Single-step teacher forcing (or meta-training for MAML).
- **Phase 2** — Curriculum rollout: prediction horizon K grows 1 → 2 → 4 → 8 → 16 over 40 epochs with TBPTT (K=8). This corrects the exposure bias that makes teacher-forced models diverge in closed-loop autonomous mode.

---

## Repository Structure

```
chaotic-dynamics-learning/
└── chaotic_dynamics_systems/
    │
    ├── 01_lorenz.ipynb                    ← Lorenz attractor experiments
    ├── 02_Duffing.ipynb                   ← Duffing oscillator experiments
    ├── 03_KS.ipynb                        ← Kuramoto-Sivashinsky PDE experiments
    ├── 04_MG.ipynb                        ← Mackey-Glass DDE experiments
    │
    ├── 4 model evaluation CSVs/           ← MSE, RMSE, R² for all 12 variants × 4 systems
    ├── Correlation dimension results 4 files/  ← D2 estimates and D2_err per model × system
    ├── IC robustness results — 8 files/   ← Per-IC metrics (11 manual + 25 random ICs × 4 systems)
    ├── VPT results 4 files/               ← Valid Prediction Time per model × system
    │
    ├── VPT + D2 Figures/                  ← Summary VPT and D2 bar charts (all systems)
    ├── model comparison figures/          ← Composite leaderboards and radar charts (all systems)
    │
    ├── figures_Lorenz_dynamics/           ← Lorenz attractor trajectory and phase-space plots
    ├── figures_Lorenz_manual/             ← Lorenz per-IC prediction plots (11 manual ICs)
    ├── figures_Lorenz_model/              ← Lorenz R², VPT, and leaderboard bar charts
    ├── figures_Lorenz_rand/               ← Lorenz per-IC prediction plots (25 random ICs)
    ├── figures_lorenz/                    ← Lorenz sensitivity sweep figures (λ, spectral radius)
    │
    ├── figures_Duffing_dynamics/          ← Duffing phase-portrait and trajectory plots
    ├── figures_Duffing_manual/            ← Duffing per-IC prediction plots (11 manual ICs)
    ├── figures_Duffing_model/             ← Duffing R², VPT, and leaderboard bar charts
    ├── figures_Duffing_rand/              ← Duffing per-IC prediction plots (25 random ICs)
    │
    ├── figures_KS_dynamics/               ← KS Fourier-mode trajectory and attractor plots
    ├── figures_KS_manual/                 ← KS per-IC prediction plots (11 manual ICs)
    ├── figures_KS_model/                  ← KS R², VPT, and leaderboard bar charts
    ├── figures_KS_rand/                   ← KS per-IC prediction plots (25 random ICs)
    │
    ├── figures_MG_dynamics/               ← MG delay-embedding trajectory and attractor plots
    ├── figures_MG_manual/                 ← MG per-IC prediction plots (11 manual ICs)
    ├── figures_MG_model/                  ← MG R², VPT, and leaderboard bar charts
    ├── figures_MG_rand/                   ← MG per-IC prediction plots (25 random ICs)
    │
    ├── LICENSE
    ├── README.md
    └── requirements.txt
```

---

## Results

All metrics are evaluated on held-out test trajectories with initial conditions different from training.

### R² — higher is better

| Method | Lorenz | Duffing | KS | MG |
|---|---|---|---|---|
| Baseline RNN | 0.9502 | 0.9910 | 0.9980 | 0.9994 |
| Baseline RNN – MS | 0.9493 | 0.9901 | 0.9984 | 0.9996 |
| PINN large-data | 0.9614 | 0.9839 | 0.9970 | 0.9974 |
| Transfer Learning | 0.9991 | 0.9843 | 0.9963 | 0.9996 |
| Transfer Learning – MS | **0.9996** | 0.9857 | 0.9976 | **0.9997** |
| MAML | 0.9948 | 0.9803 | 0.9228 | 0.9973 |
| MAML – MS | 0.9992 | 0.9842 | 0.9967 | 0.9995 |
| PINN-small | 0.9651 | **0.9925** | 0.9657 | 0.9712 |
| PINN-small – MS | 0.9773 | 0.9848 | 0.9716 | 0.9933 |
| Hybrid TL+PINN | 0.9975 | 0.9857 | 0.9963 | 0.9991 |
| Hybrid TL+PINN – MS | 0.9980 | 0.9835 | **0.9982** | **0.9997** |
| **ESN** | **1.0000** | **1.0000** | **0.9999** | **0.9999** |

### D2 Error — `|D2 − D2_true|`, lower is better

True values: Lorenz = 2.06 · Duffing = 1.0 · KS = 8.0 · MG = 3.6

| Method | Lorenz | Duffing | KS | MG |
|---|---|---|---|---|
| ESN | 0.345 | 0.051 | 7.573 | 2.251 |
| Hybrid TL+PINN – MS | 0.448 | **0.017** | 7.969 | 1.893 |
| Transfer Learning | 0.669 | 0.291 | 7.772 | 1.834 |
| Transfer Learning – MS | 0.887 | 0.956 | **7.059** | **1.717** |
| MAML – MS | 1.032 | 0.970 | 7.575 | 1.736 |
| Baseline RNN – MS | 0.986 | 0.915 | 7.899 | 1.850 |

> **KS note:** The true KS attractor dimension is ~8.0 for the full 64-mode system. All models operate on a 3-dimensional Fourier observable, so every D2 estimate is structurally bounded below 3. The error column reflects this embedding constraint, not a method failure.

---

## Key Findings

**ESN ranks first on R² across all four systems.** Its closed-form ridge regression training and implicit memory structure are well-matched to both periodic and chaotic dynamics.

**Transfer Learning – MS is the most data-efficient neural method.** On Lorenz, it achieves R² = 0.9996 with 10× less training data than the Tier-1 baseline, closing nearly the full performance gap.

**Phase 2 curriculum rollout is essential for autonomous prediction.** Without it, all LSTM methods achieved near-zero Valid Prediction Time on Lorenz and KS despite high one-step R², confirming that teacher-forcing alone does not prepare models for closed-loop operation.

**PINN benefits depend on physics completeness.** On Lorenz and Duffing the full ODE right-hand side is available as a residual, and PINN methods improve D2 accuracy. On KS (linear residual only, nonlinear term excluded due to partial observability) and MG (delay term unavailable in the Takens embedding), the benefit is weaker.

---

## How to Run

### Google Colab — recommended

1. Click any badge in the Notebooks table above.
2. In Colab: `Runtime → Change runtime type → T4 GPU`.
3. Run **Section 0** first. It mounts Google Drive and initialises the checkpoint system.
4. Run all remaining cells in order. Any step completed in a prior session is automatically skipped.

**Estimated runtimes on T4 GPU:**

| Notebook | Phase 1 | Phase 2 | Total |
|---|---|---|---|
| 01 Lorenz | ~20 min | ~15 min | ~35 min |
| 02 Duffing | ~20 min | ~15 min | ~35 min |
| 03 KS | ~30 min | ~30 min | ~60 min |
| 04 MG | ~20 min | ~15 min | ~35 min |

### Local setup

```bash
git clone https://github.com/VivakaNand/chaotic-dynamics-learning.git
cd chaotic-dynamics-learning
pip install -r chaotic_dynamics_systems/requirements.txt
jupyter notebook chaotic_dynamics_systems/01_lorenz.ipynb
```

Running locally skips Google Drive checkpointing. Model weights save to the working directory. Ensure a CUDA-capable GPU is available for reasonable runtime.

---

## Reproducibility

All notebooks use fixed random seeds (`np.random.seed(42)`, `tf.random.set_seed(42)`). Results are reproducible to floating-point tolerance on the same hardware and TensorFlow version.

The checkpoint system saves a `progress.json` manifest to Drive after every completed training step. If a Colab session resets mid-training, re-running all cells resumes from the last saved checkpoint automatically.

---

## System Reference

| System | Equation | Parameters | Lyapunov time | D2 (true) |
|---|---|---|---|---|
| Lorenz | 3D ODE | σ=10, ρ=28, β=8/3 | ~1.1 time units | 2.06 |
| Duffing | 2D ODE (conservative) | α=−1, β=1 | periodic | ~1.0 (closed orbit) |
| KS PDE | PDE, L=36 periodic domain | N=64 modes, 3 observed | ~10.6 time units | ~8.0 (full system) |
| Mackey-Glass | Scalar DDE | β=0.2, γ=0.1, n=10, τ=17 | ~182 steps | 3.6 (Farmer 1982) |

---

## Citation

```bibtex
@mastersthesis{vivaka2025chaotic,
  title  = {Data-Efficient Learning of Chaotic Dynamical Systems},
  author = {VivakaNand},
  school = {HSE University},
  year   = {2025}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
