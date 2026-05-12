# Comparative Evaluation of Data-Efficient Methods for Learning Chaotic Dynamical Systems

**Master's Thesis — HSE University**

This repository contains all code, results, and figures for the thesis *"Data-Efficient Learning of Chaotic Dynamical Systems"*. Six learning methods are benchmarked across four dynamical systems of increasing complexity. The central question is whether data-efficient techniques can approach the performance of a fully data-fed model when training data is cut by a factor of ten.

---

## Quick Start

All experiments live in the [`chaotic_dynamics_systems/`](chaotic_dynamics_systems/) folder. Open any notebook directly in Google Colab — no local setup required.

| System | Type | Notebook | Open |
|---|---|---|---|
| Lorenz attractor | 3D ODE, chaos | [`01_lorenz.ipynb`](chaotic_dynamics_systems/01_lorenz.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VivakaNand/chaotic-dynamics-learning/blob/main/chaotic_dynamics_systems/01_lorenz.ipynb) |
| Duffing oscillator | 2D ODE, conservative | [`02_Duffing.ipynb`](chaotic_dynamics_systems/02_Duffing.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VivakaNand/chaotic-dynamics-learning/blob/main/chaotic_dynamics_systems/02_Duffing.ipynb) |
| Kuramoto-Sivashinsky PDE | PDE, partial observability | [`03_KS.ipynb`](chaotic_dynamics_systems/03_KS.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VivakaNand/chaotic-dynamics-learning/blob/main/chaotic_dynamics_systems/03_KS.ipynb) |
| Mackey-Glass DDE | Scalar DDE, delay embedding | [`04_MG.ipynb`](chaotic_dynamics_systems/04_MG.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VivakaNand/chaotic-dynamics-learning/blob/main/chaotic_dynamics_systems/04_MG.ipynb) |

---

## What Is Compared

Seven methods trained on the same LSTM backbone (`LSTM(256) → LSTM(128) → Dense(d)`), differing only in training strategy and data volume:

| Method | Data | Approach |
|---|---|---|
| Baseline RNN | D_large — 100k steps | Standard single-step teacher forcing |
| PINN (large) | D_large | Teacher forcing + physics residual loss |
| Transfer Learning | D_small — 10k steps | Pretrain on D_large, fine-tune head on D_small |
| MAML | D_small | Meta-learn an initialisation that adapts in one gradient step |
| PINN-small | D_small | Physics residual, no pretrained backbone |
| Hybrid (TL + PINN) | D_small | Transfer learning + physics residual combined |
| ESN | D_small | Fixed random reservoir, ridge regression readout only |

Training uses a two-phase protocol. Phase 1 is standard single-step training. Phase 2 applies a curriculum rollout where the prediction horizon K grows from 1 to 16 steps over 40 epochs — this corrects the exposure bias that causes teacher-forced models to fail in closed-loop autonomous prediction.

---

## Results at a Glance

R² on held-out test trajectories (higher is better):

| Method | Lorenz | Duffing | KS | MG |
|---|---|---|---|---|
| Baseline RNN – MS | 0.9493 | 0.9901 | 0.9984 | 0.9996 |
| Transfer Learning – MS | **0.9996** | 0.9857 | 0.9976 | **0.9997** |
| MAML – MS | 0.9992 | 0.9842 | 0.9967 | 0.9995 |
| Hybrid TL+PINN – MS | 0.9980 | 0.9835 | **0.9982** | **0.9997** |
| **ESN** | **1.0000** | **1.0000** | **0.9999** | **0.9999** |

Full results including RMSE, VPT (Valid Prediction Time), and D2 (correlation dimension) are in [`chaotic_dynamics_systems/`](chaotic_dynamics_systems/).

---

## Repository Layout

```
chaotic-dynamics-learning/
├── README.md                        ← you are here
└── chaotic_dynamics_systems/
    ├── 01_lorenz.ipynb
    ├── 02_Duffing.ipynb
    ├── 03_KS.ipynb
    ├── 04_MG.ipynb
    ├── 4 model evaluation CSVs/
    ├── Correlation dimension results 4 files/
    ├── IC robustness results — 8 files/
    ├── VPT results 4 files/
    ├── VPT + D2 Figures/
    ├── model comparison figures/
    ├── figures_Lorenz_*/
    ├── figures_Duffing_*/
    ├── figures_KS_*/
    ├── figures_MG_*/
    ├── requirements.txt
    └── README.md                    ← detailed documentation
```

See [`chaotic_dynamics_systems/README.md`](chaotic_dynamics_systems/README.md) for the full description of every folder, all results tables, and detailed run instructions.

---

## Dependencies

```bash
pip install tensorflow numpy scipy scikit-learn matplotlib seaborn pandas tabulate
```

All notebooks are designed for **Google Colab with a T4 GPU**. Each one runs end-to-end in under 60 minutes and uses Google Drive checkpointing so a session reset never loses progress.

---

## Citation

```bibtex
@mastersthesis{vivaka2025chaotic,
  title  = {Comparative Evaluation of Data-Efficient Methods for Learning Chaotic Dynamical Systems},
  author = {VivakaNand},
  school = {HSE University},
  year   = {2025}
}
```

---

## License

MIT — see [`chaotic_dynamics_systems/LICENSE`](chaotic_dynamics_systems/LICENSE).
