# Predicting the parabolic oxidation rate constant (kp) of steels with machine learning

Machine-learning pipeline that predicts the **parabolic growth rate constant** `kp`
of high-temperature oxidation from a steel's **composition and test temperature**.
Several regression models can be trained, hyperparameter-tuned, compared, and used to
predict `kp` for new compositions.

The target is `log10(kp)` (kp in g²·cm⁻⁴·s⁻¹). Inputs are the raw element fractions
(wt%) plus the temperature, encoded as `invT = 1000 / (T_°C + 273)`.

## Repository structure

| File | Purpose |
|------|---------|
| `ModelSelection.ipynb` | **Main notebook** — train + Optuna-tune one model (set `MODEL_NAME`), evaluate with leak-free CV, plot parity + feature importance, save `<model>.pkl`. |
| `ModelComparison.ipynb` | Train every model with default parameters and rank them by leak-free CV R². |
| `FeatureEngineering.ipynb` | Recursive feature-elimination analysis (permutation importance). |
| `otherModels/` | The non-tree baselines: `SVR`, `KRR`, `MLPR`, `GPR`, `ANN` (PyTorch). |
| `functions.py` | Shared functions: `data_split`, `data_sampling`, `leakfree_cv`, `predict_composition`. |
| `docs/Data_base.xlsx` | Training data (one row per measurement: element columns + `invT` + `kp`). |
| `requirements.txt` | Python dependencies. |

Trained models are saved to the repo root as `<model>.pkl` (e.g. `gbr.pkl`, `xgboost.pkl`).

## Installation

Run everything from the repository root (so `import functions` and the `docs/` paths
resolve). Python 3.10+.

```bash
pip install -r requirements.txt
```

XGBoost / LightGBM / CatBoost are optional and only needed if you select them in
`ModelSelection.ipynb`. PyTorch is only needed for `otherModels/ANN.ipynb`
(see the note in `requirements.txt` if `pip install torch` hits the Windows path limit).

## Usage

### Train and tune a model

Open `ModelSelection.ipynb`, set the model at the top, and run all cells:

```python
MODEL_NAME = 'gbr'   # gbr | histgb | randomforest | extratrees | xgboost | lightgbm | catboost
SEED = 42
N_TRIALS = 100       # Optuna hyperparameter-tuning trials
```

Optuna maximises R² using a **leak-free cross-validation** (the training data is
resampled *inside* each fold, so the bootstrap duplication does not leak into the
validation score). The best model is retrained on the full database and saved as
`<MODEL_NAME>.pkl`.

### Compare all models

Run `ModelComparison.ipynb` to evaluate every installed model on the same data and
rank them by leak-free CV R² (with a bar chart). Use this to pick the best model type.

### Predict a new composition

```python
import joblib, functions as fs

model = joblib.load('gbr.pkl')
# composition in wt% (elements you omit default to 0); temperature in °C
log_kp, kp = fs.predict_composition(
    {'Fe': 70, 'Cr': 18, 'Ni': 8, 'Mn': 1, 'Si': 1},
    model, temperature_C=900)
print(log_kp, kp)
```

## Notes

- **Features = composition + temperature.** External thermodynamic descriptors are not
  used; tree/boosting models learn the relevant interactions from the raw inputs.
- **Judge models by CV R².** The held-out test split deliberately contains the extreme
  min/max compositions, which unfairly punishes non-tree models (SVR/KRR/MLPR/GPR/ANN)
  that extrapolate. Tree/boosting models generalise best.
- Temperature is one of the most important features — `kp` follows an Arrhenius
  (diffusion-controlled) temperature dependence, so it is not temperature-independent.

## Data / attribution

The training dataset is compiled from published high-temperature oxidation measurements
of steels (composition, temperature, and measured `kp`); see Aghaeian et al.,
*Corrosion Science* **221** (2023) 111309, "Predicting the parabolic growth rate constant
for high-temperature oxidation of steels using machine learning models".

The ML pipeline structure was adapted from the code accompanying: Tan, Xingru, et al.,
"Machine learning and high-throughput computational guided development of high temperature
oxidation-resisting Ni-Co-Cr-Al-Fe based high-entropy alloys." *npj Computational Materials*
**11**, no. 1 (2025): 93.
