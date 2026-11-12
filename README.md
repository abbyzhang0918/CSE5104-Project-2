# CSE5104 Project 2 — Wine Quality Classification

Implementation, data, and experiment outputs for Random Forest, SVM, ANN, and
Decision Tree on two red-wine quality classification tasks. This repository
contains the Part A/B experiments; it does not generate written question answers.
Part C dimension reduction is not implemented.

## Repository layout

```text
run_experiments.py           Data preparation, model training, plots, and exports
requirements.txt            Python dependencies
data/raw/                   Original red/white CSVs and dataset documentation
outputs/
  data/                     Processed datasets, train/validation splits, CV folds
  eda/                      Feature plots, class counts, and descriptive statistics
  models/
    binary/<model>/          Results for quality below 6 versus at least 6
    multiclass/<model>/      Results for original quality scores 3–8
  results_summary.csv       Metrics and timings for all eight experiments
  model_comparison.png      Validation accuracy and weighted F1 comparison
  run_metadata.json         Run status, versions, seed, and code checksums
```

Each model folder contains `cv_results.csv`, `cross_validation.png`,
`result.json` (including selected parameters), `confusion_matrix.png`, training
and validation predictions, classification reports, and captured warnings.
The processed datasets have 11 predictors followed by the response column.
Source-row identifiers and partition assignments are saved separately in
`outputs/data/split_manifest.csv`.

## Run

Use Python 3.11 or newer. From the project directory:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run_experiments.py
```

On Windows, use `.venv\Scripts\python` instead of `.venv/bin/python`.
The script resolves default paths relative to its own location. Custom paths:

```bash
.venv/bin/python run_experiments.py --data data/raw/winequality-red.csv --output outputs
```

A complete run performs 705 cross-validation fits and eight final fits. Progress
is printed for each model; ANN usually takes the longest. Re-running overwrites
generated results in the selected output directory. Check that
`outputs/run_metadata.json` reports `"status": "complete"` before using them.
Package versions are recorded there; dependency ranges allow installation on
other machines, but version changes can affect results. Runtime naturally varies.

## Data and methods

- The experiments use the red-wine CSV: 1,599 rows and 11 predictors. Removing
  240 exact duplicate records leaves 1,359 observations. The white-wine CSV is
  retained as part of the supplied dataset but is not used by this script.
- Binary response: quality below 6 is 0, quality at least 6 is 1. Multiclass
  response: original quality scores 3–8. Neither response is used as a predictor.
- Both tasks share a 90/10 split stratified by original quality, with seed 42:
  1,223 training observations and 136 final validation observations. Duplicate
  removal precedes splitting to prevent identical records crossing partitions.
- EDA uses training data. Each task uses five stratified shuffled CV folds,
  shared across its four models. The fold assignments are exported.
- SVM and ANN use `StandardScaler` inside a `Pipeline`, so scaling is fitted
  separately within each training fold. Final scaling uses the full training
  partition. Tree models use the original units.
- Binary models are selected by mean CV weighted F1; multiclass models are
  selected by mean CV accuracy. Exact ties use grid enumeration order. RF, SVM,
  and Decision Tree use balanced class weights for binary classification and
  no class weighting for multiclass classification.

### Parameter searches

| Model | Binary search | Multiclass search |
| --- | --- | --- |
| Random Forest | `n_estimators`: 50, 100, 200, 300, 500 | Same tree counts; `min_samples_leaf`: 1, 2, 3, 5, 10 |
| SVM (RBF) | `C`: 0.01, 0.1, 1, 10, 100 | Add `C=1000`; `gamma`: scale, 0.001, 0.01, 0.1, 1, 10 |
| ANN | One hidden layer: 50, 100, 150, 200, 300 neurons | Same widths; `alpha`: 0.0001, 0.001, 0.01, 0.1, 1 |
| Decision Tree | `max_depth`: 2, 4, 6, 8, 10 | Add depths 16, 24; `min_samples_leaf`: 1, 2, 5, 10, 20 |

ANN uses ReLU, Adam, `max_iter=5000`, `tol=0.001`, and
`n_iter_no_change=20`, without early stopping. See `model_specs()` for all
settings. For multiclass searches, each CV plot shows the secondary setting
with the highest CV mean at each primary parameter value; the CSV contains all
combinations and fold scores.

Final models fit all training data after parameter selection. Outputs report
accuracy, weighted precision/recall/F1, macro F1, balanced accuracy, and per-class
metrics. Training and validation prediction times are measured separately with
`perf_counter`, including pipeline preprocessing but excluding scoring and file
exports. Tuning time is reported separately. Runs use one job and one BLAS thread.

The multiclass search was expanded during earlier development after holdout
results had been examined. These validation results are therefore exploratory;
new independent data would be needed to confirm generalization. The script's
parameter selection itself uses only training folds.

## Working with a partner

Edit `model_specs()` to change model settings. Keep the split seed and task
labels consistent when comparing experiments. Use `--output` with a different
folder to preserve existing results during an experiment. Commit the script,
requirements, README, raw data, and experiment outputs. Local virtual environments,
caches, and macOS metadata are excluded through `.gitignore`.

## Sources

Adapted from the ESE4170 project by Abby Zhang, Evelyn Cheng, and Emily Shi,
specifically `Final Project Code.ipynb` and `random_forest.py` at commit
`6d4a7f5c710f9a7634644fd387b8c606b59a3f2a`.
[Reference repository](https://github.com/abbyzhang0918/ESE4170_Project_Group5/tree/6d4a7f5c710f9a7634644fd387b8c606b59a3f2a).
This implementation adds Decision Tree and multiclass ANN and revises duplicate
handling, splitting, scaling, cross-validation, and runtime reporting.

Dataset details: `data/raw/winequality.names`. P. Cortez et al. (2009),
*Modeling wine preferences by data mining from physicochemical properties*,
Decision Support Systems 47(4), 547–553.
[Original study](https://doi.org/10.1016/j.dss.2009.05.016).
