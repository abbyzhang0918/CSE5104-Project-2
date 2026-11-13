"""Part C: redo the Part B experiments after reducing 11 features to 5 with PCA.

Uses the same data split, CV folds, models and parameter searches as
run_experiments.py. Each pipeline becomes StandardScaler -> PCA -> model.
Run after run_experiments.py; see README.md.
"""

import hashlib
import platform
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import run_experiments as base

N_COMPONENTS = len(base.FEATURES) // 2  # 11 features -> 5 components
ROOT = base.ROOT
OUTPUT = ROOT / "outputs" / "reduced"


def pca_specs(task):
    """Same models and grids as Part B, with scaling + PCA in front of each model."""
    specs = base.model_specs(task)
    for spec in specs.values():
        spec["name"] = spec["name"] + " + PCA"
        spec["pipeline"] = Pipeline([
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=N_COMPONENTS, random_state=base.SEED)),
            ("model", spec["model"]),
        ])
    return specs


def plot_pca_variance(tasks):
    """Fit PCA on the full training set and plot how much variance each component keeps."""
    train = tasks["binary"]["train"]
    scaled = StandardScaler().fit_transform(train[base.FEATURES])
    ratios = PCA(random_state=base.SEED).fit(scaled).explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    positions = np.arange(1, len(ratios) + 1)
    colors = ["#427a9a" if i <= N_COMPONENTS else "#b4bdc6" for i in positions]
    ax.bar(positions, ratios, color=colors, label="Each component")
    ax.plot(positions, np.cumsum(ratios), "o-", color="#c57b40", label="Cumulative")
    ax.axvline(N_COMPONENTS + 0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(positions)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_ylim(0, 1.03)
    ax.set_title(f"PCA on training data | first {N_COMPONENTS} components kept")
    ax.legend(loc="center right")
    base.save_figure(fig, OUTPUT / "pca_explained_variance.png")
    return float(ratios[:N_COMPONENTS].sum())


def export_reduced_data(tasks):
    """Save the training and validation sets after scaling + PCA (fitted on training data)."""
    for task, task_data in tasks.items():
        train, validation = task_data["train"], task_data["validation"]
        target = task_data["target"]
        reducer = Pipeline([
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=N_COMPONENTS, random_state=base.SEED)),
        ]).fit(train[base.FEATURES])
        columns = [f"PC{i + 1}" for i in range(N_COMPONENTS)]
        for partition, frame in [("train", train), ("validation", validation)]:
            reduced = pd.DataFrame(reducer.transform(frame[base.FEATURES]), columns=columns)
            reduced[target] = frame[target].to_numpy()
            reduced.to_csv(OUTPUT / "data" / f"{task}_{partition}_pca.csv", index=False)


def compare():
    """Put Part B (all 11 features) and Part C (5 PCA components) side by side."""
    original = pd.read_csv(ROOT / "outputs" / "results_summary.csv")
    reduced = pd.read_csv(OUTPUT / "results_summary.csv")
    columns = ["cv_mean_selection_score", "validation_accuracy", "validation_weighted_f1",
               "training_seconds", "prediction_seconds"]
    table = original[["task", "model_key", "model", "best_value"] + columns].merge(
        reduced[["task", "model_key", "best_value"] + columns],
        on=["task", "model_key"], suffixes=("_original", "_pca"),
    )
    table.to_csv(OUTPUT / "comparison_original_vs_pca.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for ax, task in zip(axes, ["binary", "multiclass"]):
        subset = table.loc[table.task == task]
        x = np.arange(len(subset))
        ax.bar(x - 0.18, subset.validation_weighted_f1_original, 0.36,
               label="All 11 features", color="#427a9a")
        ax.bar(x + 0.18, subset.validation_weighted_f1_pca, 0.36,
               label=f"{N_COMPONENTS} PCA components", color="#c57b40")
        ax.set_xticks(x, subset.model, rotation=15)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Final validation weighted F1")
        ax.set_title(task.title())
        ax.legend()
    base.save_figure(fig, OUTPUT / "comparison_original_vs_pca.png")
    return table


def main():
    for folder in ["data", "models"]:
        (OUTPUT / folder).mkdir(parents=True, exist_ok=True)
    data_path = ROOT / "data" / "raw" / "winequality-red.csv"
    metadata_path = OUTPUT / "run_metadata.json"
    metadata = {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "method": "StandardScaler + PCA inside each model pipeline",
        "n_components": N_COMPONENTS, "original_features": len(base.FEATURES),
        "random_seed": base.SEED, "cv_folds": base.FOLDS,
        "python": sys.version, "platform": platform.platform(),
        "packages": {"numpy": np.__version__, "pandas": pd.__version__,
                     "scikit-learn": sklearn.__version__},
        "code_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                        for name in ["run_experiments.py", "dimension_reduction.py"]},
    }
    base.write_json(metadata_path, metadata)

    # Rebuild the exact Part B split and CV folds (same seed). prepare_data also
    # writes the split files, so send those copies to a temporary folder.
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        for folder in ["data", "eda"]:
            (temp / folder).mkdir()
        tasks, _ = base.prepare_data(data_path, temp)
    saved = pd.read_csv(ROOT / "outputs" / "data" / "split_manifest.csv", index_col="source_row")
    if not saved.index[saved.partition == "train"].equals(tasks["binary"]["train"].index):
        raise ValueError("Training split differs from Part B. Check the data file and seed.")

    export_reduced_data(tasks)
    variance_kept = plot_pca_variance(tasks)
    print(f"{N_COMPONENTS} PCA components keep {variance_kept:.1%} of the training variance.", flush=True)

    results = []
    with threadpool_limits(limits=1):
        for task, task_data in tasks.items():
            for model_key, spec in pca_specs(task).items():
                fits = base.FOLDS * len(ParameterGrid(base.parameter_grid(spec)))
                print(f"Running {task} / {spec['name']} ({fits} CV fits + final fit)...", flush=True)
                result = base.train_model(task, task_data, model_key, spec, OUTPUT)
                results.append(result)
                print(f"  Best {result['best_params']}; "
                      f"CV {result['selection_metric']}={result['cv_mean_selection_score']:.4f}; "
                      f"validation accuracy={result['validation']['accuracy']:.4f}", flush=True)

    base.summarize_results(results, OUTPUT)
    table = compare()
    print(table[["task", "model", "validation_weighted_f1_original",
                 "validation_weighted_f1_pca"]].round(3).to_string(index=False))
    metadata["status"] = "complete"
    metadata["completed_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["pca_variance_kept"] = variance_kept
    metadata["experiments_completed"] = len(results)
    base.write_json(metadata_path, metadata)
    print(f"Complete. Part C results saved to {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
