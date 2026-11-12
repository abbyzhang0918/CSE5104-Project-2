"""Reproduce the wine-quality classification experiments and outputs.

Adapted from ESE4170_Project_Group5's notebook and random_forest.py.
Run from any directory with Python 3.11+; see README.md for installation.
"""

import argparse
import hashlib
import json
import os
import platform
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import stats
from sklearn import ensemble, svm, neural_network, tree
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


SEED = 42
FOLDS = 5
FEATURES = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol",
]
REFERENCE_URL = "https://github.com/abbyzhang0918/ESE4170_Project_Group5"
REFERENCE_COMMIT = "6d4a7f5c710f9a7634644fd387b8c606b59a3f2a"


def model_specs(task="binary"):
    """Define the models and training-only parameter searches for each task."""
    rf_model = ensemble.RandomForestClassifier(
        class_weight="balanced", random_state=SEED, n_jobs=1,
    )
    svm_model = svm.SVC(
        kernel="rbf", gamma="scale", class_weight="balanced", random_state=SEED,
    )
    ann_model = neural_network.MLPClassifier(
        activation="relu", solver="adam", max_iter=5000,
        tol=0.001, early_stopping=False, n_iter_no_change=20, random_state=SEED,
    )
    dt_model = tree.DecisionTreeClassifier(
        class_weight="balanced", random_state=SEED,
    )

    specs = {
        "random_forest": {
            "name": "Random Forest",
            "model": rf_model,
            "parameter": "n_estimators",
            "values": [50, 100, 200, 300, 500],
        },
        "svm": {
            "name": "SVM",
            "model": svm_model,
            "parameter": "C",
            "values": [0.01, 0.1, 1, 10, 100],
        },
        "ann": {
            "name": "ANN",
            "model": ann_model,
            "parameter": "hidden_layer_sizes",
            "values": [(50,), (100,), (150,), (200,), (300,)],
        },
        "decision_tree": {
            "name": "Decision Tree",
            "model": dt_model,
            "parameter": "max_depth",
            "values": [2, 4, 6, 8, 10],
        },
    }
    for spec in specs.values():
        spec["scoring"] = "weighted_f1"
        spec["extra_grid"] = {}
    if task == "multiclass":
        for spec in specs.values():
            spec["scoring"] = "accuracy"
        for key in ["random_forest", "svm", "decision_tree"]:
            specs[key]["model"].set_params(class_weight=None)
        specs["random_forest"]["extra_grid"] = {"min_samples_leaf": [1, 2, 3, 5, 10]}
        specs["svm"]["values"] = [0.01, 0.1, 1, 10, 100, 1000]
        specs["svm"]["extra_grid"] = {"gamma": ["scale", 0.001, 0.01, 0.1, 1, 10]}
        specs["ann"]["extra_grid"] = {"alpha": [0.0001, 0.001, 0.01, 0.1, 1]}
        specs["decision_tree"]["values"] = [2, 4, 6, 8, 10, 16, 24]
        specs["decision_tree"]["extra_grid"] = {"min_samples_leaf": [1, 2, 5, 10, 20]}

    # Putting scaling inside the pipeline fits it separately in each CV fold.
    for model_key, spec in specs.items():
        steps = []
        if model_key in ["svm", "ann"]:
            steps.append(("scale", StandardScaler()))
        steps.append(("model", spec["model"]))
        spec["pipeline"] = Pipeline(steps)
    return specs


def parameter_grid(spec):
    grid = {"model__" + spec["parameter"]: spec["values"]}
    for name, values in spec["extra_grid"].items():
        grid["model__" + name] = values
    return grid


def write_json(path, content):
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")


def save_figure(fig, path):
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def prepare_data(data_path, output):
    raw = pd.read_csv(data_path, sep=";")
    if raw.columns.tolist() != FEATURES + ["quality"]:
        raise ValueError("Expected the original red wine CSV with 11 features and quality last.")
    if not np.isfinite(raw.to_numpy(dtype=float)).all():
        raise ValueError("Unexpected missing/non-finite values. Review preprocessing before running.")
    if not np.equal(raw.quality, raw.quality.astype(int)).all():
        raise ValueError("Quality labels must be integers.")

    # Exact duplicate records have no distinct sample identifiers in this source.
    # Keep one copy before splitting so identical records cannot cross partitions.
    clean = raw.drop_duplicates().copy()
    if clean[FEATURES].duplicated().any():
        raise ValueError("Identical feature rows have conflicting labels; review them before splitting.")
    clean.index.name = "source_row"
    if len(clean) < 200 or len(FEATURES) < 10:
        raise ValueError("Dataset does not meet the assignment's minimum size.")
    if clean.quality.nunique() < 3:
        raise ValueError("The multiclass task needs at least three classes.")

    # Stratify by original quality, preserving both tasks with the same held-out rows.
    train_ids, validation_ids = train_test_split(
        clean.index.to_numpy(), test_size=0.10, random_state=SEED,
        stratify=clean.quality,
    )
    train_ids = np.sort(train_ids)
    validation_ids = np.sort(validation_ids)
    split = pd.DataFrame(index=clean.index)
    split["partition"] = "train"
    split.loc[validation_ids, "partition"] = "validation"
    split["quality"] = clean.quality
    split["quality_binary"] = (clean.quality >= 6).astype(int)
    split.to_csv(output / "data" / "split_manifest.csv")

    tasks = {}
    for task, target in [("binary", "quality_binary"), ("multiclass", "quality")]:
        frame = clean[FEATURES].copy()
        labels = clean.quality
        if task == "binary":
            labels = (labels >= 6).astype(int)
        frame[target] = labels
        train = frame.loc[train_ids]
        validation = frame.loc[validation_ids]
        if train[target].value_counts().min() < FOLDS:
            raise ValueError(f"Too few training samples per class for {FOLDS}-fold CV in {task}.")
        frame.to_csv(output / "data" / f"{task}_full.csv", index=False)
        train.to_csv(output / "data" / f"{task}_train.csv", index=False)
        validation.to_csv(output / "data" / f"{task}_validation.csv", index=False)
        cv = list(StratifiedKFold(
            n_splits=FOLDS, shuffle=True, random_state=SEED,
        ).split(train[FEATURES], train[target]))
        fold_manifest = pd.DataFrame({"source_row": train_ids, "cv_fold": -1})
        for fold_number, (_, fold_validation) in enumerate(cv, 1):
            fold_manifest.loc[fold_validation, "cv_fold"] = fold_number
        fold_manifest.to_csv(output / "data" / f"{task}_cv_folds.csv", index=False)
        counts = pd.DataFrame({
            "full": frame[target].value_counts(),
            "train": train[target].value_counts(),
            "validation": validation[target].value_counts(),
        }).fillna(0).astype(int).sort_index()
        counts.index.name = "class"
        counts.to_csv(output / "eda" / f"{task}_class_counts.csv")
        tasks[task] = {"train": train, "validation": validation,
                       "target": target, "cv": cv, "counts": counts}

    summary = {
        "raw_samples": len(raw), "features": len(FEATURES),
        "exact_duplicates_removed": len(raw) - len(clean),
        "analysis_samples": len(clean), "train_samples": len(train_ids),
        "validation_samples": len(validation_ids), "missing_values": int(raw.isna().sum().sum()),
        "quality_classes": sorted(int(x) for x in clean.quality.unique()),
        "input_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
    }
    write_json(output / "eda" / "data_summary.json", summary)
    return tasks, summary


def plot_eda(tasks, output):
    # Reserve final validation observations before any exploratory modelling work.
    train = tasks["multiclass"]["train"]
    directory = output / "eda"
    descriptive = train[FEATURES].describe().T
    descriptive["skewness"] = train[FEATURES].skew()
    descriptive["excess_kurtosis"] = train[FEATURES].kurt()
    descriptive.to_csv(directory / "training_feature_summary.csv")

    fig, axes = plt.subplots(3, 4, figsize=(15, 10), layout="constrained")
    for ax, feature in zip(axes.flat, FEATURES):
        ax.hist(train[feature], bins=30, color="#427a9a", edgecolor="white", linewidth=0.3)
        ax.set_title(feature)
        ax.set_ylabel("Training samples")
    axes.flat[-1].axis("off")
    fig.suptitle("Red wine feature distributions | training data only", fontsize=16)
    save_figure(fig, directory / "feature_distributions.png")

    fig, axes = plt.subplots(3, 4, figsize=(15, 10), layout="constrained")
    for ax, feature in zip(axes.flat, FEATURES):
        stats.probplot(train[feature], dist="norm", plot=ax)
        ax.set_title(feature)
        ax.set_xlabel("Theoretical normal quantile")
        ax.set_ylabel("Observed value")
    axes.flat[-1].axis("off")
    fig.suptitle("Normal Q-Q plots | training data only", fontsize=16)
    save_figure(fig, directory / "normal_qq_plots.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, (task, data) in zip(axes, tasks.items()):
        counts = data["counts"]["train"]
        bars = ax.bar(counts.index.astype(str), counts, color="#427a9a")
        ax.bar_label(bars, padding=3)
        ax.set_ylim(0, counts.max() * 1.2)
        ax.set_title(f"{task.title()} task")
        ax.set_xlabel(data["target"])
        ax.set_ylabel("Training samples")
    save_figure(fig, directory / "label_distributions.png")

    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    qualities = sorted(train.quality.unique())
    ax.boxplot([train.loc[train.quality == q, "alcohol"] for q in qualities])
    ax.set_xticks(range(1, len(qualities) + 1), qualities)
    ax.set_xlabel("Quality score")
    ax.set_ylabel("Alcohol (% by volume)")
    ax.set_title("Alcohol by wine quality | training data only")
    save_figure(fig, directory / "alcohol_by_quality.png")

    correlation = train.corr()
    fig, ax = plt.subplots(figsize=(11, 9), layout="constrained")
    chart = ax.imshow(correlation, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(correlation)), correlation.columns, rotation=60, ha="right")
    ax.set_yticks(range(len(correlation)), correlation.columns)
    for row in range(len(correlation)):
        for col in range(len(correlation)):
            value = correlation.iloc[row, col]
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(value) > 0.65 else "black")
    fig.colorbar(chart, ax=ax, shrink=0.8, label="Pearson correlation")
    ax.set_title("Feature and quality correlations | training data only")
    save_figure(fig, directory / "correlations.png")


def performance(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def candidate_label(value):
    if isinstance(value, tuple):
        return str(value[0])
    return str(value)


def train_model(task, task_data, model_key, spec, output):
    directory = output / "models" / task / model_key
    directory.mkdir(parents=True, exist_ok=True)
    train, validation = task_data["train"], task_data["validation"]
    target = task_data["target"]
    x_train, y_train = train[FEATURES], train[target]
    x_validation, y_validation = validation[FEATURES], validation[target]
    parameter = "model__" + spec["parameter"]
    selection_metric = spec["scoring"]
    search = GridSearchCV(
        spec["pipeline"], parameter_grid(spec),
        scoring={"accuracy": "accuracy", "weighted_f1": "f1_weighted", "macro_f1": "f1_macro"},
        cv=task_data["cv"], refit=False,
        n_jobs=1, return_train_score=True, error_score="raise",
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", ConvergenceWarning)
        started = perf_counter()
        search.fit(x_train, y_train)
        tuning_seconds = perf_counter() - started
        cv_results = pd.DataFrame(search.cv_results_)
        # Keep the generic score columns for downstream readers of earlier runs.
        for prefix in ["mean_test", "std_test", "mean_train", "std_train"]:
            cv_results[f"{prefix}_score"] = cv_results[f"{prefix}_{selection_metric}"]
        for fold in range(FOLDS):
            cv_results[f"split{fold}_test_score"] = cv_results[f"split{fold}_test_{selection_metric}"]
        cv_results.to_csv(directory / "cv_results.csv", index=False)
        if not np.isfinite(cv_results["mean_test_score"]).all():
            raise ValueError(f"Non-finite CV score for {task}/{model_key}.")
        best_index = int(cv_results.mean_test_score.idxmax())
        best_params = cv_results.loc[best_index, "params"]
        best_value = best_params[parameter]
        best_score = float(cv_results.loc[best_index, "mean_test_score"])
        best_model = clone(spec["pipeline"]).set_params(**best_params)

        # Time only the final full-training-data fit, including scaling when used.
        started = perf_counter()
        best_model.fit(x_train, y_train)
        training_seconds = perf_counter() - started
        train_prediction = best_model.predict(x_train)

        # Exactly one final validation prediction call per selected model.
        started = perf_counter()
        validation_prediction = best_model.predict(x_validation)
        prediction_seconds = perf_counter() - started

    warning_records = [{"category": w.category.__name__, "message": str(w.message)}
                       for w in captured]
    write_json(directory / "warnings.json", warning_records)
    for partition, frame, predicted in [
        ("train", train, train_prediction),
        ("validation", validation, validation_prediction),
    ]:
        pd.DataFrame({"source_row": frame.index, "actual": frame[target],
                      "predicted": predicted}).to_csv(directory / f"{partition}_predictions.csv", index=False)
        report = classification_report(frame[target], predicted, output_dict=True, zero_division=0)
        write_json(directory / f"{partition}_classification_report.json", report)

    result = {
        "task": task, "model": spec["name"], "model_key": model_key,
        "hyperparameter": spec["parameter"], "best_value": candidate_label(best_value),
        "best_params": {name.removeprefix("model__"): value for name, value in best_params.items()},
        "selection_metric": selection_metric,
        "cv_mean_selection_score": best_score,
        "cv_std_selection_score": float(cv_results.loc[best_index, "std_test_score"]),
        "cv_mean_weighted_f1": float(cv_results.loc[best_index, "mean_test_weighted_f1"]),
        "cv_std_weighted_f1": float(cv_results.loc[best_index, "std_test_weighted_f1"]),
        "cv_mean_accuracy": float(cv_results.loc[best_index, "mean_test_accuracy"]),
        "cv_std_accuracy": float(cv_results.loc[best_index, "std_test_accuracy"]),
        "tuning_seconds": tuning_seconds, "training_seconds": training_seconds,
        "prediction_seconds": prediction_seconds,
        "train": performance(y_train, train_prediction),
        "validation": performance(y_validation, validation_prediction),
        "convergence_warning_count": sum(w["category"] == "ConvergenceWarning" for w in warning_records),
        "final_iterations": getattr(best_model.named_steps["model"], "n_iter_", None),
    }
    # SVC stores one iteration count per one-vs-one classifier.
    if isinstance(result["final_iterations"], np.ndarray):
        result["final_iterations"] = result["final_iterations"].tolist()
    write_json(directory / "result.json", result)

    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    positions = np.arange(len(spec["values"]))
    profile_indices = []
    for value in spec["values"]:
        matching = cv_results.params.map(lambda params: params[parameter] == value)
        profile_indices.append(cv_results.loc[matching, "mean_test_score"].idxmax())
    profile = cv_results.loc[profile_indices].reset_index(drop=True)
    best_position = spec["values"].index(best_value)
    for fold in range(FOLDS):
        ax.plot(positions, profile[f"split{fold}_test_score"],
                color="#b4bdc6", alpha=0.65, linewidth=1)
    ax.errorbar(positions, profile.mean_test_score, yerr=profile.std_test_score,
                fmt="o-", color="#235d80", capsize=5, label="CV mean +/- 1 SD")
    ax.scatter(best_position, best_score, marker="*", s=180, color="#b35c13",
               zorder=5, label="Selected by highest mean")
    ax.set_xticks(positions, [candidate_label(value) for value in spec["values"]])
    ax.set_xlabel(spec["parameter"])
    ax.set_ylabel(selection_metric.replace("_", " ").title())
    ax.set_ylim(0, 1.03)
    ax.set_title(f"{spec['name']} | {task} | 5-fold cross-validation")
    if spec["extra_grid"]:
        secondary = ", ".join(spec["extra_grid"])
        ax.set_xlabel(f"{spec['parameter']}\nAt each value, select the best CV mean over {secondary}")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.2)
    save_figure(fig, directory / "cross_validation.png")

    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    ConfusionMatrixDisplay.from_predictions(
        y_validation, validation_prediction, labels=sorted(train[target].unique()),
        cmap="Blues", colorbar=False, ax=ax,
    )
    ax.set_title(f"{spec['name']} | {task} | final validation")
    save_figure(fig, directory / "confusion_matrix.png")
    return result


def summarize_results(results, output):
    rows = []
    for result in results:
        row = {key: value for key, value in result.items()
               if key not in {"train", "validation", "best_params", "final_iterations"}}
        for partition in ["train", "validation"]:
            row.update({f"{partition}_{key}": value for key, value in result[partition].items()})
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "results_summary.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for ax, task in zip(axes, ["binary", "multiclass"]):
        subset = summary.loc[summary.task == task]
        x = np.arange(len(subset))
        ax.bar(x - 0.18, subset.validation_accuracy, 0.36, label="Accuracy", color="#427a9a")
        ax.bar(x + 0.18, subset.validation_weighted_f1, 0.36, label="Weighted F1", color="#c57b40")
        ax.set_xticks(x, subset.model, rotation=15)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Final validation score")
        ax.set_title(task.title())
        ax.legend()
    save_figure(fig, output / "model_comparison.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "raw" / "winequality-red.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    data_path, output = args.data.resolve(), args.output.resolve()
    for folder in ["data", "eda", "models"]:
        (output / folder).mkdir(parents=True, exist_ok=True)
    metadata_path = output / "run_metadata.json"
    metadata = {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "reference_repository": REFERENCE_URL, "reference_commit": REFERENCE_COMMIT,
        "source_file": str(data_path), "random_seed": SEED, "cv_folds": FOLDS,
        "validation_fraction_requested": 0.1,
        "selection_metric": {"binary": "weighted_f1", "multiclass": "accuracy"},
        "parallel_jobs": 1, "blas_threads": 1,
        "python": sys.version, "platform": platform.platform(),
        "packages": {"numpy": np.__version__, "pandas": pd.__version__,
                     "scipy": scipy.__version__, "scikit-learn": sklearn.__version__,
                     "matplotlib": matplotlib.__version__},
        "code_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                        for name in ["run_experiments.py"]},
    }
    write_json(metadata_path, metadata)
    try:
        tasks, data_summary = prepare_data(data_path, output)
        print(f"Prepared {data_summary['analysis_samples']} unique samples, "
              f"{data_summary['train_samples']} training / "
              f"{data_summary['validation_samples']} validation.", flush=True)
        plot_eda(tasks, output)
        specs_by_task = {task: model_specs(task) for task in tasks}
        results = []
        with threadpool_limits(limits=1):
            for task, task_data in tasks.items():
                for model_key, spec in specs_by_task[task].items():
                    fits = FOLDS * len(ParameterGrid(parameter_grid(spec)))
                    print(f"Running {task} / {spec['name']} ({fits} CV fits + final fit)...", flush=True)
                    result = train_model(task, task_data, model_key, spec, output)
                    results.append(result)
                    print(f"  Best {result['best_params']}; "
                          f"CV {result['selection_metric']}={result['cv_mean_selection_score']:.4f}; "
                          f"validation accuracy={result['validation']['accuracy']:.4f}; "
                          f"convergence warnings={result['convergence_warning_count']}", flush=True)
        summarize_results(results, output)
        metadata["status"] = "complete"
        metadata["completed_utc"] = datetime.now(timezone.utc).isoformat()
        metadata["convergence_warning_count"] = sum(r["convergence_warning_count"] for r in results)
        metadata["experiments_completed"] = len(results)
        write_json(metadata_path, metadata)
        print(f"Complete. Experiment results saved to {output}", flush=True)
    except Exception as error:
        metadata["status"] = "failed"
        metadata["error"] = str(error)
        write_json(metadata_path, metadata)
        raise


if __name__ == "__main__":
    main()
