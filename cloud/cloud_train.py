from __future__ import annotations

import argparse
import pickle
from pathlib import Path
import sys
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from feature_config import KEYPOINT_DIM, POSE_LABELS

DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "cloud_model.pkl"


def align_feature_matrix(data: np.ndarray, target_dim: int, source_name: str) -> np.ndarray:
    """Pad or truncate loaded CSV features to target model dimensionality."""
    current_dim = data.shape[1]
    if current_dim == target_dim:
        return data

    if current_dim < target_dim:
        aligned = np.zeros((data.shape[0], target_dim), dtype=np.float32)
        aligned[:, :current_dim] = data
        print(
            f"Padding {source_name}: {current_dim} -> {target_dim} features "
            "(legacy samples detected)."
        )
        return aligned

    print(
        f"Truncating {source_name}: {current_dim} -> {target_dim} features "
        "(newer samples than target dimensionality)."
    )
    return data[:, :target_dim]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train cloud RandomForest model for pose-to-meme classification."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(ROOT_DIR / "data"),
        help="Directory containing label CSV files.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Output path for trained cloud model pickle.",
    )
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-split", type=int, default=2)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument(
        "--max-features",
        type=str,
        default="sqrt",
        help="One of: sqrt, log2, none, float (0-1], int.",
    )
    parser.add_argument(
        "--class-weight",
        type=str,
        default="none",
        help="One of: none, balanced, balanced_subsample.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--quick-grid-search",
        action="store_true",
        help="Run a small GridSearchCV sweep before training.",
    )
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--scoring", type=str, default="f1_macro")
    parser.add_argument(
        "--report-path",
        type=str,
        default="",
        help="Optional path to save classification + confusion matrix report text.",
    )
    parser.add_argument(
        "--confusion-matrix-csv",
        type=str,
        default="",
        help="Optional path to save confusion matrix as CSV.",
    )
    return parser.parse_args()


def default_config() -> dict[str, Any]:
    return {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": "none",
        "test_size": 0.2,
        "random_state": 42,
        "quick_grid_search": False,
        "cv_folds": 3,
        "scoring": "f1_macro",
    }


def parse_max_features(value: str) -> Any:
    lowered = value.strip().lower()
    if lowered in {"none", "null"}:
        return None
    if lowered in {"sqrt", "log2"}:
        return lowered
    try:
        numeric = float(value)
    except ValueError:
        raise ValueError(
            f"Invalid --max-features '{value}'. Use sqrt/log2/none/float/int."
        ) from None
    if numeric.is_integer() and numeric > 1:
        return int(numeric)
    if 0 < numeric <= 1:
        return numeric
    raise ValueError(
        f"Invalid --max-features '{value}'. Float must be in (0, 1], or integer > 1."
    )


def parse_class_weight(value: str) -> str | None:
    lowered = value.strip().lower()
    if lowered in {"none", "null"}:
        return None
    if lowered in {"balanced", "balanced_subsample"}:
        return lowered
    raise ValueError(
        f"Invalid --class-weight '{value}'. Use none, balanced, or balanced_subsample."
    )


def load_dataset(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[str] = []
    supported_labels = set(POSE_LABELS)

    for csv_path in sorted(data_dir.glob("*.csv")):
        label = csv_path.stem
        if label not in supported_labels:
            print(f"Skipping {csv_path.name}: label '{label}' not in supported label set.")
            continue
        try:
            data = np.loadtxt(csv_path, delimiter=",", dtype=np.float32)
        except ValueError:
            continue

        if data.size == 0:
            continue
        if data.ndim == 1:
            data = data.reshape(1, -1)

        data = align_feature_matrix(data, KEYPOINT_DIM, csv_path.name)

        features.append(data)
        labels.extend([label] * data.shape[0])

    if not features:
        raise RuntimeError("No valid training samples found in data/.")

    x = np.vstack(features)
    y = np.asarray(labels)
    return x, y


def build_model(config: dict[str, Any]) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=int(config["n_estimators"]),
        max_depth=config["max_depth"],
        min_samples_split=int(config["min_samples_split"]),
        min_samples_leaf=int(config["min_samples_leaf"]),
        max_features=parse_max_features(str(config["max_features"])),
        class_weight=parse_class_weight(str(config["class_weight"])),
        random_state=int(config["random_state"]),
        n_jobs=-1,
    )


def build_confusion_matrix_report(cm: np.ndarray, labels: list[str]) -> str:
    cell_width = max(10, max(len(label) for label in labels) + 2)
    header = "true\\pred".ljust(cell_width) + "".join(
        label[: cell_width - 1].ljust(cell_width) for label in labels
    )

    lines = [header]
    for row_idx, label in enumerate(labels):
        row_values = "".join(str(int(value)).ljust(cell_width) for value in cm[row_idx])
        lines.append(label[: cell_width - 1].ljust(cell_width) + row_values)
    return "\n".join(lines)


def save_confusion_matrix_csv(path: Path, cm: np.ndarray, labels: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("," + ",".join(labels) + "\n")
        for label, row in zip(labels, cm):
            handle.write(label + "," + ",".join(str(int(v)) for v in row) + "\n")


def can_make_holdout_split(y: np.ndarray) -> bool:
    unique_labels, counts = np.unique(y, return_counts=True)
    return len(unique_labels) > 1 and np.all(counts >= 2)


def fit_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    config: dict[str, Any],
) -> tuple[RandomForestClassifier, str]:
    if not bool(config.get("quick_grid_search", False)):
        model = build_model(config)
        model.fit(x_train, y_train)
        return model, ""

    min_class_count = int(np.min(np.unique(y_train, return_counts=True)[1]))
    cv_folds = min(max(2, int(config.get("cv_folds", 3))), min_class_count)
    if cv_folds < 2:
        model = build_model(config)
        model.fit(x_train, y_train)
        return model, "Grid search skipped: not enough samples per class for CV."

    param_grid = {
        "n_estimators": [150, 300],
        "max_depth": [None, 20],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", 0.5],
        "class_weight": [None, "balanced_subsample"],
    }
    base = RandomForestClassifier(
        random_state=int(config["random_state"]),
        min_samples_split=int(config["min_samples_split"]),
        n_jobs=-1,
    )
    search = GridSearchCV(
        estimator=base,
        param_grid=param_grid,
        cv=cv_folds,
        scoring=str(config.get("scoring", "f1_macro")),
        n_jobs=-1,
        verbose=0,
    )
    search.fit(x_train, y_train)
    summary = (
        "Quick grid-search complete. "
        f"Best {config.get('scoring', 'f1_macro')}: {search.best_score_:.4f}, "
        f"Best params: {search.best_params_}"
    )
    return search.best_estimator_, summary


def train_model(
    x: np.ndarray,
    y: np.ndarray,
    config: dict[str, Any],
) -> tuple[RandomForestClassifier, str, str, np.ndarray, list[str], str]:
    labels_in_data = [label for label in POSE_LABELS if label in set(y.tolist())]

    if can_make_holdout_split(y):
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=float(config["test_size"]),
            random_state=int(config["random_state"]),
            stratify=y,
        )
        model, grid_summary = fit_model(x_train, y_train, config)
        y_pred = model.predict(x_test)
        report = classification_report(
            y_test,
            y_pred,
            labels=labels_in_data,
            target_names=labels_in_data,
            zero_division=0,
        )
        cm = confusion_matrix(y_test, y_pred, labels=labels_in_data)
        return model, report, "holdout test split", cm, labels_in_data, grid_summary

    model = build_model(config)
    model.fit(x, y)
    y_pred = model.predict(x)
    report = (
        "Not enough class diversity/count for a stratified holdout split. "
        "Metrics below are on the training set and may be optimistic.\n\n"
    ) + classification_report(
        y,
        y_pred,
        labels=labels_in_data,
        target_names=labels_in_data,
        zero_division=0,
    )
    cm = confusion_matrix(y, y_pred, labels=labels_in_data)
    return model, report, "training set", cm, labels_in_data, ""


def train_cloud_model(
    data_dir: Path,
    model_path: Path,
    config: dict[str, Any] | None = None,
) -> str:
    effective_config = default_config()
    if config:
        effective_config.update(config)

    x, y = load_dataset(data_dir)
    model, report, split_scope, cm, cm_labels, grid_summary = train_model(
        x,
        y,
        effective_config,
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as model_file:
        pickle.dump(model, model_file)

    unique_labels, counts = np.unique(y, return_counts=True)
    class_distribution = "\n".join(
        [f"- {label}: {int(count)}" for label, count in zip(unique_labels, counts)]
    )
    cm_report = build_confusion_matrix_report(cm, cm_labels)

    print(f"Loaded samples: {x.shape[0]}")
    print(f"Feature dimension: {x.shape[1]}")
    print("Class distribution:")
    print(class_distribution)
    print(f"Evaluation scope: {split_scope}")
    if grid_summary:
        print(grid_summary)
    print("Classification report:")
    print(report)
    print("Confusion matrix:")
    print(cm_report)
    print(f"Saved cloud model to {model_path}")

    report_path = effective_config.get("report_path")
    if report_path:
        report_file = Path(str(report_path))
        report_file.parent.mkdir(parents=True, exist_ok=True)
        with report_file.open("w", encoding="utf-8") as handle:
            handle.write(f"Loaded samples: {x.shape[0]}\n")
            handle.write(f"Feature dimension: {x.shape[1]}\n")
            handle.write("Class distribution:\n")
            handle.write(class_distribution + "\n\n")
            handle.write(f"Evaluation scope: {split_scope}\n")
            if grid_summary:
                handle.write(grid_summary + "\n")
            handle.write("\nClassification report:\n")
            handle.write(report + "\n")
            handle.write("\nConfusion matrix:\n")
            handle.write(cm_report + "\n")
        print(f"Saved report to {report_file}")

    cm_csv_path = effective_config.get("confusion_matrix_csv")
    if cm_csv_path:
        csv_file = Path(str(cm_csv_path))
        save_confusion_matrix_csv(csv_file, cm, cm_labels)
        print(f"Saved confusion matrix CSV to {csv_file}")

    return report


def main() -> None:
    args = parse_args()
    config = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_samples_split": args.min_samples_split,
        "min_samples_leaf": args.min_samples_leaf,
        "max_features": args.max_features,
        "class_weight": args.class_weight,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "quick_grid_search": args.quick_grid_search,
        "cv_folds": args.cv_folds,
        "scoring": args.scoring,
        "report_path": args.report_path,
        "confusion_matrix_csv": args.confusion_matrix_csv,
    }
    train_cloud_model(Path(args.data_dir), Path(args.model_path), config=config)


if __name__ == "__main__":
    main()
