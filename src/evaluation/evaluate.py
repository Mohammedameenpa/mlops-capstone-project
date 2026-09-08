
import json
from pathlib import Path

import joblib
import mlflow
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# -----------------------------------------
# File paths
# -----------------------------------------

TEST_DATA_PATH = "data/processed/test_data.csv"
MODELS_DIR = "models"
REPORTS_DIR = "reports"
MLFLOW_DB_PATH = "mlflow.db"
EXPERIMENT_NAME = "customer-churn"


# -----------------------------------------
# Model names (must match train.py output)
# -----------------------------------------

MODEL_NAMES = [
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
]


def _set_mlflow_tracking_uri() -> None:
    """Point MLflow at the local SQLite tracking backend.

    Mirrors the built tracking URI used in train.py. The resolved
    path is used as-is so the drive letter and space resolve correctly.
    """
    db_path = Path(MLFLOW_DB_PATH).resolve().as_posix()
    tracking_uri = f"sqlite:///{db_path}"
    mlflow.set_tracking_uri(tracking_uri)


def _find_run_id_by_model_name(model_name: str):
    """Return the MLflow run_id of the latest training run for a model.

    Training creates one run per model in train.py, named with
    the model name (e.g. ``logistic_regression``). Look that run up
    in the existing experiment and return its run_id, or None if
    no matching run exists.
    """
    _set_mlflow_tracking_uri()

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        return None

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{model_name}'",
        order_by=["start_time DESC"],
    )

    if runs.empty:
        return None

    return runs.iloc[0]["run_id"]


def evaluate_model(model_name: str):
    """Evaluate a single trained model on the test set."""

    # -----------------------------------------
    # 1. Load test dataset
    # -----------------------------------------

    df = pd.read_csv(TEST_DATA_PATH)

    print(f"Test data loaded: {df.shape[0]} rows")

    # -----------------------------------------
    # 2. Separate features and target
    # -----------------------------------------

    target_col = "churn"

    y = df[target_col].astype(int)

    X = df.drop(columns=[target_col])

    # -----------------------------------------
    # 3. Load trained model
    # -----------------------------------------

    model_path = f"{MODELS_DIR}/churn_model_{model_name}.joblib"

    model = joblib.load(model_path)

    print(f"Model loaded from {model_path}")

    # -----------------------------------------
    # 4. Make predictions
    # -----------------------------------------

    y_pred = model.predict(X)

    y_probability = model.predict_proba(X)[:, 1]

    # -----------------------------------------
    # 5. Calculate metrics
    # -----------------------------------------

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y, y_probability),
        "pr_auc": average_precision_score(y, y_probability),
    }

    # -----------------------------------------
    # 5a. Log metrics to the existing MLflow run
    # -----------------------------------------
    # Evaluate.py runs standalone (no active MLflow run), so the
    # metrics are attached to the matching training run by its
    # run_id rather than creating a new run.

    run_id = _find_run_id_by_model_name(model_name)

    if run_id is not None:
        mlflow.log_metrics(metrics, run_id=run_id)
        print(f"\nMetrics logged to MLflow run {run_id}")
    else:
        print(
            f"\nNo MLflow run found for '{model_name}' — "
            f"metrics not logged to MLflow."
        )

    # -----------------------------------------
    # 6. Print evaluation metrics
    # -----------------------------------------

    print(f"\nEvaluation Metrics: {model_name}")
    print("=" * 40)

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    # -----------------------------------------
    # 7. Confusion matrix
    # -----------------------------------------

    print("\nConfusion Matrix")
    print("=" * 40)

    cm = confusion_matrix(y, y_pred)

    print(cm)

    # -----------------------------------------
    # 8. Classification report
    # -----------------------------------------

    print("\nClassification Report")
    print("=" * 40)

    report = classification_report(y, y_pred, zero_division=0)

    print(report)

    # -----------------------------------------
    # 9. Save metrics
    # -----------------------------------------

    Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)

    metrics_path = f"{REPORTS_DIR}/metrics_{model_name}.json"

    with open(metrics_path, "w") as file:
        json.dump(metrics, file, indent=4)

    print(f"\nMetrics saved to {metrics_path}")

    return metrics


def evaluate_all():
    """Evaluate all models and print comparison."""

    all_metrics = {}

    for name in MODEL_NAMES:
        print(f"\n{'#' * 50}")
        print(f"# Evaluating: {name}")
        print(f"{'#' * 50}")

        all_metrics[name] = evaluate_model(name)

    # -----------------------------------------
    # Print comparison table
    # -----------------------------------------

    print(f"\n{'=' * 60}")
    print("Model Comparison")
    print(f"{'=' * 60}")

    header = f"{'Model':<25} {'Accuracy':>9} {'F1':>9} {'ROC-AUC':>9}"
    print(header)
    print("-" * 60)

    for name, metrics in all_metrics.items():
        print(
            f"{name:<25} "
            f"{metrics['accuracy']:>9.4f} "
            f"{metrics['f1']:>9.4f} "
            f"{metrics['roc_auc']:>9.4f}"
        )

    return all_metrics


# -----------------------------------------
# Main
# -----------------------------------------

if __name__ == "__main__":
    evaluate_all()
