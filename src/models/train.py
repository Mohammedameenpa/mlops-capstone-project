from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.features.preprocess import (
    create_preprocessor,
    prepare_features,
)


# -----------------------------------------
# Paths
# -----------------------------------------

DATA_PATH = "data/raw/customer_churn_2026.csv"
MODELS_DIR = "models"
TEST_DATA_PATH = "data/processed/test_data.csv"
MLFLOW_DB_PATH = "mlflow.db"

EXPERIMENT_NAME = "customer-churn"


# -----------------------------------------
# Model configurations
# -----------------------------------------

MODELS = {
    "logistic_regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
    ),
}


def train_model():
    """Train all models, save artifacts, log to MLflow."""

    # Point MLflow at a local SQLite database backend.
    # The resolved path is used as-is; URL-encoding the drive letter
    # or the space makes MLflow resolve to a different file.
    db_path = Path(MLFLOW_DB_PATH).resolve().as_posix()
    tracking_uri = f"sqlite:///{db_path}"
    mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(EXPERIMENT_NAME)

    # -----------------------------------------
    # 1. Load and prepare data
    # -----------------------------------------

    df = pd.read_csv(DATA_PATH)

    customer_ids, X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(f"Training rows: {len(X_train)}")
    print(f"Test rows:     {len(X_test)}")
    print(f"Training churn rate: {y_train.mean():.2%}")
    print(f"Test churn rate:     {y_test.mean():.2%}")

    # -----------------------------------------
    # 2. Create shared preprocessor
    # -----------------------------------------

    preprocessor = create_preprocessor()

    # -----------------------------------------
    # 3. Train each model
    # -----------------------------------------

    Path(MODELS_DIR).mkdir(
        parents=True,
        exist_ok=True,
    )

    results = {}

    for name, classifier in MODELS.items():
        print(f"\n{'=' * 50}")
        print(f"Training: {name}")
        print(f"{'=' * 50}")

        with mlflow.start_run(run_name=name):
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", classifier),
                ]
            )

            pipeline.fit(X_train, y_train)

            # ---------------------------------
            # Log hyperparameters
            # ---------------------------------

            mlflow.log_params(classifier.get_params())

            # ---------------------------------
            # Compute and log metrics
            # ---------------------------------

            y_pred = pipeline.predict(X_test)
            y_probability = pipeline.predict_proba(X_test)[:, 1]

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_probability),
                "pr_auc": average_precision_score(y_test, y_probability),
                "train_accuracy": pipeline.score(X_train, y_train),
                "test_accuracy": pipeline.score(X_test, y_test),
            }

            mlflow.log_metrics(metrics)

            # ---------------------------------
            # Log model artifact
            # ---------------------------------

            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                skops_trusted_types=[
                    "numpy.dtype",
                    "numpy.ndarray",
                    "sklearn.pipeline.Pipeline",
                ],
            )

            # ---------------------------------
            # Log training metadata
            # ---------------------------------

            mlflow.log_params({
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "random_state": 42,
            })

            print(f"Train accuracy: {metrics['train_accuracy']:.4f}")
            print(f"Test accuracy:  {metrics['test_accuracy']:.4f}")
            print(f"F1:             {metrics['f1']:.4f}")
            print(f"ROC-AUC:        {metrics['roc_auc']:.4f}")

            # ---------------------------------
            # Save model locally
            # ---------------------------------

            model_path = f"{MODELS_DIR}/churn_model_{name}.joblib"
            joblib.dump(pipeline, model_path)
            print(f"Model saved to {model_path}")

            results[name] = metrics

    # -----------------------------------------
    # 4. Save test split for evaluation
    # -----------------------------------------

    Path(TEST_DATA_PATH).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    test_df = pd.concat(
        [
            X_test.reset_index(drop=True),
            y_test.reset_index(drop=True),
        ],
        axis=1,
    )

    test_df.to_csv(TEST_DATA_PATH, index=False)
    print(f"\nTest data saved to {TEST_DATA_PATH}")

    # -----------------------------------------
    # 5. Print comparison
    # -----------------------------------------

    print(f"\n{'=' * 50}")
    print("Model Comparison")
    print(f"{'=' * 50}")

    for name, metrics in results.items():
        print(
            f"{name:<25} "
            f"Train: {metrics['train_accuracy']:.4f}  "
            f"Test: {metrics['test_accuracy']:.4f}  "
            f"F1: {metrics['f1']:.4f}  "
            f"ROC-AUC: {metrics['roc_auc']:.4f}"
        )

    return results


if __name__ == "__main__":
    train_model()
