import hashlib
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow.tracking import MlflowClient
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


DATA_PATH = "data/raw/customer_churn_2026.csv"
MLFLOW_DB_PATH = "mlflow.db"
EXPERIMENT_NAME = "customer-churn"

REGISTERED_MODEL_NAME = "customer-churn-logistic-regression"
PROMOTED_VARIANT = "candidate-a"
PRODUCTION_ALIAS = "champion"

SPLIT_SEED = 42
TEST_SIZE = 0.20

VARIANTS = {
    "candidate-a": LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    ),
    "candidate-b": LogisticRegression(
        C=0.1,
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    ),
}


def _tracking_uri() -> str:
    db_path = Path(MLFLOW_DB_PATH).resolve().as_posix()
    return f"sqlite:///{db_path}"


def _data_fingerprint() -> str:
    return hashlib.sha256(Path(DATA_PATH).read_bytes()).hexdigest()


def _load_data():
    _, X, y = prepare_features(pd.read_csv(DATA_PATH))
    return X, y


def _build_pipeline(classifier):
    return Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            ("classifier", classifier),
        ]
    )


def _find_finished_run_by_variant(variant: str):
    mlflow.set_tracking_uri(_tracking_uri())
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        return None
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(
            "tags.purpose = 'model-registry' "
            f"and tags.variant = '{variant}'"
        ),
        order_by=["start_time DESC"],
    )
    if runs.empty:
        return None
    return runs.iloc[0]["run_id"]


def _train_variant(variant: str, classifier):
    mlflow.set_tracking_uri(_tracking_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    X, y = _load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=SPLIT_SEED,
        stratify=y,
    )

    pipeline = _build_pipeline(classifier)
    pipeline.fit(X_train, y_train)

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

    mlflow.start_run(run_name=variant)
    run_id = mlflow.active_run().info.run_id

    mlflow.set_tags(
        {
            "purpose": "model-registry",
            "variant": variant,
        }
    )

    mlflow.log_params(classifier.get_params())
    mlflow.log_params(
        {
            "split_seed": SPLIT_SEED,
            "test_size": TEST_SIZE,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "data_fingerprint": _data_fingerprint(),
            "feature_columns": len(X.columns),
        }
    )

    mlflow.log_metrics(metrics)

    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="model",
        skops_trusted_types=[
            "numpy.dtype",
            "numpy.ndarray",
            "sklearn.pipeline.Pipeline",
        ],
    )

    mlflow.end_run()
    return run_id


def _register_version(run_id: str, variant: str):
    mlflow.set_tracking_uri(_tracking_uri())
    client = MlflowClient(_tracking_uri())

    existing = client.search_model_versions(
        filter_string=f"name = '{REGISTERED_MODEL_NAME}'"
    )

    for version in existing:
        if version.tags.get("registry.run_id") == run_id:
            print(
                f"Version for run {run_id} already registered "
                f"(version {version.version})."
            )
            return version

    registered = mlflow.register_model(
        model_uri=f"runs:/{run_id}/model",
        name=REGISTERED_MODEL_NAME,
        await_registration_for=300,
        tags={
            "registry.run_id": run_id,
            "registry.variant": variant,
            "registry.source": f"runs:/{run_id}/model",
        },
    )
    print(
        f"Registered {REGISTERED_MODEL_NAME} "
        f"version {registered.version} from run {run_id}."
    )
    return registered


def _promote_selected_version():
    mlflow.set_tracking_uri(_tracking_uri())
    client = MlflowClient(_tracking_uri())

    versions = sorted(
        client.search_model_versions(
            filter_string=f"name = '{REGISTERED_MODEL_NAME}'"
        ),
        key=lambda v: int(v.version),
    )

    if not versions:
        raise RuntimeError(f"No registered model '{REGISTERED_MODEL_NAME}' found.")

    selected = None
    for version in versions:
        run_id = version.tags.get("registry.run_id")
        run = mlflow.get_run(run_id)
        metrics = run.data.metrics
        print(
            f"  version {version.version} "
            f"({version.tags.get('registry.variant')}): "
            f"recall = {metrics.get('recall'):.4f}  "
            f"f1 = {metrics.get('f1'):.4f}  "
            f"pr_auc = {metrics.get('pr_auc'):.4f}"
        )
        if version.tags.get("registry.variant") == PROMOTED_VARIANT:
            selected = version

    if selected is None:
        raise RuntimeError(
            f"No version tagged with variant '{PROMOTED_VARIANT}' found."
        )

    staged = client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=str(selected.version),
        stage="Production",
        archive_existing_versions=True,
    )
    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias=PRODUCTION_ALIAS,
        version=str(selected.version),
    )
    print(
        f"\nPromoted version {selected.version} "
        f"({PROMOTED_VARIANT}) to Production."
    )
    print(f"Alias '{PRODUCTION_ALIAS}' -> version {selected.version}.")
    return staged


def main():
    mlflow.set_tracking_uri(_tracking_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    client = MlflowClient(_tracking_uri())

    existing_models = client.search_registered_models(
        filter_string=f"name = '{REGISTERED_MODEL_NAME}'"
    )
    if not existing_models:
        client.create_registered_model(
            REGISTERED_MODEL_NAME,
            tags={"task": "binary-classification", "target": "churn"},
            description=(
                "Logistic regression for customer churn, selected as the best "
                "model by PR-AUC. Registered from separate training runs for "
                "controlled candidate comparison."
            ),
        )
        print(f"Created registered model '{REGISTERED_MODEL_NAME}'.")

    print("Training / reusing registry candidates:")
    for variant, classifier in VARIANTS.items():
        run_id = _find_finished_run_by_variant(variant)
        if run_id is not None:
            print(f"  {variant}: reusing run {run_id}")
        else:
            run_id = _train_variant(variant, classifier)
            print(f"  {variant}: trained run {run_id}")
        _register_version(run_id, variant)

    print(f"\nSelecting promotion candidate '{PROMOTED_VARIANT}':")
    _promote_selected_version()


if __name__ == "__main__":
    main()