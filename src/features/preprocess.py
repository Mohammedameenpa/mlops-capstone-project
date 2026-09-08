
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "churn"
ID_COLUMN = "customer_id"


NUMERICAL_FEATURES = [
    "age",
    "tenure_months",
    "num_services",
    "monthly_charges",
    "total_charges",
    "payment_failures_last_year",
    "support_tickets_last_year",
    "avg_resolution_hours",
    "app_logins_per_month",
    "avg_session_minutes",
    "monthly_usage_hours",
    "days_since_last_login",
    "email_open_rate_pct",
    "satisfaction_score",
    "nps_score",
    "price_increase_last_year_pct",
    "discount_pct",
    "referrals_made",
]


CATEGORICAL_FEATURES = [
    "service_segment",
    "signup_channel",
    "primary_device",
    "contract_type",
    "plan_tier",
    "payment_method",
]


BINARY_FEATURES = [
    "has_family_plan",
    "auto_pay",
    "uses_mobile_app",
    "competitor_offer_received",
    "discount_offered",
]


def create_preprocessor() -> ColumnTransformer:
    """Create the preprocessing pipeline."""

    # Numerical features
    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    # Categorical features
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    # Binary features
    #
    # Boolean values are converted to float
    # inside prepare_features() before this
    # pipeline receives the data.
    #
    # True  -> 1.0
    # False -> 0.0
    binary_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            )
        ]
    )

    # Combine all preprocessing pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                NUMERICAL_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
            (
                "binary",
                binary_pipeline,
                BINARY_FEATURES,
            ),
        ],
        remainder="drop",
    )

    return preprocessor


def prepare_features(df: pd.DataFrame):
    """Separate IDs, features, and target."""

    # Keep customer IDs separately
    customer_ids = df[ID_COLUMN].copy()

    # Remove ID and target from features
    X = df.drop(
        columns=[
            ID_COLUMN,
            TARGET_COLUMN,
        ]
    )

    # Convert binary Boolean columns to numeric
    #
    # True  -> 1.0
    # False -> 0.0
    #
    # This prevents SimpleImputer from receiving
    # Boolean dtype.
    for column in BINARY_FEATURES:
        if column in X.columns:
            X[column] = X[column].astype(float)

    # Convert target from Yes/No to 1/0
    y = (
        df[TARGET_COLUMN]
        .map(
            {
                "No": 0,
                "Yes": 1,
            }
        )
        .astype(int)
    )

    return customer_ids, X, y


def save_preprocessor(
    preprocessor,
    path="models/preprocessor.joblib",
):
    """Save fitted preprocessor."""

    # Create models directory if it doesn't exist
    Path(path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save fitted preprocessor
    joblib.dump(
        preprocessor,
        path,
    )

    print(
        f"Preprocessor saved to {path}"
    )


if __name__ == "__main__":

    # ----------------------------------
    # 1. Load raw dataset
    # ----------------------------------

    df = pd.read_csv(
        "data/raw/customer_churn_2026.csv"
    )

    print(
        "Dataset loaded successfully."
    )

    print(
        f"Dataset shape: {df.shape}"
    )

    # ----------------------------------
    # 2. Prepare features and target
    # ----------------------------------

    customer_ids, X, y = prepare_features(df)

    # ----------------------------------
    # 3. Check binary data types
    # ----------------------------------

    print(
        "\nBinary feature data types:"
    )

    print(
        X[BINARY_FEATURES].dtypes
    )

    # ----------------------------------
    # 4. Create preprocessor
    # ----------------------------------

    preprocessor = create_preprocessor()

    # ----------------------------------
    # 5. Fit and transform
    # ----------------------------------

    X_processed = preprocessor.fit_transform(X)

    # ----------------------------------
    # 6. Display results
    # ----------------------------------

    print(
        "\nPreprocessing completed successfully."
    )

    print(
        f"Original feature count: {X.shape[1]}"
    )

    print(
        f"Processed feature count: "
        f"{X_processed.shape[1]}"
    )

    print(
        f"Target shape: {y.shape}"
    )

    # ----------------------------------
    # 7. Save fitted preprocessor
    # ----------------------------------

    save_preprocessor(
        preprocessor
    )
