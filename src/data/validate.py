from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "customer_id",
    "age",
    "service_segment",
    "signup_channel",
    "primary_device",
    "tenure_months",
    "contract_type",
    "plan_tier",
    "num_services",
    "has_family_plan",
    "monthly_charges",
    "total_charges",
    "payment_method",
    "auto_pay",
    "payment_failures_last_year",
    "support_tickets_last_year",
    "avg_resolution_hours",
    "uses_mobile_app",
    "app_logins_per_month",
    "avg_session_minutes",
    "monthly_usage_hours",
    "days_since_last_login",
    "email_open_rate_pct",
    "satisfaction_score",
    "nps_score",
    "competitor_offer_received",
    "price_increase_last_year_pct",
    "discount_offered",
    "discount_pct",
    "referrals_made",
    "churn",
]


def validate_data(path: str) -> pd.DataFrame:
    """Load and validate the raw customer churn dataset."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    # Check required columns
    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # Check duplicate customer IDs
    if df["customer_id"].duplicated().any():
        raise ValueError("Duplicate customer_id detected.")

    # Check missing values
    if df[REQUIRED_COLUMNS].isnull().any().any():
        raise ValueError("Missing values detected.")

    # Check target values
    valid_targets = {"Yes", "No"}

    if not set(df["churn"].unique()).issubset(valid_targets):
        raise ValueError(
            f"Invalid churn values detected: {df['churn'].unique()}"
        )

    # Basic numerical sanity checks
    if (df["age"] < 0).any():
        raise ValueError("Invalid age detected.")

    if (df["tenure_months"] < 0).any():
        raise ValueError("Invalid tenure_months detected.")

    if (df["monthly_charges"] < 0).any():
        raise ValueError("Invalid monthly_charges detected.")

    if (df["total_charges"] < 0).any():
        raise ValueError("Invalid total_charges detected.")

    print("Data validation passed.")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Churn rate: {(df['churn'] == 'Yes').mean():.2%}")

    return df


if __name__ == "__main__":
    validate_data("data/raw/customer_churn_2026.csv")