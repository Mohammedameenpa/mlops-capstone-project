import os

import joblib
import pytest

from fastapi.testclient import TestClient

MODEL_PATH = "models/churn_model_logistic_regression.joblib"

# The app reads MODEL_PATH at import time. Point startup at the bundled model
# so the lifespan does not depend on a local MLflow registry (absent in CI).
os.environ["MODEL_PATH"] = MODEL_PATH

from src.api.main import app
from src.features.preprocess import prepare_features


def _valid_features() -> dict:
    return {
        "age": 37,
        "service_segment": "Streaming",
        "signup_channel": "Referral",
        "primary_device": "Mobile",
        "tenure_months": 16,
        "contract_type": "Month-to-month",
        "plan_tier": "Premium",
        "num_services": 6,
        "has_family_plan": False,
        "monthly_charges": 29.41,
        "total_charges": 468.65,
        "payment_method": "Bank transfer",
        "auto_pay": False,
        "payment_failures_last_year": 3,
        "support_tickets_last_year": 3,
        "avg_resolution_hours": 7.8,
        "uses_mobile_app": True,
        "app_logins_per_month": 25,
        "avg_session_minutes": 30.1,
        "monthly_usage_hours": 12.5,
        "days_since_last_login": 25,
        "email_open_rate_pct": 35.7,
        "satisfaction_score": 9.6,
        "nps_score": 10,
        "competitor_offer_received": False,
        "price_increase_last_year_pct": 7.1,
        "discount_offered": True,
        "discount_pct": 8.7,
        "referrals_made": 0,
    }


@pytest.fixture(scope="session")
def client():
    model = joblib.load(MODEL_PATH)
    with TestClient(app) as client:
        client.app.state.model = model
        client.app.state.model_name = "customer-churn-logistic-regression"
        client.app.state.model_version = 1
        yield client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_name"] == "customer-churn-logistic-regression"
    assert isinstance(body["model_version"], int)


def test_predict_valid_input(client):
    response = client.post("/api/v1/predict", json={"features": _valid_features()})
    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == "customer-churn-logistic-regression"
    assert isinstance(body["model_version"], int)
    assert isinstance(body["churn"], bool)
    assert isinstance(body["churn_probability"], float)
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn"] == (body["churn_probability"] >= 0.5)


def test_predict_missing_features_field(client):
    response = client.post("/api/v1/predict", json={})
    assert response.status_code == 422


def test_predict_invalid_categorical_value(client):
    payload = _valid_features()
    payload["service_segment"] = "Bogus"
    response = client.post("/api/v1/predict", json={"features": payload})
    assert response.status_code == 422


def test_predict_out_of_range(client):
    payload = _valid_features()
    payload["age"] = 150
    response = client.post("/api/v1/predict", json={"features": payload})
    assert response.status_code == 422


def test_predict_wrong_type(client):
    payload = _valid_features()
    payload["num_services"] = "six"
    response = client.post("/api/v1/predict", json={"features": payload})
    assert response.status_code == 422


def test_predict_missing_required_field(client):
    payload = _valid_features()
    del payload["tenure_months"]
    response = client.post("/api/v1/predict", json={"features": payload})
    assert response.status_code == 422


def test_predict_extra_field_rejected(client):
    payload = _valid_features()
    payload["unknown"] = "unexpected"
    response = client.post("/api/v1/predict", json={"features": payload})
    assert response.status_code == 422
