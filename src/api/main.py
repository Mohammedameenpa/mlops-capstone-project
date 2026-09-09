import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient

from src.api.schemas import HealthResponse, PredictRequest, PredictResponse
from src.features.preprocess import prepare_features


MLFLOW_DB_PATH = "mlflow.db"
MODEL_NAME = "customer-churn-logistic-regression"
MODEL_ALIAS = "champion"
MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
MODEL_PATH = os.getenv("MODEL_PATH", "").strip()


def _tracking_uri() -> str:
    db_path = Path(MLFLOW_DB_PATH).resolve().as_posix()
    return f"sqlite:///{db_path}"


def _load_model():
    if MODEL_PATH:
        model = joblib.load(MODEL_PATH)
        return model, MODEL_NAME, 0
    mlflow.set_tracking_uri(_tracking_uri())
    client = MlflowClient(_tracking_uri())
    version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    model = mlflow.sklearn.load_model(MODEL_URI)
    return model, MODEL_NAME, int(version.version)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model, app.state.model_name, app.state.model_version = _load_model()
    yield


app = FastAPI(
    title="Customer Churn Inference Service",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_name=app.state.model_name,
        model_version=app.state.model_version,
    )


@app.post("/api/v1/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    pipeline = app.state.model
    payload = request.features.model_dump()

    frame = pd.DataFrame([payload])
    frame["customer_id"] = "API_INPUT"
    frame["churn"] = "No"
    _, X, _ = prepare_features(frame)

    try:
        probability = float(pipeline.predict_proba(X)[0, 1])
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {error}",
        ) from error

    churn = probability >= 0.5

    return PredictResponse(
        model_name=app.state.model_name,
        model_version=app.state.model_version,
        churn=churn,
        churn_probability=round(probability, 4),
    )