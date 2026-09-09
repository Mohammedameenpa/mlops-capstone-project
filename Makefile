# =============================================================
# Makefile - Customer Churn MLOps project
# =============================================================
# Quick reference commands for common project tasks.
#
# Prerequisites:
#   - uv        (https://docs.astral.sh/uv/) - the package manager used
#               by this project (uv.lock / pyproject.toml).
#   - Docker    - required only for the docker-build / docker-run targets.
#   - Raw data  - data/raw/customer_churn_2026.csv must exist.
#                 (It is tracked with DVC and is present in the working tree.)
#
# Default shell: this Makefile uses POSIX shell commands.

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

# API model artifact used by run-api (mirrors the Dockerfile MODEL_PATH).
MODEL := models/churn_model_logistic_regression.joblib

# Docker image name used by docker-build / docker-run.
IMAGE := customer-churn-api

# API port. Override when needed, e.g. `make run-api PORT=9000`.
PORT ?= 8000

# -------------------------------------------------------------------
# Targets
# -------------------------------------------------------------------

.PHONY: help install test validate-data train evaluate register \
        run-api mlflow docker-build docker-run clean

## help: list all available targets.
help:
	@echo "Customer Churn MLOps project - available targets:"
	@echo "  make install        Install project + dependencies into .venv"
	@echo "  make test           Run the test suite (pytest)"
	@echo "  make validate-data  Validate the raw dataset"
	@echo "  make train          Train all models, log runs to MLflow, save artifacts"
	@echo "  make evaluate       Evaluate all trained models on the test split"
	@echo "  make register       Register + promote the champion model in MLflow"
	@echo "  make run-api        Start the FastAPI service locally (0.0.0.0:$(PORT))"
	@echo "  make mlflow         Launch the MLflow UI (http://127.0.0.1:5000)"
	@echo "  make docker-build   Build the Docker image ($(IMAGE))"
	@echo "  make docker-run     Run the Docker container on port 8000"
	@echo "  make clean          Remove Python caches and generated processed data"

## install: install the project and all dependencies (incl. dev/test deps).
install:
	uv sync

## test: run the pytest test suite.
test:
	uv run pytest -q

## validate-data: check the raw dataset for required columns, duplicates and missing values.
validate-data:
	uv run python -m src.data.validate

## train: train all models, save joblib artifacts, and log runs to MLflow (mlflow.db).
#  Also writes data/processed/test_data.csv, which evaluate() consumes.
train:
	uv run python -m src.models.train

## evaluate: evaluate every trained model on the test split; writes reports/metrics_*.json.
#  Requires the test split + model artifacts from `make train`.
evaluate:
	uv run python -m src.evaluation.evaluate

## register: train register.py candidates and promote the champion alias in MLflow.
register:
	uv run python -m src.models.register

## run-api: start the FastAPI service, loading the bundled model artifact.
#  Listens on 0.0.0.0; default port 8000 (override with `make run-api PORT=...`).
run-api:
	MODEL_PATH=$(MODEL) uv run uvicorn src.api.main:app --host 0.0.0.0 --port $(PORT)

## mlflow: start the MLflow Tracking UI against the project SQLite backend.
mlflow:
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000

## docker-build: build the deployment image using the project Dockerfile.
docker-build:
	docker build -t $(IMAGE) .

## docker-run: run the image in a container, mapping host port 8000 to the container port 8000.
docker-run:
	docker run --rm -p 8000:8000 $(IMAGE)

## clean: remove Python caches and regenerable processed data.
#  NOTE: does NOT remove models/ (the Docker build + tests need the committed
#  logistic-regression artifact) nor mlflow.db / mlruns/ (the model registry).
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -f .coverage
	rm -rf data/processed