# =============================================================
# Production Dockerfile for the Customer Churn Inference Service
# =============================================================

# ---------- Build stage: resolve + install dependencies via uv ----------
FROM python:3.13-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# Install uv from the official image
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /uvx /bin/

WORKDIR /app

# Install dependencies and the project itself.
# uv sync produces a virtual environment at /app/.venv.
COPY pyproject.toml uv.lock ./
COPY src ./src
RUN uv sync --no-dev --no-install-project

# ---------- Runtime stage ----------
FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    UVICORN_HOST=0.0.0.0

WORKDIR /app

# Create a non-root user to run the service
RUN groupadd --system appuser \
 && useradd --system --gid appuser --create-home appuser

# Copy the installed virtual environment from the builder
COPY --from=builder /app/.venv ./.venv

# Copy application source
COPY src ./src
# Copy the trained model artifact (bundled via MODEL_PATH at runtime)
COPY models/churn_model_logistic_regression.joblib ./models/churn_model_logistic_regression.joblib

# Set the model path so the app loads the bundled joblib directly
ENV MODEL_PATH=/app/models/churn_model_logistic_regression.joblib

# Own the app directory so the non-root user can write if needed
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${PORT:-8000}/health')" || exit 1

CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
