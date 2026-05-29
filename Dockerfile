# Multi-stage-ish single image running the pipeline + API.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# libgomp1 is required by LightGBM at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY nhs_forecast ./nhs_forecast
COPY config ./config

# Install core + api extras (skip heavy torch by default; add `.[all]` if needed).
RUN pip install --upgrade pip && pip install ".[api]"

# Generate an initial forecast at build time so the API has data on first boot.
RUN python -m nhs_forecast.pipeline.cli run --synthetic || true

EXPOSE 8000
CMD ["uvicorn", "nhs_forecast.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
