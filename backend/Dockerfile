FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    REQUIRE_CHECKPOINT=true

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 1001 appuser && \
    useradd --uid 1001 --gid 1001 --create-home appuser

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/backend/requirements.txt

COPY . /app

RUN mkdir -p /app/backend/uploads /app/backend/artifacts && \
    chown -R appuser:appuser /app/backend/uploads /app/backend/artifacts

USER appuser

EXPOSE ${PORT:-10000}

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT:-10000}/health')" || exit 1

CMD uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-10000}
