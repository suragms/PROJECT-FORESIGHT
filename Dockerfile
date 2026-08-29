FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FORESIGHT_LOG_LEVEL=INFO \
    FORESIGHT_ENV=production \
    FORESIGHT_API_HOST=0.0.0.0 \
    FORESIGHT_API_PORT=8000 \
    FORESIGHT_API_AUTH_ENABLED=true \
    PATH="/home/appuser/.local/bin:${PATH}"

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --disable-pip-version-check -r requirements.txt \
    && rm -rf /root/.cache/pip

COPY --chown=appuser:appuser src/ src/
COPY --chown=appuser:appuser docs/ docs/
# Bundles docs/final_model_registry.json for runtime hash verification.
COPY --chown=appuser:appuser models/final/ models/final/

RUN mkdir -p /app/outputs /app/data/samples /app/logs /app/data/auth \
    && chown -R appuser:appuser /app/outputs /app/data /app/logs

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import os,urllib.request; port=os.environ.get('PORT', os.environ.get('FORESIGHT_API_PORT','8000')); urllib.request.urlopen(f'http://127.0.0.1:{port}/health')"

CMD ["sh", "-c", "uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
