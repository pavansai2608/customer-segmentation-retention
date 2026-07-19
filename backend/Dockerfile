# Build context: project root (customer-segmentation-retention/)
# Run as: docker build -f backend/Dockerfile .
FROM python:3.11-slim

WORKDIR /app

# System deps needed by xgboost/scikit-learn wheels on slim images
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt dvc

# main.py resolves paths as BASE_DIR.parent / "models", so backend/ and
# models/ must sit side by side inside the image, same as on disk.
COPY backend/ ./backend/
COPY .dvc/ ./.dvc/
COPY models/*.dvc ./models/
COPY .dvcignore ./

WORKDIR /app/backend

COPY backend/entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT:-8000}/health')" || exit 1

CMD ["./entrypoint.sh"]