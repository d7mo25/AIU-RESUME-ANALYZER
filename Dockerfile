# Minimal Dockerfile to avoid network timeouts
FROM python:3.11-slim

WORKDIR /app

# Environment variables for faster builds
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DEFAULT_TIMEOUT=100

# Install only essential system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 100 -r requirements.txt

# Copy application files
COPY main.py .
COPY serviceAccountKey.json .
COPY static/ ./static/
COPY templates/ ./templates/

# Create non-root user
RUN adduser --disabled-password appuser && chown -R appuser /app
USER appuser

# Download spaCy model at startup (not build time) and start app
CMD ["sh", "-c", "python -c 'import subprocess, sys; subprocess.run([sys.executable, \"-m\", \"spacy\", \"download\", \"en_core_web_sm\"], check=False)' && uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
