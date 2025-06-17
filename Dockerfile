# Foolproof Dockerfile that avoids PORT environment variable issues
FROM python:3.11-slim

WORKDIR /app

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY main.py .
COPY serviceAccountKey.json .
COPY static/ ./static/
COPY templates/ ./templates/

# Create non-root user
RUN adduser --disabled-password appuser && chown -R appuser /app
USER appuser

# Use Python to handle PORT - NO shell expansion needed
CMD ["python", "main.py"]
