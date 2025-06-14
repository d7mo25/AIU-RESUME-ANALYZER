# Use Python 3.11 slim image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Create necessary directories
RUN mkdir -p templates static

# Copy application code
COPY . .

# Create empty template files if they don't exist (Railway compatibility)
RUN touch templates/index.html || true
RUN touch templates/admin-dashboard.html || true
RUN touch templates/resume-upload.html || true
RUN touch templates/register.html || true
RUN touch templates/admin-login.html || true
RUN touch templates/profile.html || true
RUN touch templates/forgot-password.html || true

# Expose port (Railway will override with PORT env var)
EXPOSE $PORT

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:${PORT:-8000}/api/health || exit 1

# Start the application
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --log-level info"]
