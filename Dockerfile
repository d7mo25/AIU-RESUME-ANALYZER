FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Set work directory
WORKDIR /app

# Install system dependencies (including curl for health checks)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model during build (not runtime)
RUN python -m spacy download en_core_web_sm

# Copy the rest of the application
COPY . .

# Create directories if they don't exist
RUN mkdir -p static templates

# Expose port
EXPOSE $PORT

# Remove health check for Railway (Railway handles this externally)

# Run the application
CMD ["python", "main.py"]
