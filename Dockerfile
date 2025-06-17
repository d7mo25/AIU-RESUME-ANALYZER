FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
# Remove any HEALTHCHECK lines
CMD ["python", "main.py"]
