FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH="/app/{{ model_name }}" \
    ENV=production

# Create a non-root user and set up directories
RUN adduser --disabled-password --gecos "" appuser && \
    mkdir -p /app && \
    chown -R appuser:appuser /app

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY --chown=appuser:appuser app/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=appuser:appuser app/ .

# Copy model file to the expected location
COPY --chown=appuser:appuser app/{{ model_name }} /app/

# Switch to non-root user
USER appuser

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
