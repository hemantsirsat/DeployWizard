{% if use_gpu %}
# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# Set NVIDIA environment variables
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install Python and other system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-venv \
    build-essential \
    {% if system_deps %}{{ system_deps | join(" \\n    ") }}{% endif %} \
    && rm -rf /var/lib/apt/lists/*

# Create a symlink to python3.10
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

{% else %}
# Use standard Python image for CPU
FROM python:{{ python_version }}-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    {% if system_deps %}{{ system_deps | join(" \\n    ") }}{% endif %} \
    && rm -rf /var/lib/apt/lists/*

{% endif %}

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

# Copy requirements first to leverage Docker cache
{% if requirements_file and requirements_file != 'requirements.txt' %}
# Using custom requirements file
COPY --chown=appuser:appuser app/{{ requirements_file }} requirements.txt
{% else %}
COPY --chown=appuser:appuser app/requirements.txt .
{% endif %}

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
