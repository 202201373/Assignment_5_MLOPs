# Dockerfile for deploying the trained MNIST model
FROM python:3.10-slim

# Accept the MLflow Run ID as a build argument
ARG RUN_ID

# Set environment variable for the Run ID
ENV MODEL_RUN_ID=${RUN_ID}

# Set the working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY train.py .
COPY check_threshold.py .

# Simulate downloading the model from MLflow
RUN echo "Downloading model for Run ID: ${RUN_ID}"

# Default command
CMD ["python", "-c", "import os; print(f'Serving model from Run ID: {os.environ.get(\"MODEL_RUN_ID\", \"unknown\")}')"]
