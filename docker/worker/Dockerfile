# Worker Node Docker Image
# Runs simulations and contributes to the network

FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir hivemind numpy

# Copy framework
COPY decision_robustness/ /app/decision_robustness/
COPY pyproject.toml /app/

# Install framework
RUN pip install -e .

# Create directories
RUN mkdir -p /app/keys /app/logs

# Default command (override with your policy)
CMD ["python", "-c", "print('Worker ready. Override CMD to run evaluations.')"]
