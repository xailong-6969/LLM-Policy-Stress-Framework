# Robustness Network Node
# Combined bootstrap peer + worker

FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir hivemind numpy

# Copy framework
COPY decision_robustness/ /app/decision_robustness/
COPY pyproject.toml /app/
COPY scripts/run_node.py /app/

# Install framework
RUN pip install -e .

# Create directories
RUN mkdir -p /app/keys /app/logs

# Expose DHT port
EXPOSE 38751

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import socket; s=socket.socket(); s.connect(('localhost', 38751)); s.close()" || exit 1

# Run node
CMD ["python", "run_node.py", "--port", "38751", "--identity", "/app/keys/node.pem"]
