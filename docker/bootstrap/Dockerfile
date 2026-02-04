# Bootstrap Peer Docker Image
# Run on VPS to enable peer discovery

FROM python:3.11-slim

WORKDIR /app

# Install hivemind
RUN pip install --no-cache-dir hivemind

# Copy bootstrap code
COPY decision_robustness/swarm/hivemind_backend.py /app/hivemind_backend.py
COPY decision_robustness/swarm/bootstrap.py /app/bootstrap.py

# Create keys directory
RUN mkdir -p /app/keys

# Expose DHT port
EXPOSE 38751

# Default command
CMD ["python", "-c", "from hivemind_backend import run_bootstrap_peer; run_bootstrap_peer()"]
