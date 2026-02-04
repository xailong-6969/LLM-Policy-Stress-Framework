#!/bin/bash
# ============================================
# Docker Setup Script
# ============================================
# Installs Docker and runs the robustness node
# ============================================

set -e

GREEN='\033[0;32m'
NC='\033[0m'

echo_green() {
    echo -e "${GREEN}$1${NC}"
}

echo "=============================================="
echo "  Robustness Network: Docker Setup"
echo "=============================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo_green ">> Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
    echo ""
    echo "Docker installed. Please log out and back in, then run this script again."
    exit 0
fi

# Check if docker-compose is available
if ! docker compose version &> /dev/null; then
    echo_green ">> Installing Docker Compose plugin..."
    sudo apt-get update
    sudo apt-get install -y docker-compose-plugin
fi

# Clone repo if not present
if [ ! -d "LLM-Policy-Stress-Framework" ]; then
    echo_green ">> Cloning repository..."
    git clone https://github.com/xailong-6969/LLM-Policy-Stress-Framework.git
fi

cd LLM-Policy-Stress-Framework/docker

# Create directories
mkdir -p keys logs

# Build and run
echo_green ">> Building Docker image..."
docker compose build

echo_green ">> Starting node..."
docker compose up -d

echo ""
echo "=============================================="
echo_green "  Docker Setup Complete!"
echo "=============================================="
echo ""
echo "Node is running in background."
echo ""
echo "Commands:"
echo "  View logs:    docker compose logs -f"
echo "  Stop:         docker compose down"
echo "  Restart:      docker compose restart"
echo ""
echo "To get connection string:"
echo "  docker compose logs | grep initial_peers"
echo ""
