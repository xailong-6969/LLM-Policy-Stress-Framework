#!/bin/bash
# ============================================
# Robustness Network: VPS Setup Script
# ============================================
# Installs everything needed to run a node:
# - Python 3.11
# - Hivemind
# - Framework
# ============================================

set -e

GREEN='\033[0;32m'
NC='\033[0m'

echo_green() {
    echo -e "${GREEN}$1${NC}"
}

echo "=============================================="
echo "  Robustness Network: VPS Setup"
echo "=============================================="
echo ""

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "Please run as non-root user with sudo privileges"
    exit 1
fi

# Detect OS
if [ -f /etc/debian_version ]; then
    PKG_MANAGER="apt"
elif [ -f /etc/redhat-release ]; then
    PKG_MANAGER="yum"
else
    echo "Unsupported OS. Please install Python 3.11+ manually."
    exit 1
fi

# Install system dependencies
echo_green ">> Installing system dependencies..."
if [ "$PKG_MANAGER" = "apt" ]; then
    sudo apt update
    sudo apt install -y python3 python3-pip python3-venv git curl build-essential
elif [ "$PKG_MANAGER" = "yum" ]; then
    sudo yum install -y python3 python3-pip git curl gcc gcc-c++ make
fi

# Create working directory
INSTALL_DIR="${INSTALL_DIR:-$HOME/robustness-network}"
echo_green ">> Creating directory: $INSTALL_DIR"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Clone repository
if [ ! -d "LLM-Policy-Stress-Framework" ]; then
    echo_green ">> Cloning repository..."
    git clone https://github.com/xailong-6969/LLM-Policy-Stress-Framework.git
else
    echo_green ">> Updating repository..."
    cd LLM-Policy-Stress-Framework
    git pull
    cd ..
fi

cd LLM-Policy-Stress-Framework

# Create virtual environment
echo_green ">> Creating Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
echo_green ">> Upgrading pip..."
pip install --upgrade pip wheel setuptools

# Install hivemind (required for distributed execution)
echo_green ">> Installing hivemind..."
pip install hivemind

# Install other dependencies
echo_green ">> Installing dependencies..."
pip install numpy

# Install the framework
echo_green ">> Installing decision-robustness framework..."
pip install -e .

# Open firewall
echo_green ">> Configuring firewall..."
if command -v ufw &> /dev/null; then
    sudo ufw allow 38751/tcp
    echo "Port 38751 opened in UFW"
elif command -v firewall-cmd &> /dev/null; then
    sudo firewall-cmd --permanent --add-port=38751/tcp
    sudo firewall-cmd --reload
    echo "Port 38751 opened in firewalld"
else
    echo "No firewall detected. Manually open port 38751 if needed."
fi

# Create keys directory
mkdir -p keys

echo ""
echo "=============================================="
echo_green "  Setup Complete!"
echo "=============================================="
echo ""
echo "To start the node:"
echo "  cd $INSTALL_DIR/LLM-Policy-Stress-Framework"
echo "  source .venv/bin/activate"
echo "  python scripts/run_node.py"
echo ""
echo "To install as systemd service:"
echo "  sudo cp scripts/swarm-gym.service /etc/systemd/system/"
echo "  sudo sed -i 's/YOUR_USERNAME/$USER/g' /etc/systemd/system/swarm-gym.service"
echo "  sudo systemctl daemon-reload"
echo "  sudo systemctl enable swarm-gym"
echo "  sudo systemctl start swarm-gym"
echo ""
