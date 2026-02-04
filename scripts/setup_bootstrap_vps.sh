#!/bin/bash
# VPS Bootstrap Peer Setup Script
# Run this on your VPS to set up the robustness network bootstrap peer

set -e

echo "=============================================="
echo "  Robustness Network Bootstrap Peer Setup"
echo "=============================================="

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "Please run as non-root user with sudo privileges"
    exit 1
fi

# Install Python if not present
if ! command -v python3 &> /dev/null; then
    echo "Installing Python..."
    sudo apt update
    sudo apt install -y python3 python3-pip python3-venv
fi

# Create directory
INSTALL_DIR="$HOME/robustness-bootstrap"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# Install hivemind
echo "Installing hivemind..."
pip install --upgrade pip
pip install hivemind

# Create bootstrap script
cat > bootstrap_peer.py << 'EOF'
#!/usr/bin/env python3
"""
Bootstrap peer for the Decision Robustness Network.

This creates a DHT node that other peers can use for discovery.
"""

import time
import argparse
from hivemind import DHT


def main():
    parser = argparse.ArgumentParser(description="Run bootstrap peer")
    parser.add_argument("--port", type=int, default=38751)
    parser.add_argument("--identity", type=str, default="./bootstrap.pem")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Starting Bootstrap Peer...")
    print("=" * 60)
    
    dht = DHT(
        host_maddrs=[f"/ip4/0.0.0.0/tcp/{args.port}"],
        identity_path=args.identity,
        start=True,
        wait_timeout=60,
    )
    
    peer_id = dht.peer_id
    maddrs = dht.get_visible_maddrs()
    
    print(f"\nPeer ID: {peer_id}")
    print(f"\nVisible addresses:")
    for addr in maddrs:
        print(f"  {addr}")
    
    print("\n" + "=" * 60)
    print("COPY THIS FOR YOUR CONFIG:")
    print("=" * 60)
    if maddrs:
        print(f'\ninitial_peers=["{maddrs[0]}"]')
    print("\n" + "=" * 60)
    
    print("\nBootstrap peer running. Press Ctrl+C to stop.")
    
    try:
        while True:
            time.sleep(60)
            print(f"[{time.strftime('%H:%M:%S')}] Still running...")
    except KeyboardInterrupt:
        print("\nShutting down...")
        dht.shutdown()


if __name__ == "__main__":
    main()
EOF

# Create systemd service file
cat > robustness-bootstrap.service << EOF
[Unit]
Description=Decision Robustness Bootstrap Peer
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$INSTALL_DIR
ExecStart=$INSTALL_DIR/.venv/bin/python $INSTALL_DIR/bootstrap_peer.py --port 38751
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

echo ""
echo "=============================================="
echo "  Setup Complete!"
echo "=============================================="
echo ""
echo "To start manually:"
echo "  cd $INSTALL_DIR"
echo "  source .venv/bin/activate"
echo "  python bootstrap_peer.py"
echo ""
echo "To install as systemd service:"
echo "  sudo cp robustness-bootstrap.service /etc/systemd/system/"
echo "  sudo systemctl daemon-reload"
echo "  sudo systemctl enable robustness-bootstrap"
echo "  sudo systemctl start robustness-bootstrap"
echo ""
echo "Don't forget to open port 38751 in your firewall!"
echo "  sudo ufw allow 38751/tcp"
echo ""
