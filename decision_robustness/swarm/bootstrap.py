"""
Bootstrap peer launcher script.

Run on your VPS to enable peer discovery for the robustness network.

Usage:
    python -m decision_robustness.swarm.bootstrap --port 38751 --identity ./bootstrap.pem
"""

from decision_robustness.swarm.hivemind_backend import run_bootstrap_peer

if __name__ == "__main__":
    run_bootstrap_peer()
