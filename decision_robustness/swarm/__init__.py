"""Swarm module - Parallel world execution and outcome collection."""

from decision_robustness.swarm.executor import SwarmExecutor, SwarmConfig, SwarmResult
from decision_robustness.swarm.collector import OutcomeCollector, OutcomeDistribution

# Import Hivemind backend if available
try:
    from decision_robustness.swarm.hivemind_backend import (
        HivemindSwarmExecutor,
        HivemindConfig,
        BootstrapPeer,
    )
    HIVEMIND_AVAILABLE = True
except ImportError:
    HIVEMIND_AVAILABLE = False
    HivemindSwarmExecutor = None
    HivemindConfig = None
    BootstrapPeer = None

__all__ = [
    "SwarmExecutor",
    "SwarmConfig",
    "SwarmResult",
    "OutcomeCollector",
    "OutcomeDistribution",
    "HivemindSwarmExecutor",
    "HivemindConfig",
    "BootstrapPeer",
    "HIVEMIND_AVAILABLE",
]
