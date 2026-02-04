"""Swarm module - Parallel world execution and outcome collection."""

from decision_robustness.swarm.executor import SwarmExecutor, SwarmConfig
from decision_robustness.swarm.collector import OutcomeCollector, OutcomeDistribution

__all__ = [
    "SwarmExecutor",
    "SwarmConfig",
    "OutcomeCollector",
    "OutcomeDistribution",
]
