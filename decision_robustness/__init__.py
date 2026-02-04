"""
Decision Robustness Framework

Evaluating AI decisions by how many futures they survive.

This framework stress-tests LLM decision policies across many possible futures
to measure robustness under uncertainty and delayed consequences.

Core insight: Single outcomes lie. Distributions don't.
"""

from decision_robustness.engine.world import WorldState, World
from decision_robustness.engine.simulator import Simulator
from decision_robustness.policies.base import Policy, Action, DecisionContext
from decision_robustness.swarm.executor import SwarmExecutor
from decision_robustness.metrics.survival import SurvivalAnalyzer
from decision_robustness.metrics.collapse import CollapseAnalyzer
from decision_robustness.metrics.regret import RegretAnalyzer

__version__ = "0.1.0"
__author__ = "xailong-6969"

__all__ = [
    # Core types
    "WorldState",
    "World",
    "Simulator",
    # Policy interface
    "Policy",
    "Action",
    "DecisionContext",
    # Swarm execution
    "SwarmExecutor",
    # Metrics
    "SurvivalAnalyzer",
    "CollapseAnalyzer",
    "RegretAnalyzer",
]
