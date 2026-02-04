"""
Collapse probability analysis.

Measures the probability and timing of catastrophic failures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from decision_robustness.engine.simulator import SimulationResult


@dataclass
class CollapseMetrics:
    """
    Metrics related to system collapse/failure.
    
    Attributes:
        collapse_probability: Overall probability of failure
        mean_time_to_collapse: Average time before collapse (for failures only)
        std_time_to_collapse: Std deviation of time to collapse
        collapse_by_horizon: Dict mapping horizon -> collapse probability
        irreversible_collapse_rate: Rate of irreversible failures
        early_collapse_rate: Rate of failures in first 20% of max time
        late_collapse_rate: Rate of failures in last 20% of max time
    """
    collapse_probability: float
    mean_time_to_collapse: Optional[float]
    std_time_to_collapse: Optional[float]
    collapse_by_horizon: Dict[int, float]
    irreversible_collapse_rate: float
    early_collapse_rate: float
    late_collapse_rate: float
    total_runs: int = 0
    collapse_count: int = 0
    
    def describe(self) -> str:
        """Human-readable description."""
        lines = [
            f"Collapse Analysis ({self.total_runs} runs):",
            f"  Overall collapse probability: {self.collapse_probability:.1%}",
        ]
        
        if self.mean_time_to_collapse is not None:
            lines.append(f"  Mean time to collapse: {self.mean_time_to_collapse:.1f} steps")
            lines.append(f"  Std time to collapse: {self.std_time_to_collapse:.1f} steps")
        
        lines.append(f"  Early collapse rate (first 20%): {self.early_collapse_rate:.1%}")
        lines.append(f"  Late collapse rate (last 20%): {self.late_collapse_rate:.1%}")
        lines.append(f"  Irreversible collapse rate: {self.irreversible_collapse_rate:.1%}")
        
        if self.collapse_by_horizon:
            lines.append("  Collapse probability by horizon:")
            for horizon, prob in sorted(self.collapse_by_horizon.items()):
                lines.append(f"    Step {horizon}: {prob:.1%}")
        
        return "\n".join(lines)


class CollapseAnalyzer:
    """
    Analyzes collapse/failure characteristics.
    
    Focuses on understanding when and why systems fail,
    with particular attention to irreversible failures.
    """
    
    def __init__(
        self,
        results: List["SimulationResult"],
        max_steps: Optional[int] = None,
    ):
        """
        Initialize analyzer.
        
        Args:
            results: List of simulation results
            max_steps: Maximum steps in simulation (for horizon calculations)
        """
        self.results = results
        self.max_steps = max_steps or max(r.total_steps for r in results) if results else 100
    
    def compute_metrics(
        self,
        horizons: Optional[List[int]] = None,
    ) -> CollapseMetrics:
        """
        Compute comprehensive collapse metrics.
        
        Args:
            horizons: List of time horizons to compute collapse probability for
            
        Returns:
            CollapseMetrics object
        """
        if not self.results:
            return CollapseMetrics(
                collapse_probability=0.0,
                mean_time_to_collapse=None,
                std_time_to_collapse=None,
                collapse_by_horizon={},
                irreversible_collapse_rate=0.0,
                early_collapse_rate=0.0,
                late_collapse_rate=0.0,
            )
        
        n_total = len(self.results)
        failures = [r for r in self.results if r.is_failure()]
        n_failures = len(failures)
        
        # Basic collapse probability
        collapse_prob = n_failures / n_total
        
        # Time to collapse statistics
        if failures:
            failure_times = [r.total_steps for r in failures]
            mean_ttc = float(np.mean(failure_times))
            std_ttc = float(np.std(failure_times))
        else:
            mean_ttc = None
            std_ttc = None
        
        # Collapse by horizon
        if horizons is None:
            # Default horizons: 10%, 25%, 50%, 75%, 100% of max steps
            horizons = [
                int(self.max_steps * 0.1),
                int(self.max_steps * 0.25),
                int(self.max_steps * 0.5),
                int(self.max_steps * 0.75),
                self.max_steps,
            ]
        
        collapse_by_horizon = {}
        for horizon in horizons:
            failures_by_horizon = sum(1 for r in failures if r.total_steps <= horizon)
            collapse_by_horizon[horizon] = failures_by_horizon / n_total
        
        # Irreversible collapses (marked in metadata or by specific events)
        irreversible_count = 0
        for result in failures:
            events = result.events_occurred
            has_irreversible = any(e.get("is_irreversible", False) for e in events)
            if has_irreversible:
                irreversible_count += 1
        irreversible_rate = irreversible_count / n_total
        
        # Early vs late collapse
        early_threshold = int(self.max_steps * 0.2)
        late_threshold = int(self.max_steps * 0.8)
        
        early_collapses = sum(1 for r in failures if r.total_steps <= early_threshold)
        late_collapses = sum(1 for r in failures if r.total_steps >= late_threshold)
        
        early_rate = early_collapses / n_total
        late_rate = late_collapses / n_total
        
        return CollapseMetrics(
            collapse_probability=collapse_prob,
            mean_time_to_collapse=mean_ttc,
            std_time_to_collapse=std_ttc,
            collapse_by_horizon=collapse_by_horizon,
            irreversible_collapse_rate=irreversible_rate,
            early_collapse_rate=early_rate,
            late_collapse_rate=late_rate,
            total_runs=n_total,
            collapse_count=n_failures,
        )
    
    def get_collapse_triggers(self) -> Dict[str, float]:
        """
        Identify events that commonly precede collapse.
        
        Returns:
            Dict mapping event name to association with collapse
        """
        if not self.results:
            return {}
        
        failures = [r for r in self.results if r.is_failure()]
        successes = [r for r in self.results if r.is_success()]
        
        if not failures or not successes:
            return {}
        
        # Count events in failures vs successes
        failure_events: Dict[str, int] = {}
        success_events: Dict[str, int] = {}
        
        for result in failures:
            for event in result.events_occurred:
                name = event.get("name", "unknown")
                failure_events[name] = failure_events.get(name, 0) + 1
        
        for result in successes:
            for event in result.events_occurred:
                name = event.get("name", "unknown")
                success_events[name] = success_events.get(name, 0) + 1
        
        # Compute relative risk
        all_events = set(failure_events.keys()) | set(success_events.keys())
        triggers = {}
        
        for event_name in all_events:
            fail_rate = failure_events.get(event_name, 0) / len(failures)
            success_rate = success_events.get(event_name, 0) / len(successes)
            
            # Relative risk: how much more likely in failures
            if success_rate > 0:
                relative_risk = fail_rate / success_rate
            else:
                relative_risk = float('inf') if fail_rate > 0 else 1.0
            
            triggers[event_name] = relative_risk
        
        # Sort by risk
        return dict(sorted(triggers.items(), key=lambda x: -x[1]))
    
    def get_conditional_collapse(
        self,
        condition_fn,
    ) -> float:
        """
        Compute collapse probability conditional on a predicate.
        
        Args:
            condition_fn: Function taking SimulationResult, returning bool
            
        Returns:
            Conditional collapse probability
        """
        matching = [r for r in self.results if condition_fn(r)]
        if not matching:
            return 0.0
        
        failures = sum(1 for r in matching if r.is_failure())
        return failures / len(matching)
