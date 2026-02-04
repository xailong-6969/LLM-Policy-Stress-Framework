"""
Outcome collector and distribution analysis.

Aggregates results from swarm execution into statistical distributions
for analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from decision_robustness.swarm.executor import SwarmResult
    from decision_robustness.engine.simulator import SimulationResult


@dataclass
class OutcomeDistribution:
    """
    Statistical distribution of outcomes.
    
    Attributes:
        values: Raw values
        mean: Mean value
        std: Standard deviation
        median: Median value
        percentiles: Dict of percentile values (e.g., {5: x, 25: y, 75: z, 95: w})
        min_val: Minimum value
        max_val: Maximum value
    """
    values: List[float]
    mean: float = 0.0
    std: float = 0.0
    median: float = 0.0
    percentiles: Dict[int, float] = field(default_factory=dict)
    min_val: float = 0.0
    max_val: float = 0.0
    
    @classmethod
    def from_values(
        cls,
        values: List[float],
        percentile_points: Optional[List[int]] = None,
    ) -> "OutcomeDistribution":
        """Create distribution from list of values."""
        if not values:
            return cls(values=[])
        
        arr = np.array(values)
        percentile_points = percentile_points or [5, 10, 25, 50, 75, 90, 95]
        
        percentiles = {}
        for p in percentile_points:
            percentiles[p] = float(np.percentile(arr, p))
        
        return cls(
            values=values,
            mean=float(np.mean(arr)),
            std=float(np.std(arr)),
            median=float(np.median(arr)),
            percentiles=percentiles,
            min_val=float(np.min(arr)),
            max_val=float(np.max(arr)),
        )
    
    def describe(self) -> str:
        """Human-readable description."""
        return (
            f"Mean: {self.mean:.3f} ± {self.std:.3f}\n"
            f"Median: {self.median:.3f}\n"
            f"Range: [{self.min_val:.3f}, {self.max_val:.3f}]\n"
            f"5th-95th percentile: [{self.percentiles.get(5, 0):.3f}, {self.percentiles.get(95, 0):.3f}]"
        )


@dataclass
class OutcomeSummary:
    """
    Summary of outcome categories.
    
    Attributes:
        total: Total number of runs
        success_count: Number of successes
        failure_count: Number of failures
        timeout_count: Number of timeouts
        success_rate: Proportion of successes
        failure_rate: Proportion of failures
        timeout_rate: Proportion of timeouts
    """
    total: int
    success_count: int
    failure_count: int
    timeout_count: int
    success_rate: float
    failure_rate: float
    timeout_rate: float
    
    @classmethod
    def from_results(cls, results: List["SimulationResult"]) -> "OutcomeSummary":
        """Create summary from list of results."""
        total = len(results)
        if total == 0:
            return cls(0, 0, 0, 0, 0.0, 0.0, 0.0)
        
        success_count = sum(1 for r in results if r.is_success())
        failure_count = sum(1 for r in results if r.is_failure())
        timeout_count = sum(1 for r in results if r.is_timeout())
        
        return cls(
            total=total,
            success_count=success_count,
            failure_count=failure_count,
            timeout_count=timeout_count,
            success_rate=success_count / total,
            failure_rate=failure_count / total,
            timeout_rate=timeout_count / total,
        )
    
    def describe(self) -> str:
        """Human-readable description."""
        return (
            f"Total Runs: {self.total}\n"
            f"Success: {self.success_count} ({self.success_rate:.1%})\n"
            f"Failure: {self.failure_count} ({self.failure_rate:.1%})\n"
            f"Timeout: {self.timeout_count} ({self.timeout_rate:.1%})"
        )


class OutcomeCollector:
    """
    Collects and analyzes outcomes from swarm execution.
    
    Provides methods to compute various distributions and statistics
    from simulation results.
    """
    
    def __init__(self, swarm_result: "SwarmResult"):
        """
        Initialize collector with swarm results.
        
        Args:
            swarm_result: Results from SwarmExecutor.run()
        """
        self.swarm_result = swarm_result
        self.results = swarm_result.results
    
    @property
    def n_runs(self) -> int:
        return len(self.results)
    
    def get_outcome_summary(self) -> OutcomeSummary:
        """Get summary of outcome categories."""
        return OutcomeSummary.from_results(self.results)
    
    def get_score_distribution(self) -> OutcomeDistribution:
        """Get distribution of outcome scores."""
        scores = [r.outcome_score for r in self.results]
        return OutcomeDistribution.from_values(scores)
    
    def get_survival_distribution(self) -> OutcomeDistribution:
        """Get distribution of survival times (steps before failure)."""
        times = [r.survival_time for r in self.results]
        return OutcomeDistribution.from_values(times)
    
    def get_variable_distribution(
        self,
        variable_name: str,
        from_final_state: bool = True,
    ) -> OutcomeDistribution:
        """
        Get distribution of a specific state variable.
        
        Args:
            variable_name: Name of the variable to analyze
            from_final_state: If True, use final state; if False, use initial
            
        Returns:
            Distribution of the variable across all runs
        """
        values = []
        for result in self.results:
            state = result.final_state if from_final_state else result.initial_state
            value = state.get("variables", {}).get(variable_name)
            if value is not None and isinstance(value, (int, float)):
                values.append(float(value))
        
        return OutcomeDistribution.from_values(values)
    
    def get_event_frequencies(self) -> Dict[str, int]:
        """Get frequency of each event type across all runs."""
        frequencies: Dict[str, int] = {}
        
        for result in self.results:
            for event in result.events_occurred:
                name = event.get("name", "unknown")
                frequencies[name] = frequencies.get(name, 0) + 1
        
        return frequencies
    
    def get_events_by_outcome(self) -> Dict[str, Dict[str, int]]:
        """
        Get event frequencies grouped by outcome.
        
        Returns:
            Dict mapping outcome -> {event_name: count}
        """
        by_outcome: Dict[str, Dict[str, int]] = {
            "success": {},
            "failure": {},
            "timeout": {},
        }
        
        for result in self.results:
            outcome = result.outcome or "timeout"
            if outcome not in by_outcome:
                by_outcome[outcome] = {}
            
            for event in result.events_occurred:
                name = event.get("name", "unknown")
                by_outcome[outcome][name] = by_outcome[outcome].get(name, 0) + 1
        
        return by_outcome
    
    def compare_to(self, other: "OutcomeCollector") -> Dict[str, Any]:
        """
        Compare this collector's results to another.
        
        Useful for comparing two different policies.
        
        Args:
            other: Another OutcomeCollector
            
        Returns:
            Dict with comparison statistics
        """
        self_summary = self.get_outcome_summary()
        other_summary = other.get_outcome_summary()
        
        self_scores = self.get_score_distribution()
        other_scores = other.get_score_distribution()
        
        return {
            "success_rate_diff": self_summary.success_rate - other_summary.success_rate,
            "failure_rate_diff": self_summary.failure_rate - other_summary.failure_rate,
            "mean_score_diff": self_scores.mean - other_scores.mean,
            "median_score_diff": self_scores.median - other_scores.median,
            "self_summary": self_summary,
            "other_summary": other_summary,
        }
    
    def get_tail_risk(self, threshold: float = 0.1) -> Dict[str, Any]:
        """
        Analyze tail risk (worst outcomes).
        
        Args:
            threshold: Proportion of worst outcomes to analyze (e.g., 0.1 = worst 10%)
            
        Returns:
            Analysis of tail outcomes
        """
        scores = sorted([r.outcome_score for r in self.results])
        n_tail = max(1, int(len(scores) * threshold))
        tail_scores = scores[:n_tail]
        
        # Get results in the tail
        score_threshold = tail_scores[-1] if tail_scores else 0
        tail_results = [r for r in self.results if r.outcome_score <= score_threshold]
        
        # Analyze tail
        tail_events: Dict[str, int] = {}
        for result in tail_results:
            for event in result.events_occurred:
                name = event.get("name", "unknown")
                tail_events[name] = tail_events.get(name, 0) + 1
        
        return {
            "threshold": threshold,
            "n_tail": len(tail_results),
            "worst_score": min(scores) if scores else 0,
            "tail_threshold_score": score_threshold,
            "tail_event_frequencies": tail_events,
            "tail_failure_rate": sum(1 for r in tail_results if r.is_failure()) / max(1, len(tail_results)),
        }
