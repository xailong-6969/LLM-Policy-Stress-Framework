"""
Regret distribution analysis.

Measures decision quality by comparing actual outcomes to
optimal hindsight decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from decision_robustness.engine.simulator import SimulationResult


@dataclass
class RegretDistribution:
    """
    Distribution of regret across simulation runs.
    
    Regret is the difference between optimal possible outcome
    and actual outcome achieved.
    
    Attributes:
        regrets: Raw regret values for each run
        mean_regret: Average regret
        std_regret: Standard deviation of regret
        max_regret: Worst-case regret
        median_regret: Median regret
        total_regret: Sum of all regrets
        regret_percentiles: Percentile distribution
    """
    regrets: List[float]
    mean_regret: float
    std_regret: float
    max_regret: float
    median_regret: float
    total_regret: float
    regret_percentiles: Dict[int, float]
    
    @classmethod
    def from_values(cls, regrets: List[float]) -> "RegretDistribution":
        """Create from list of regret values."""
        if not regrets:
            return cls([], 0, 0, 0, 0, 0, {})
        
        arr = np.array(regrets)
        
        percentiles = {}
        for p in [10, 25, 50, 75, 90, 95, 99]:
            percentiles[p] = float(np.percentile(arr, p))
        
        return cls(
            regrets=regrets,
            mean_regret=float(np.mean(arr)),
            std_regret=float(np.std(arr)),
            max_regret=float(np.max(arr)),
            median_regret=float(np.median(arr)),
            total_regret=float(np.sum(arr)),
            regret_percentiles=percentiles,
        )
    
    def describe(self) -> str:
        """Human-readable description."""
        lines = [
            f"Regret Analysis ({len(self.regrets)} runs):",
            f"  Mean regret: {self.mean_regret:.3f} ± {self.std_regret:.3f}",
            f"  Median regret: {self.median_regret:.3f}",
            f"  Max regret (worst case): {self.max_regret:.3f}",
            f"  95th percentile: {self.regret_percentiles.get(95, 0):.3f}",
        ]
        return "\n".join(lines)


@dataclass
class DecisionRegret:
    """
    Regret information for a single decision.
    
    Attributes:
        timestep: When the decision was made
        action_taken: Action that was chosen
        best_action: Optimal action in hindsight
        regret: Difference in outcome
        alternatives: Other available actions and their outcomes
    """
    timestep: int
    action_taken: str
    best_action: Optional[str]
    regret: float
    alternatives: Dict[str, float] = field(default_factory=dict)


class RegretAnalyzer:
    """
    Analyzes decision regret across simulation runs.
    
    Regret measures how much better the outcome could have been
    with different decisions. This is a key measure of policy quality.
    """
    
    def __init__(
        self,
        results: List["SimulationResult"],
        optimal_score: float = 1.0,
    ):
        """
        Initialize analyzer.
        
        Args:
            results: List of simulation results
            optimal_score: The best possible score (default 1.0)
        """
        self.results = results
        self.optimal_score = optimal_score
    
    def compute_outcome_regret(self) -> RegretDistribution:
        """
        Compute regret based on final outcomes.
        
        Regret = optimal_score - actual_score
        
        Returns:
            RegretDistribution
        """
        if not self.results:
            return RegretDistribution.from_values([])
        
        regrets = [
            self.optimal_score - r.outcome_score
            for r in self.results
        ]
        
        return RegretDistribution.from_values(regrets)
    
    def compute_relative_regret(
        self,
        baseline_results: List["SimulationResult"],
    ) -> RegretDistribution:
        """
        Compute regret relative to a baseline policy.
        
        Args:
            baseline_results: Results from baseline policy
            
        Returns:
            RegretDistribution comparing to baseline
        """
        if not self.results or not baseline_results:
            return RegretDistribution.from_values([])
        
        # Compare matched results by seed
        results_by_seed = {r.seed: r for r in self.results}
        baseline_by_seed = {r.seed: r for r in baseline_results}
        
        regrets = []
        for seed in results_by_seed:
            if seed in baseline_by_seed:
                our_score = results_by_seed[seed].outcome_score
                baseline_score = baseline_by_seed[seed].outcome_score
                regrets.append(baseline_score - our_score)
        
        return RegretDistribution.from_values(regrets)
    
    def compute_cumulative_regret(self) -> Dict[int, float]:
        """
        Compute cumulative regret over time.
        
        Returns:
            Dict mapping timestep to cumulative average regret
        """
        if not self.results:
            return {}
        
        # Get max timesteps
        max_t = max(r.total_steps for r in self.results)
        
        cumulative = {}
        running_regret = 0.0
        running_count = 0
        
        for t in range(max_t + 1):
            # Find runs that ended at this timestep
            ended = [r for r in self.results if r.total_steps == t]
            
            for r in ended:
                regret = self.optimal_score - r.outcome_score
                running_regret += regret
                running_count += 1
            
            if running_count > 0:
                cumulative[t] = running_regret / running_count
        
        return cumulative
    
    def get_regret_by_decision_type(
        self,
        action_classifier: Optional[Callable[[str], str]] = None,
    ) -> Dict[str, float]:
        """
        Analyze regret grouped by decision types.
        
        Args:
            action_classifier: Function to classify action names into types
            
        Returns:
            Dict mapping decision type to average regret for runs with that decision
        """
        if not self.results:
            return {}
        
        if action_classifier is None:
            action_classifier = lambda x: x
        
        # Group runs by primary decision type (most common action)
        type_regrets: Dict[str, List[float]] = {}
        
        for result in self.results:
            # Count action types in trajectory
            action_counts: Dict[str, int] = {}
            for step in result.trajectory:
                action = step.get("action", {})
                action_name = action.get("name", str(action)) if isinstance(action, dict) else str(action)
                action_type = action_classifier(action_name)
                action_counts[action_type] = action_counts.get(action_type, 0) + 1
            
            if action_counts:
                # Get most common action type
                primary_type = max(action_counts, key=action_counts.get)
                regret = self.optimal_score - result.outcome_score
                
                if primary_type not in type_regrets:
                    type_regrets[primary_type] = []
                type_regrets[primary_type].append(regret)
        
        # Average regret by type
        return {
            action_type: np.mean(regrets)
            for action_type, regrets in type_regrets.items()
        }
    
    def identify_costly_decisions(
        self,
        regret_threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Identify decisions that led to high regret.
        
        Args:
            regret_threshold: Minimum regret to flag
            
        Returns:
            List of high-regret decision patterns
        """
        costly = []
        
        for result in self.results:
            regret = self.optimal_score - result.outcome_score
            if regret >= regret_threshold:
                # Analyze trajectory for patterns
                last_actions = []
                if result.trajectory:
                    # Get last few actions before failure
                    for step in result.trajectory[-3:]:
                        action = step.get("action", {})
                        action_name = action.get("name", str(action)) if isinstance(action, dict) else str(action)
                        last_actions.append(action_name)
                
                costly.append({
                    "seed": result.seed,
                    "regret": regret,
                    "outcome": result.outcome,
                    "total_steps": result.total_steps,
                    "last_actions": last_actions,
                    "events": [e.get("name") for e in result.events_occurred[-3:]],
                })
        
        return sorted(costly, key=lambda x: -x["regret"])
