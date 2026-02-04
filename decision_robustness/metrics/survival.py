"""
Survival curve analysis.

Calculates survival curves showing the probability of "surviving"
(not failing) up to each timestep.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from decision_robustness.engine.simulator import SimulationResult


@dataclass
class SurvivalCurve:
    """
    Kaplan-Meier style survival curve.
    
    Attributes:
        timesteps: Time points
        survival_prob: Probability of survival at each time
        at_risk: Number of runs still at risk at each time
        events: Number of failure events at each time
        confidence_lower: Lower confidence bound (optional)
        confidence_upper: Upper confidence bound (optional)
    """
    timesteps: List[int]
    survival_prob: List[float]
    at_risk: List[int]
    events: List[int]
    confidence_lower: Optional[List[float]] = None
    confidence_upper: Optional[List[float]] = None
    
    def survival_at(self, t: int) -> float:
        """Get survival probability at time t."""
        if t < 0:
            return 1.0
        if t >= len(self.timesteps):
            return self.survival_prob[-1] if self.survival_prob else 0.0
        
        # Find the timestep
        for i, ts in enumerate(self.timesteps):
            if ts > t:
                return self.survival_prob[i - 1] if i > 0 else 1.0
        return self.survival_prob[-1]
    
    def median_survival(self) -> Optional[int]:
        """Get median survival time (time when 50% have failed)."""
        for i, prob in enumerate(self.survival_prob):
            if prob <= 0.5:
                return self.timesteps[i]
        return None
    
    def describe(self) -> str:
        """Human-readable description."""
        median = self.median_survival()
        final_survival = self.survival_prob[-1] if self.survival_prob else 0
        
        lines = [
            f"Survival Analysis:",
            f"  Final survival rate: {final_survival:.1%}",
            f"  Median survival time: {median if median else 'N/A (>50% survive)'}",
        ]
        
        # Key percentiles
        for target in [0.9, 0.75, 0.5, 0.25]:
            for i, prob in enumerate(self.survival_prob):
                if prob <= target:
                    lines.append(f"  Time to {(1-target)*100:.0f}% failure: {self.timesteps[i]}")
                    break
        
        return "\n".join(lines)


class SurvivalAnalyzer:
    """
    Analyzes survival characteristics across simulation runs.
    
    Uses Kaplan-Meier style estimation to compute survival curves
    from censored time-to-event data.
    """
    
    def __init__(self, results: List["SimulationResult"]):
        """
        Initialize analyzer with simulation results.
        
        Args:
            results: List of simulation results
        """
        self.results = results
    
    def compute_survival_curve(
        self,
        confidence_level: float = 0.95,
    ) -> SurvivalCurve:
        """
        Compute Kaplan-Meier survival curve.
        
        Args:
            confidence_level: Confidence level for bounds (default 95%)
            
        Returns:
            SurvivalCurve object
        """
        if not self.results:
            return SurvivalCurve([], [], [], [])
        
        # Extract failure times and censoring status
        # A run is "censored" if it ended without failure (success or timeout)
        failure_times = []
        is_event = []  # True if failure, False if censored
        
        for result in self.results:
            failure_times.append(result.total_steps)
            is_event.append(result.is_failure())
        
        # Get unique event times
        max_time = max(failure_times)
        unique_times = sorted(set(failure_times))
        
        # Kaplan-Meier estimation
        timesteps = []
        survival_prob = []
        at_risk = []
        events = []
        
        n = len(self.results)
        current_survival = 1.0
        
        for t in unique_times:
            # Number at risk just before time t
            n_at_risk = sum(1 for ft in failure_times if ft >= t)
            
            # Number of events at time t
            n_events = sum(1 for i, ft in enumerate(failure_times) 
                          if ft == t and is_event[i])
            
            # Update survival probability
            if n_at_risk > 0 and n_events > 0:
                current_survival *= (n_at_risk - n_events) / n_at_risk
            
            timesteps.append(t)
            survival_prob.append(current_survival)
            at_risk.append(n_at_risk)
            events.append(n_events)
        
        # Compute confidence intervals (Greenwood's formula)
        conf_lower = []
        conf_upper = []
        
        z = 1.96 if confidence_level == 0.95 else 1.645  # Approximate z-score
        
        var_sum = 0.0
        for i, t in enumerate(timesteps):
            n_i = at_risk[i]
            d_i = events[i]
            
            if n_i > d_i and n_i > 0:
                var_sum += d_i / (n_i * (n_i - d_i))
            
            se = survival_prob[i] * np.sqrt(var_sum) if var_sum > 0 else 0
            
            lower = max(0, survival_prob[i] - z * se)
            upper = min(1, survival_prob[i] + z * se)
            
            conf_lower.append(lower)
            conf_upper.append(upper)
        
        return SurvivalCurve(
            timesteps=timesteps,
            survival_prob=survival_prob,
            at_risk=at_risk,
            events=events,
            confidence_lower=conf_lower,
            confidence_upper=conf_upper,
        )
    
    def get_hazard_rate(self, time_window: int = 5) -> List[Tuple[int, float]]:
        """
        Compute hazard rate (failure rate per time step).
        
        Args:
            time_window: Window size for smoothing
            
        Returns:
            List of (time, hazard_rate) tuples
        """
        if not self.results:
            return []
        
        max_time = max(r.total_steps for r in self.results)
        
        hazard_rates = []
        
        for t in range(0, max_time, time_window):
            # At risk at start of window
            at_risk = sum(1 for r in self.results if r.total_steps >= t)
            
            # Failures in window
            failures = sum(1 for r in self.results 
                          if r.is_failure() and t <= r.total_steps < t + time_window)
            
            if at_risk > 0:
                hazard = failures / (at_risk * time_window)
                hazard_rates.append((t, hazard))
        
        return hazard_rates
    
    def compare_survival(
        self,
        other: "SurvivalAnalyzer",
    ) -> Dict[str, Any]:
        """
        Compare survival curves using log-rank test.
        
        Args:
            other: Another SurvivalAnalyzer
            
        Returns:
            Comparison statistics
        """
        curve1 = self.compute_survival_curve()
        curve2 = other.compute_survival_curve()
        
        # Simple comparison metrics
        median1 = curve1.median_survival()
        median2 = curve2.median_survival()
        
        final1 = curve1.survival_prob[-1] if curve1.survival_prob else 0
        final2 = curve2.survival_prob[-1] if curve2.survival_prob else 0
        
        return {
            "median_survival_diff": (median1 or float('inf')) - (median2 or float('inf')),
            "final_survival_diff": final1 - final2,
            "curve1": curve1,
            "curve2": curve2,
        }
