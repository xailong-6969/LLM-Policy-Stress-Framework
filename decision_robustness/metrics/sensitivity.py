"""
Sensitivity analysis.

Measures how sensitive policy outcomes are to noise, perturbations,
and parameter variations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from decision_robustness.engine.simulator import SimulationResult


@dataclass
class SensitivityReport:
    """
    Report on policy sensitivity to various factors.
    
    Attributes:
        noise_sensitivity: How much outcomes vary with noise
        parameter_sensitivity: Sensitivity to parameter changes
        initial_condition_sensitivity: Sensitivity to initial state
        brittleness_score: Overall brittleness (0-1, higher = more brittle)
        stability_score: Inverse of brittleness (0-1, higher = more stable)
    """
    noise_sensitivity: float
    parameter_sensitivity: Dict[str, float]
    initial_condition_sensitivity: float
    brittleness_score: float
    stability_score: float
    details: Dict[str, Any] = field(default_factory=dict)
    
    def describe(self) -> str:
        """Human-readable description."""
        stability_grade = (
            "A (Very Stable)" if self.stability_score >= 0.8 else
            "B (Stable)" if self.stability_score >= 0.6 else
            "C (Moderate)" if self.stability_score >= 0.4 else
            "D (Sensitive)" if self.stability_score >= 0.2 else
            "F (Brittle)"
        )
        
        lines = [
            f"Sensitivity Analysis:",
            f"  Brittleness Score: {self.brittleness_score:.3f}",
            f"  Stability Score: {self.stability_score:.3f} ({stability_grade})",
            f"  Noise Sensitivity: {self.noise_sensitivity:.3f}",
            f"  Initial Condition Sensitivity: {self.initial_condition_sensitivity:.3f}",
        ]
        
        if self.parameter_sensitivity:
            lines.append("  Parameter Sensitivity:")
            for param, sens in sorted(self.parameter_sensitivity.items(), 
                                       key=lambda x: -x[1]):
                lines.append(f"    {param}: {sens:.3f}")
        
        return "\n".join(lines)


class SensitivityAnalyzer:
    """
    Analyzes policy sensitivity to perturbations.
    
    A robust policy should produce consistent outcomes across
    different random seeds and parameter variations.
    """
    
    def __init__(self, results: List["SimulationResult"]):
        """
        Initialize analyzer.
        
        Args:
            results: List of simulation results
        """
        self.results = results
    
    def compute_noise_sensitivity(self) -> float:
        """
        Compute sensitivity to random noise (seed variation).
        
        Measured as coefficient of variation of outcome scores.
        Higher values = more sensitive to noise.
        
        Returns:
            Noise sensitivity score (0 = insensitive, higher = more sensitive)
        """
        if not self.results:
            return 0.0
        
        scores = [r.outcome_score for r in self.results]
        
        if len(scores) < 2:
            return 0.0
        
        mean = np.mean(scores)
        std = np.std(scores)
        
        # Coefficient of variation (normalized std)
        if mean > 0:
            return float(std / mean)
        else:
            return float(std)
    
    def compute_outcome_variance(self) -> Dict[str, float]:
        """
        Compute variance in various outcome measures.
        
        Returns:
            Dict with variance statistics
        """
        if not self.results:
            return {}
        
        scores = [r.outcome_score for r in self.results]
        survival_times = [r.total_steps for r in self.results]
        
        return {
            "score_variance": float(np.var(scores)),
            "score_std": float(np.std(scores)),
            "score_range": max(scores) - min(scores),
            "survival_variance": float(np.var(survival_times)),
            "survival_std": float(np.std(survival_times)),
        }
    
    def compute_initial_condition_sensitivity(
        self,
        state_grouper: Optional[Callable[[Dict[str, Any]], str]] = None,
    ) -> float:
        """
        Compute sensitivity to initial conditions.
        
        Args:
            state_grouper: Function to group initial states
            
        Returns:
            Sensitivity score
        """
        if not self.results or len(self.results) < 2:
            return 0.0
        
        # Group by initial state characteristics
        if state_grouper is None:
            # Default: use first variable as grouper
            def state_grouper(s):
                vars = s.get("variables", {})
                if vars:
                    key = list(vars.keys())[0]
                    return str(vars[key])
                return "default"
        
        groups: Dict[str, List[float]] = {}
        for result in self.results:
            group = state_grouper(result.initial_state)
            if group not in groups:
                groups[group] = []
            groups[group].append(result.outcome_score)
        
        if len(groups) < 2:
            return 0.0
        
        # Compute between-group variance relative to within-group variance
        group_means = [np.mean(scores) for scores in groups.values()]
        overall_mean = np.mean([r.outcome_score for r in self.results])
        
        between_var = np.var(group_means)
        within_var = np.mean([np.var(scores) for scores in groups.values() if len(scores) > 1])
        
        if within_var > 0:
            return float(between_var / within_var)
        else:
            return float(between_var)
    
    def compute_brittleness_score(self) -> float:
        """
        Compute overall brittleness score (0-1).
        
        Combines multiple sensitivity measures into a single
        brittleness metric. Higher = more brittle/fragile.
        
        Returns:
            Brittleness score (0-1)
        """
        if not self.results:
            return 0.0
        
        # Components of brittleness
        noise_sens = self.compute_noise_sensitivity()
        
        # Failure rate contributes to brittleness
        failure_rate = sum(1 for r in self.results if r.is_failure()) / len(self.results)
        
        # Score variance
        scores = [r.outcome_score for r in self.results]
        score_range = max(scores) - min(scores) if scores else 0
        
        # Tail risk (fraction of very bad outcomes)
        if scores:
            threshold = np.percentile(scores, 10)
            tail_risk = sum(1 for s in scores if s <= threshold) / len(scores)
        else:
            tail_risk = 0
        
        # Combine into brittleness score
        # Weighted average, clamped to 0-1
        brittleness = (
            0.3 * min(1.0, noise_sens) +        # Noise sensitivity
            0.3 * failure_rate +                 # Failure rate
            0.2 * score_range +                  # Outcome spread
            0.2 * tail_risk                      # Tail risk
        )
        
        return min(1.0, max(0.0, brittleness))
    
    def compute_full_report(self) -> SensitivityReport:
        """
        Compute comprehensive sensitivity report.
        
        Returns:
            SensitivityReport with all metrics
        """
        noise_sens = self.compute_noise_sensitivity()
        ic_sens = self.compute_initial_condition_sensitivity()
        brittleness = self.compute_brittleness_score()
        variance_stats = self.compute_outcome_variance()
        
        return SensitivityReport(
            noise_sensitivity=noise_sens,
            parameter_sensitivity={},  # Requires parameter sweep
            initial_condition_sensitivity=ic_sens,
            brittleness_score=brittleness,
            stability_score=1.0 - brittleness,
            details=variance_stats,
        )
    
    def compare_sensitivity(
        self,
        other: "SensitivityAnalyzer",
    ) -> Dict[str, Any]:
        """
        Compare sensitivity between two policies.
        
        Args:
            other: Another SensitivityAnalyzer
            
        Returns:
            Comparison results
        """
        report1 = self.compute_full_report()
        report2 = other.compute_full_report()
        
        return {
            "brittleness_diff": report1.brittleness_score - report2.brittleness_score,
            "stability_diff": report1.stability_score - report2.stability_score,
            "noise_sensitivity_diff": report1.noise_sensitivity - report2.noise_sensitivity,
            "report1": report1,
            "report2": report2,
            "more_stable": "policy1" if report1.stability_score > report2.stability_score else "policy2",
        }
