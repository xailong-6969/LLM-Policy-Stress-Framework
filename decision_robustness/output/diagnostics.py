"""
Decision diagnostics and risk profiling.

Combines all metrics into comprehensive decision diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from decision_robustness.swarm.executor import SwarmResult
    from decision_robustness.metrics.survival import SurvivalCurve
    from decision_robustness.metrics.collapse import CollapseMetrics
    from decision_robustness.metrics.regret import RegretDistribution
    from decision_robustness.metrics.sensitivity import SensitivityReport


@dataclass
class RiskProfile:
    """
    Comprehensive risk profile for a policy.
    
    Attributes:
        failure_probability: Overall P(failure)
        tail_risk: Risk of catastrophic outcomes
        time_to_failure: Expected time before failure
        brittleness: How fragile the policy is
        stability_grade: Letter grade (A-F)
        key_risks: List of identified risk factors
        overall_risk_level: "low", "moderate", "high", "critical"
    """
    failure_probability: float
    tail_risk: float
    time_to_failure: Optional[float]
    brittleness: float
    stability_grade: str
    key_risks: List[str]
    overall_risk_level: str
    
    def describe(self) -> str:
        """Human-readable description."""
        lines = [
            f"═══════════════════════════════════════",
            f"           RISK PROFILE",
            f"═══════════════════════════════════════",
            f"  Overall Risk Level: {self.overall_risk_level.upper()}",
            f"  Stability Grade: {self.stability_grade}",
            f"",
            f"  Failure Probability: {self.failure_probability:.1%}",
            f"  Tail Risk (worst 10%): {self.tail_risk:.1%}",
            f"  Brittleness Score: {self.brittleness:.2f}",
        ]
        
        if self.time_to_failure:
            lines.append(f"  Mean Time to Failure: {self.time_to_failure:.1f} steps")
        
        if self.key_risks:
            lines.append(f"")
            lines.append(f"  Key Risks:")
            for risk in self.key_risks[:5]:
                lines.append(f"    ⚠ {risk}")
        
        lines.append(f"═══════════════════════════════════════")
        
        return "\n".join(lines)


@dataclass
class DecisionDiagnostics:
    """
    Complete decision diagnostics report.
    
    Combines survival, collapse, regret, and sensitivity analysis
    into a unified diagnostic report.
    """
    survival_curve: Optional["SurvivalCurve"]
    collapse_metrics: Optional["CollapseMetrics"]
    regret_distribution: Optional["RegretDistribution"]
    sensitivity_report: Optional["SensitivityReport"]
    risk_profile: Optional[RiskProfile]
    summary: Dict[str, Any]
    
    @classmethod
    def from_swarm_result(cls, swarm_result: "SwarmResult") -> "DecisionDiagnostics":
        """Create diagnostics from swarm execution result."""
        from decision_robustness.metrics.survival import SurvivalAnalyzer
        from decision_robustness.metrics.collapse import CollapseAnalyzer
        from decision_robustness.metrics.regret import RegretAnalyzer
        from decision_robustness.metrics.sensitivity import SensitivityAnalyzer
        
        results = swarm_result.results
        
        # Compute all metrics
        survival_analyzer = SurvivalAnalyzer(results)
        collapse_analyzer = CollapseAnalyzer(results)
        regret_analyzer = RegretAnalyzer(results)
        sensitivity_analyzer = SensitivityAnalyzer(results)
        
        survival_curve = survival_analyzer.compute_survival_curve()
        collapse_metrics = collapse_analyzer.compute_metrics()
        regret_dist = regret_analyzer.compute_outcome_regret()
        sensitivity_report = sensitivity_analyzer.compute_full_report()
        
        # Create risk profile
        risk_profile = cls._create_risk_profile(
            swarm_result, collapse_metrics, sensitivity_report
        )
        
        # Summary stats
        summary = {
            "total_runs": len(results),
            "success_rate": swarm_result.success_rate,
            "failure_rate": swarm_result.failure_rate,
            "timeout_rate": swarm_result.timeout_rate,
            "mean_score": sum(r.outcome_score for r in results) / len(results) if results else 0,
            "execution_time": swarm_result.total_time_seconds,
        }
        
        return cls(
            survival_curve=survival_curve,
            collapse_metrics=collapse_metrics,
            regret_distribution=regret_dist,
            sensitivity_report=sensitivity_report,
            risk_profile=risk_profile,
            summary=summary,
        )
    
    @classmethod
    def _create_risk_profile(
        cls,
        swarm_result: "SwarmResult",
        collapse_metrics: "CollapseMetrics",
        sensitivity_report: "SensitivityReport",
    ) -> RiskProfile:
        """Create risk profile from metrics."""
        failure_prob = swarm_result.failure_rate
        brittleness = sensitivity_report.brittleness_score
        
        # Compute tail risk
        scores = sorted(swarm_result.scores)
        n_tail = max(1, len(scores) // 10)
        tail_risk = sum(1 for s in scores[:n_tail] if s < 0.5) / max(1, n_tail)
        
        # Stability grade
        stability = sensitivity_report.stability_score
        if stability >= 0.8:
            grade = "A"
        elif stability >= 0.6:
            grade = "B"
        elif stability >= 0.4:
            grade = "C"
        elif stability >= 0.2:
            grade = "D"
        else:
            grade = "F"
        
        # Key risks
        risks = []
        if failure_prob > 0.5:
            risks.append(f"High failure rate ({failure_prob:.0%})")
        if brittleness > 0.6:
            risks.append(f"Brittle policy (score={brittleness:.2f})")
        if collapse_metrics.early_collapse_rate > 0.2:
            risks.append(f"Early collapse risk ({collapse_metrics.early_collapse_rate:.0%})")
        if sensitivity_report.noise_sensitivity > 0.5:
            risks.append(f"Noise sensitive (CV={sensitivity_report.noise_sensitivity:.2f})")
        
        # Overall risk level
        risk_score = (failure_prob + brittleness + tail_risk) / 3
        if risk_score >= 0.7:
            risk_level = "critical"
        elif risk_score >= 0.5:
            risk_level = "high"
        elif risk_score >= 0.3:
            risk_level = "moderate"
        else:
            risk_level = "low"
        
        return RiskProfile(
            failure_probability=failure_prob,
            tail_risk=tail_risk,
            time_to_failure=collapse_metrics.mean_time_to_collapse,
            brittleness=brittleness,
            stability_grade=grade,
            key_risks=risks,
            overall_risk_level=risk_level,
        )
    
    def describe(self) -> str:
        """Full diagnostic report as text."""
        sections = []
        
        # Risk profile
        if self.risk_profile:
            sections.append(self.risk_profile.describe())
        
        sections.append("")
        
        # Summary
        sections.append("SUMMARY")
        sections.append("-" * 40)
        sections.append(f"Total Runs: {self.summary.get('total_runs', 0)}")
        sections.append(f"Success Rate: {self.summary.get('success_rate', 0):.1%}")
        sections.append(f"Failure Rate: {self.summary.get('failure_rate', 0):.1%}")
        sections.append(f"Mean Score: {self.summary.get('mean_score', 0):.3f}")
        sections.append(f"Execution Time: {self.summary.get('execution_time', 0):.2f}s")
        sections.append("")
        
        # Survival
        if self.survival_curve:
            sections.append(self.survival_curve.describe())
            sections.append("")
        
        # Collapse
        if self.collapse_metrics:
            sections.append(self.collapse_metrics.describe())
            sections.append("")
        
        # Regret
        if self.regret_distribution:
            sections.append(self.regret_distribution.describe())
            sections.append("")
        
        # Sensitivity
        if self.sensitivity_report:
            sections.append(self.sensitivity_report.describe())
        
        return "\n".join(sections)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export as dictionary."""
        return {
            "summary": self.summary,
            "risk_profile": {
                "failure_probability": self.risk_profile.failure_probability,
                "tail_risk": self.risk_profile.tail_risk,
                "brittleness": self.risk_profile.brittleness,
                "stability_grade": self.risk_profile.stability_grade,
                "overall_risk_level": self.risk_profile.overall_risk_level,
                "key_risks": self.risk_profile.key_risks,
            } if self.risk_profile else None,
        }
