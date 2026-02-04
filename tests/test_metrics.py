"""
Tests for robustness metrics.
"""

import pytest
from decision_robustness.engine.simulator import SimulationResult
from decision_robustness.metrics.survival import SurvivalAnalyzer, SurvivalCurve
from decision_robustness.metrics.collapse import CollapseAnalyzer, CollapseMetrics
from decision_robustness.metrics.regret import RegretAnalyzer, RegretDistribution
from decision_robustness.metrics.sensitivity import SensitivityAnalyzer, SensitivityReport


def make_result(outcome, total_steps, score, seed=0, events=None):
    """Helper to create simulation results."""
    return SimulationResult(
        trajectory=[],
        initial_state={"variables": {}},
        final_state={"variables": {}},
        total_steps=total_steps,
        outcome=outcome,
        outcome_score=score,
        events_occurred=events or [],
        seed=seed,
    )


class TestSurvivalAnalyzer:
    """Tests for SurvivalAnalyzer."""
    
    def test_all_survive(self):
        results = [
            make_result("success", 10, 0.9, seed=i)
            for i in range(10)
        ]
        
        analyzer = SurvivalAnalyzer(results)
        curve = analyzer.compute_survival_curve()
        
        # All survived, so final survival should be 1.0
        assert curve.survival_prob[-1] == 1.0
    
    def test_all_fail(self):
        results = [
            make_result("failure", 5, 0.2, seed=i)
            for i in range(10)
        ]
        
        analyzer = SurvivalAnalyzer(results)
        curve = analyzer.compute_survival_curve()
        
        # All failed at step 5, so survival at step 5 should be 0
        assert curve.survival_prob[-1] == 0.0
    
    def test_mixed_outcomes(self):
        results = [
            make_result("success", 10, 0.9, seed=0),
            make_result("success", 10, 0.9, seed=1),
            make_result("failure", 5, 0.2, seed=2),
            make_result("failure", 5, 0.2, seed=3),
        ]
        
        analyzer = SurvivalAnalyzer(results)
        curve = analyzer.compute_survival_curve()
        
        # 2 out of 4 failed
        assert 0 < curve.survival_prob[-1] < 1
    
    def test_median_survival(self):
        # All fail at step 5
        results = [
            make_result("failure", 5, 0.2, seed=i)
            for i in range(10)
        ]
        
        analyzer = SurvivalAnalyzer(results)
        curve = analyzer.compute_survival_curve()
        
        median = curve.median_survival()
        assert median == 5


class TestCollapseAnalyzer:
    """Tests for CollapseAnalyzer."""
    
    def test_no_collapses(self):
        results = [
            make_result("success", 10, 0.9, seed=i)
            for i in range(10)
        ]
        
        analyzer = CollapseAnalyzer(results)
        metrics = analyzer.compute_metrics()
        
        assert metrics.collapse_probability == 0.0
        assert metrics.collapse_count == 0
    
    def test_all_collapse(self):
        results = [
            make_result("failure", 5, 0.2, seed=i)
            for i in range(10)
        ]
        
        analyzer = CollapseAnalyzer(results)
        metrics = analyzer.compute_metrics()
        
        assert metrics.collapse_probability == 1.0
        assert metrics.collapse_count == 10
        assert metrics.mean_time_to_collapse == 5.0
    
    def test_early_vs_late_collapse(self):
        # 5 early failures, 5 late failures
        early = [make_result("failure", 2, 0.1, seed=i) for i in range(5)]
        late = [make_result("failure", 9, 0.2, seed=i+5) for i in range(5)]
        
        analyzer = CollapseAnalyzer(early + late, max_steps=10)
        metrics = analyzer.compute_metrics()
        
        assert metrics.early_collapse_rate == 0.5  # 5 out of 10
        assert metrics.late_collapse_rate == 0.5   # 5 out of 10


class TestRegretAnalyzer:
    """Tests for RegretAnalyzer."""
    
    def test_no_regret(self):
        results = [
            make_result("success", 10, 1.0, seed=i)  # Perfect score
            for i in range(10)
        ]
        
        analyzer = RegretAnalyzer(results, optimal_score=1.0)
        dist = analyzer.compute_outcome_regret()
        
        assert dist.mean_regret == 0.0
        assert dist.max_regret == 0.0
    
    def test_full_regret(self):
        results = [
            make_result("failure", 5, 0.0, seed=i)  # Zero score
            for i in range(10)
        ]
        
        analyzer = RegretAnalyzer(results, optimal_score=1.0)
        dist = analyzer.compute_outcome_regret()
        
        assert dist.mean_regret == 1.0
        assert dist.max_regret == 1.0
    
    def test_mixed_regret(self):
        results = [
            make_result("success", 10, 0.8, seed=0),
            make_result("success", 10, 0.6, seed=1),
            make_result("failure", 5, 0.2, seed=2),
        ]
        
        analyzer = RegretAnalyzer(results, optimal_score=1.0)
        dist = analyzer.compute_outcome_regret()
        
        # Regrets: 0.2, 0.4, 0.8 => mean = 0.467
        assert 0.4 < dist.mean_regret < 0.5
        assert dist.max_regret == 0.8


class TestSensitivityAnalyzer:
    """Tests for SensitivityAnalyzer."""
    
    def test_low_sensitivity(self):
        # All same score = no variation
        results = [
            make_result("success", 10, 0.9, seed=i)
            for i in range(10)
        ]
        
        analyzer = SensitivityAnalyzer(results)
        noise_sens = analyzer.compute_noise_sensitivity()
        
        assert noise_sens == 0.0
    
    def test_high_sensitivity(self):
        # Wide range of scores
        results = []
        for i in range(10):
            score = i / 10  # 0.0 to 0.9
            results.append(make_result("success", 10, score, seed=i))
        
        analyzer = SensitivityAnalyzer(results)
        noise_sens = analyzer.compute_noise_sensitivity()
        
        assert noise_sens > 0.5  # High variation
    
    def test_brittleness_score(self):
        # Mix of successes and failures
        results = [
            make_result("success", 10, 0.9, seed=0),
            make_result("success", 10, 0.8, seed=1),
            make_result("failure", 5, 0.3, seed=2),
            make_result("failure", 3, 0.1, seed=3),
        ]
        
        analyzer = SensitivityAnalyzer(results)
        brittleness = analyzer.compute_brittleness_score()
        
        assert 0 <= brittleness <= 1
    
    def test_full_report(self):
        results = [
            make_result("success", 10, 0.9, seed=i)
            for i in range(10)
        ]
        
        analyzer = SensitivityAnalyzer(results)
        report = analyzer.compute_full_report()
        
        assert isinstance(report, SensitivityReport)
        assert 0 <= report.brittleness_score <= 1
        assert 0 <= report.stability_score <= 1
        assert report.brittleness_score + report.stability_score == 1.0
