#!/usr/bin/env python
"""
Project Simulator Demo

Demonstrates the Decision Robustness Framework by evaluating
different policies on a software project simulation.

Run this to see the framework in action!
"""

import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from decision_robustness.examples.project_simulator.world_config import ProjectWorld
from decision_robustness.examples.project_simulator.policies import (
    AggressivePolicy,
    ConservativePolicy,
    BalancedPolicy,
)
from decision_robustness.swarm.executor import SwarmExecutor, SwarmConfig
from decision_robustness.swarm.collector import OutcomeCollector
from decision_robustness.output.diagnostics import DecisionDiagnostics
from decision_robustness.output.reporter import Reporter, ReportFormat


def run_evaluation(policy, n_worlds: int = 100, show_progress: bool = True):
    """Run evaluation for a single policy."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {policy.name}")
    print(f"{'='*60}")
    
    # Create swarm executor
    executor = SwarmExecutor(
        world_factory=lambda seed: ProjectWorld(seed=seed),
        config=SwarmConfig(
            n_worlds=n_worlds,
            base_seed=42,
            show_progress=show_progress,
        ),
        simulator_kwargs={"max_steps": 50},
    )
    
    # Run evaluation
    swarm_result = executor.run(policy)
    
    # Generate diagnostics
    diagnostics = DecisionDiagnostics.from_swarm_result(swarm_result)
    
    return swarm_result, diagnostics


def compare_policies(n_worlds: int = 100):
    """Compare all three policies."""
    policies = [
        AggressivePolicy(),
        ConservativePolicy(),
        BalancedPolicy(),
    ]
    
    results = {}
    
    for policy in policies:
        swarm_result, diagnostics = run_evaluation(policy, n_worlds, show_progress=False)
        results[policy.name] = {
            "swarm_result": swarm_result,
            "diagnostics": diagnostics,
        }
        
        # Print risk profile
        if diagnostics.risk_profile:
            print(diagnostics.risk_profile.describe())
    
    # Comparison summary
    print("\n" + "="*60)
    print("POLICY COMPARISON SUMMARY")
    print("="*60)
    print(f"\n{'Policy':<20} {'Success':>10} {'Failure':>10} {'Risk Level':>12} {'Grade':>6}")
    print("-" * 60)
    
    for name, data in results.items():
        d = data["diagnostics"]
        rp = d.risk_profile
        print(f"{name:<20} {d.summary['success_rate']:>9.1%} {d.summary['failure_rate']:>9.1%} {rp.overall_risk_level:>12} {rp.stability_grade:>6}")
    
    return results


def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║          DECISION ROBUSTNESS FRAMEWORK DEMO                  ║
║                                                              ║
║  "Evaluating AI decisions by how many futures they survive" ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    print("This demo simulates a software project with three different")
    print("decision policies and analyzes their robustness.\n")
    
    print("Project variables: progress, debt, morale, budget, bugs")
    print("Available actions: SHIP_NOW, REFACTOR, HIRE, CUT_SCOPE, FIX_BUGS, DELAY")
    print("\nRunning 100 parallel simulations per policy...\n")
    
    # Run comparison
    results = compare_policies(n_worlds=100)
    
    # Show detailed report for best policy
    best_policy = max(
        results.items(),
        key=lambda x: x[1]["diagnostics"].summary["success_rate"]
    )
    
    print(f"\n\n{'='*60}")
    print(f"DETAILED REPORT: {best_policy[0]}")
    print(f"{'='*60}")
    
    diagnostics = best_policy[1]["diagnostics"]
    print(diagnostics.describe())
    
    print("\n\nDone! The Decision Robustness Framework provides:")
    print("  • Survival curves (probability of surviving over time)")
    print("  • Collapse probability (failure rates at different horizons)")  
    print("  • Regret distribution (decision quality analysis)")
    print("  • Sensitivity analysis (brittleness scoring)")
    print("\nThis is the way to evaluate AI decision policies properly.")


if __name__ == "__main__":
    main()
