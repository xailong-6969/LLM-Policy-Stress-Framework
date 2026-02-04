"""
CLI entry point for Decision Robustness Framework.
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Decision Robustness Framework - Evaluate AI decision policies"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run the project simulator demo")
    demo_parser.add_argument(
        "-n", "--n-worlds",
        type=int,
        default=100,
        help="Number of parallel worlds to simulate (default: 100)"
    )
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a policy")
    eval_parser.add_argument(
        "--policy",
        choices=["aggressive", "conservative", "balanced"],
        default="balanced",
        help="Policy to evaluate"
    )
    eval_parser.add_argument(
        "-n", "--n-worlds",
        type=int,
        default=100,
        help="Number of parallel worlds"
    )
    eval_parser.add_argument(
        "-o", "--output",
        help="Output file for report (format inferred from extension)"
    )
    
    args = parser.parse_args()
    
    if args.command == "demo":
        from decision_robustness.examples.project_simulator.run_evaluation import main as demo_main
        demo_main()
    
    elif args.command == "evaluate":
        run_evaluation(args)
    
    else:
        parser.print_help()


def run_evaluation(args):
    """Run policy evaluation from CLI."""
    from decision_robustness.examples.project_simulator.world_config import ProjectWorld
    from decision_robustness.examples.project_simulator.policies import (
        AggressivePolicy,
        ConservativePolicy,
        BalancedPolicy,
    )
    from decision_robustness.swarm.executor import SwarmExecutor, SwarmConfig
    from decision_robustness.output.diagnostics import DecisionDiagnostics
    from decision_robustness.output.reporter import Reporter
    
    # Select policy
    policies = {
        "aggressive": AggressivePolicy,
        "conservative": ConservativePolicy,
        "balanced": BalancedPolicy,
    }
    policy = policies[args.policy]()
    
    print(f"Evaluating {policy.name} across {args.n_worlds} worlds...")
    
    # Run evaluation
    executor = SwarmExecutor(
        world_factory=lambda seed: ProjectWorld(seed=seed),
        config=SwarmConfig(
            n_worlds=args.n_worlds,
            base_seed=42,
            show_progress=True,
        ),
        simulator_kwargs={"max_steps": 50},
    )
    
    result = executor.run(policy)
    diagnostics = DecisionDiagnostics.from_swarm_result(result)
    
    # Output
    if args.output:
        reporter = Reporter(diagnostics)
        reporter.save(args.output)
        print(f"\nReport saved to: {args.output}")
    else:
        print(diagnostics.describe())


if __name__ == "__main__":
    main()
