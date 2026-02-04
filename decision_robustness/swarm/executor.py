"""
Parallel world executor for swarmed evaluation.

Runs the same policy across many parallel worlds with different
random seeds to generate outcome distributions.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type, TYPE_CHECKING
import time

if TYPE_CHECKING:
    from decision_robustness.engine.world import World
    from decision_robustness.engine.simulator import Simulator, SimulationResult
    from decision_robustness.policies.base import Policy


@dataclass
class SwarmConfig:
    """
    Configuration for swarm execution.
    
    Attributes:
        n_worlds: Number of parallel worlds to simulate
        max_workers: Maximum parallel workers (None = CPU count)
        executor_type: "thread" or "process"
        base_seed: Starting seed (subsequent worlds use base_seed + i)
        timeout_seconds: Per-world timeout
        show_progress: Whether to show progress updates
    """
    n_worlds: int = 100
    max_workers: Optional[int] = None
    executor_type: str = "thread"
    base_seed: int = 42
    timeout_seconds: Optional[float] = None
    show_progress: bool = True
    
    def __post_init__(self):
        if self.n_worlds < 1:
            raise ValueError("n_worlds must be at least 1")
        if self.executor_type not in ["thread", "process"]:
            raise ValueError("executor_type must be 'thread' or 'process'")


@dataclass
class SwarmResult:
    """
    Result of running a swarm evaluation.
    
    Attributes:
        results: List of individual simulation results
        config: Configuration used
        total_time_seconds: Total execution time
        successful_runs: Number of runs that completed
        failed_runs: Number of runs that failed/errored
    """
    results: List["SimulationResult"]
    config: SwarmConfig
    total_time_seconds: float
    successful_runs: int = 0
    failed_runs: int = 0
    errors: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.successful_runs:
            self.successful_runs = len(self.results)
    
    @property
    def outcomes(self) -> List[str]:
        """List of all outcomes (success/failure/timeout)."""
        return [r.outcome for r in self.results]
    
    @property
    def scores(self) -> List[float]:
        """List of all outcome scores."""
        return [r.outcome_score for r in self.results]
    
    @property
    def survival_times(self) -> List[int]:
        """List of survival times (steps before failure)."""
        return [r.survival_time for r in self.results]
    
    @property
    def success_rate(self) -> float:
        """Proportion of successful outcomes."""
        if not self.results:
            return 0.0
        successes = sum(1 for r in self.results if r.is_success())
        return successes / len(self.results)
    
    @property
    def failure_rate(self) -> float:
        """Proportion of failed outcomes."""
        if not self.results:
            return 0.0
        failures = sum(1 for r in self.results if r.is_failure())
        return failures / len(self.results)
    
    @property
    def timeout_rate(self) -> float:
        """Proportion of timeout outcomes."""
        if not self.results:
            return 0.0
        timeouts = sum(1 for r in self.results if r.is_timeout())
        return timeouts / len(self.results)
    
    def filter_by_outcome(self, outcome: str) -> List["SimulationResult"]:
        """Get results with specific outcome."""
        return [r for r in self.results if r.outcome == outcome]


class SwarmExecutor:
    """
    Executes policy across many parallel worlds.
    
    This is the core of distribution-based evaluation. Instead of
    running a policy once and checking the outcome, we run it
    across hundreds or thousands of parallel worlds and analyze
    the outcome distribution.
    
    Example:
        executor = SwarmExecutor(
            world_factory=lambda seed: MyWorld(seed),
            config=SwarmConfig(n_worlds=1000),
        )
        swarm_result = executor.run(my_policy)
        
        print(f"Success rate: {swarm_result.success_rate:.1%}")
        print(f"Failure rate: {swarm_result.failure_rate:.1%}")
    """
    
    def __init__(
        self,
        world_factory: Callable[[int], "World"],
        config: Optional[SwarmConfig] = None,
        simulator_kwargs: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ):
        """
        Initialize swarm executor.
        
        Args:
            world_factory: Function that takes a seed and returns a World
            config: Swarm configuration
            simulator_kwargs: Additional kwargs for Simulator
            progress_callback: Called with (completed, total) during execution
        """
        self.world_factory = world_factory
        self.config = config or SwarmConfig()
        self.simulator_kwargs = simulator_kwargs or {}
        self.progress_callback = progress_callback
    
    def _run_single_world(
        self,
        policy: "Policy",
        seed: int,
    ) -> "SimulationResult":
        """Run a single world simulation."""
        from decision_robustness.engine.simulator import Simulator
        
        # Create fresh world for this seed
        world = self.world_factory(seed)
        
        # Create simulator
        simulator = Simulator(world=world, **self.simulator_kwargs)
        
        # Run simulation
        result = simulator.run(policy=policy, seed=seed)
        
        return result
    
    def run(self, policy: "Policy") -> SwarmResult:
        """
        Run the policy across all parallel worlds.
        
        Args:
            policy: The decision policy to evaluate
            
        Returns:
            SwarmResult containing all outcomes
        """
        start_time = time.time()
        results: List["SimulationResult"] = []
        errors: List[Dict[str, Any]] = []
        
        # Generate seeds
        seeds = [self.config.base_seed + i for i in range(self.config.n_worlds)]
        
        # Choose executor type
        if self.config.executor_type == "process":
            ExecutorClass = ProcessPoolExecutor
        else:
            ExecutorClass = ThreadPoolExecutor
        
        completed = 0
        
        with ExecutorClass(max_workers=self.config.max_workers) as executor:
            # Submit all tasks
            future_to_seed = {
                executor.submit(self._run_single_world, policy, seed): seed
                for seed in seeds
            }
            
            # Collect results
            for future in as_completed(future_to_seed):
                seed = future_to_seed[future]
                completed += 1
                
                try:
                    result = future.result(timeout=self.config.timeout_seconds)
                    results.append(result)
                except Exception as e:
                    errors.append({
                        "seed": seed,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    })
                
                # Progress callback
                if self.progress_callback:
                    self.progress_callback(completed, self.config.n_worlds)
                
                # Simple progress output
                if self.config.show_progress and completed % max(1, self.config.n_worlds // 10) == 0:
                    pct = completed / self.config.n_worlds * 100
                    print(f"Progress: {completed}/{self.config.n_worlds} ({pct:.0f}%)")
        
        total_time = time.time() - start_time
        
        return SwarmResult(
            results=results,
            config=self.config,
            total_time_seconds=total_time,
            successful_runs=len(results),
            failed_runs=len(errors),
            errors=errors,
        )
    
    def run_sequential(self, policy: "Policy") -> SwarmResult:
        """
        Run worlds sequentially (for debugging).
        
        Args:
            policy: The decision policy to evaluate
            
        Returns:
            SwarmResult containing all outcomes
        """
        start_time = time.time()
        results: List["SimulationResult"] = []
        errors: List[Dict[str, Any]] = []
        
        for i in range(self.config.n_worlds):
            seed = self.config.base_seed + i
            
            try:
                result = self._run_single_world(policy, seed)
                results.append(result)
            except Exception as e:
                errors.append({
                    "seed": seed,
                    "error": str(e),
                    "error_type": type(e).__name__,
                })
            
            if self.progress_callback:
                self.progress_callback(i + 1, self.config.n_worlds)
        
        total_time = time.time() - start_time
        
        return SwarmResult(
            results=results,
            config=self.config,
            total_time_seconds=total_time,
            successful_runs=len(results),
            failed_runs=len(errors),
            errors=errors,
        )
