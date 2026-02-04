"""
Tests for swarm execution.
"""

import pytest
from decision_robustness.engine.world import World, WorldState
from decision_robustness.policies.base import Policy, Action, DecisionContext
from decision_robustness.swarm.executor import SwarmExecutor, SwarmConfig, SwarmResult
from decision_robustness.swarm.collector import OutcomeCollector, OutcomeDistribution


class DeterministicWorld(World):
    """World with deterministic outcomes for testing."""
    
    def __init__(self, seed=None, fail_at=None):
        super().__init__(seed)
        self.fail_at = fail_at
    
    def initial_state(self):
        return WorldState(
            timestep=0,
            variables={"counter": 0},
        )
    
    def step(self, state, action):
        return state.evolve(
            timestep=state.timestep + 1,
            counter=state.get("counter", 0) + 1,
        )
    
    def get_available_actions(self, state):
        return [Action(name="step")]
    
    def check_terminal(self, state):
        if self.fail_at and state.timestep >= self.fail_at:
            return state.evolve(is_terminal=True, terminal_reason="failure")
        if state.timestep >= 10:
            return state.evolve(is_terminal=True, terminal_reason="success")
        return state


class SimplePolicy(Policy):
    def decide(self, context):
        return context.available_actions[0]


class TestSwarmConfig:
    """Tests for SwarmConfig."""
    
    def test_default_config(self):
        config = SwarmConfig()
        
        assert config.n_worlds == 100
        assert config.executor_type == "thread"
    
    def test_validation(self):
        with pytest.raises(ValueError):
            SwarmConfig(n_worlds=0)
        
        with pytest.raises(ValueError):
            SwarmConfig(executor_type="invalid")


class TestSwarmExecutor:
    """Tests for SwarmExecutor."""
    
    def test_run_sequential(self):
        executor = SwarmExecutor(
            world_factory=lambda seed: DeterministicWorld(seed),
            config=SwarmConfig(n_worlds=10, show_progress=False),
        )
        
        result = executor.run_sequential(SimplePolicy())
        
        assert len(result.results) == 10
        assert result.successful_runs == 10
        assert result.failed_runs == 0
    
    def test_run_parallel(self):
        executor = SwarmExecutor(
            world_factory=lambda seed: DeterministicWorld(seed),
            config=SwarmConfig(n_worlds=10, show_progress=False),
        )
        
        result = executor.run(SimplePolicy())
        
        assert len(result.results) == 10
    
    def test_all_success(self):
        executor = SwarmExecutor(
            world_factory=lambda seed: DeterministicWorld(seed),
            config=SwarmConfig(n_worlds=10, show_progress=False),
        )
        
        result = executor.run_sequential(SimplePolicy())
        
        assert result.success_rate == 1.0
        assert result.failure_rate == 0.0
    
    def test_all_failure(self):
        executor = SwarmExecutor(
            world_factory=lambda seed: DeterministicWorld(seed, fail_at=5),
            config=SwarmConfig(n_worlds=10, show_progress=False),
        )
        
        result = executor.run_sequential(SimplePolicy())
        
        assert result.success_rate == 0.0
        assert result.failure_rate == 1.0


class TestSwarmResult:
    """Tests for SwarmResult."""
    
    def test_outcome_accessors(self):
        # Create mock results
        from decision_robustness.engine.simulator import SimulationResult
        
        results = [
            SimulationResult(
                trajectory=[],
                initial_state={},
                final_state={},
                total_steps=10,
                outcome="success",
                outcome_score=0.9,
                events_occurred=[],
                seed=i,
            )
            for i in range(10)
        ]
        
        swarm_result = SwarmResult(
            results=results,
            config=SwarmConfig(n_worlds=10),
            total_time_seconds=1.0,
        )
        
        assert len(swarm_result.outcomes) == 10
        assert len(swarm_result.scores) == 10
        assert swarm_result.success_rate == 1.0


class TestOutcomeDistribution:
    """Tests for OutcomeDistribution."""
    
    def test_from_values(self):
        values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        dist = OutcomeDistribution.from_values(values)
        
        assert dist.mean == pytest.approx(0.55, rel=0.01)
        assert dist.min_val == 0.1
        assert dist.max_val == 1.0
    
    def test_empty_values(self):
        dist = OutcomeDistribution.from_values([])
        
        assert dist.mean == 0
        assert dist.std == 0


class TestOutcomeCollector:
    """Tests for OutcomeCollector."""
    
    def test_outcome_summary(self):
        # Run a quick evaluation
        executor = SwarmExecutor(
            world_factory=lambda seed: DeterministicWorld(seed),
            config=SwarmConfig(n_worlds=20, show_progress=False),
        )
        
        swarm_result = executor.run_sequential(SimplePolicy())
        collector = OutcomeCollector(swarm_result)
        
        summary = collector.get_outcome_summary()
        
        assert summary.total == 20
        assert summary.success_count == 20
    
    def test_score_distribution(self):
        executor = SwarmExecutor(
            world_factory=lambda seed: DeterministicWorld(seed),
            config=SwarmConfig(n_worlds=20, show_progress=False),
        )
        
        swarm_result = executor.run_sequential(SimplePolicy())
        collector = OutcomeCollector(swarm_result)
        
        dist = collector.get_score_distribution()
        
        assert dist.mean > 0
        assert len(dist.values) == 20
