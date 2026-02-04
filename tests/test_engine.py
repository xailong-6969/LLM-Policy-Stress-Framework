"""
Tests for the World simulation engine.
"""

import pytest
from decision_robustness.engine.world import World, WorldState
from decision_robustness.engine.events import Event, SimpleEventGenerator
from decision_robustness.engine.simulator import Simulator
from decision_robustness.policies.base import Policy, Action, DecisionContext


class SimpleWorld(World):
    """Simple test world."""
    
    def initial_state(self):
        return WorldState(
            timestep=0,
            variables={"value": 50},
        )
    
    def step(self, state, action):
        value = state.get("value", 50)
        
        if action.name == "increase":
            value += 10
        elif action.name == "decrease":
            value -= 10
        
        return state.evolve(
            timestep=state.timestep + 1,
            value=value,
        )
    
    def get_available_actions(self, state):
        return [
            Action(name="increase"),
            Action(name="decrease"),
            Action(name="hold"),
        ]
    
    def check_terminal(self, state):
        value = state.get("value", 50)
        
        if value >= 100:
            return state.evolve(is_terminal=True, terminal_reason="success")
        elif value <= 0:
            return state.evolve(is_terminal=True, terminal_reason="failure")
        
        return state


class AlwaysIncreasePolicy(Policy):
    """Test policy that always increases."""
    
    def decide(self, context):
        return context.get_action_by_name("increase")


class AlwaysDecreasePolicy(Policy):
    """Test policy that always decreases."""
    
    def decide(self, context):
        return context.get_action_by_name("decrease")


class TestWorldState:
    """Tests for WorldState."""
    
    def test_create_state(self):
        state = WorldState(
            timestep=0,
            variables={"a": 1, "b": 2},
        )
        assert state.timestep == 0
        assert state.get("a") == 1
        assert state.get("b") == 2
    
    def test_state_immutability(self):
        state = WorldState(
            timestep=0,
            variables={"value": 10},
        )
        
        # evolve should create new state
        new_state = state.evolve(value=20)
        
        assert state.get("value") == 10
        assert new_state.get("value") == 20
    
    def test_terminal_states(self):
        success = WorldState(
            timestep=10,
            variables={},
            is_terminal=True,
            terminal_reason="success",
        )
        
        failure = WorldState(
            timestep=5,
            variables={},
            is_terminal=True,
            terminal_reason="failure",
        )
        
        assert success.is_success()
        assert not success.is_failed()
        assert failure.is_failed()
        assert not failure.is_success()


class TestWorld:
    """Tests for World class."""
    
    def test_initial_state(self):
        world = SimpleWorld(seed=42)
        state = world.initial_state()
        
        assert state.timestep == 0
        assert state.get("value") == 50
    
    def test_step(self):
        world = SimpleWorld(seed=42)
        state = world.initial_state()
        
        action = Action(name="increase")
        new_state = world.step(state, action)
        
        assert new_state.timestep == 1
        assert new_state.get("value") == 60
    
    def test_terminal_check(self):
        world = SimpleWorld(seed=42)
        
        # Success case
        high_state = WorldState(
            timestep=5,
            variables={"value": 100},
        )
        checked = world.check_terminal(high_state)
        assert checked.is_terminal
        assert checked.terminal_reason == "success"
        
        # Failure case
        low_state = WorldState(
            timestep=5,
            variables={"value": 0},
        )
        checked = world.check_terminal(low_state)
        assert checked.is_terminal
        assert checked.terminal_reason == "failure"


class TestSimulator:
    """Tests for Simulator class."""
    
    def test_run_to_success(self):
        world = SimpleWorld(seed=42)
        policy = AlwaysIncreasePolicy()
        simulator = Simulator(world, max_steps=20)
        
        result = simulator.run(policy, seed=42)
        
        assert result.outcome == "success"
        assert result.outcome_score > 0.5
        assert result.total_steps <= 6  # Should succeed in 5 steps (50 + 5*10 = 100)
    
    def test_run_to_failure(self):
        world = SimpleWorld(seed=42)
        policy = AlwaysDecreasePolicy()
        simulator = Simulator(world, max_steps=20)
        
        result = simulator.run(policy, seed=42)
        
        assert result.outcome == "failure"
        assert result.total_steps <= 6  # Should fail in 5 steps (50 - 5*10 = 0)
    
    def test_trajectory_recorded(self):
        world = SimpleWorld(seed=42)
        policy = AlwaysIncreasePolicy()
        simulator = Simulator(world, max_steps=20)
        
        result = simulator.run(policy, seed=42)
        
        assert len(result.trajectory) > 0
        assert result.trajectory[0]["timestep"] == 0


class TestEvents:
    """Tests for event system."""
    
    def test_event_creation(self):
        event = Event(
            name="test_event",
            description="A test event",
            probability=0.5,
            severity=0.3,
        )
        
        assert event.name == "test_event"
        assert event.probability == 0.5
    
    def test_event_probability_validation(self):
        with pytest.raises(ValueError):
            Event(name="bad", description="", probability=1.5)
        
        with pytest.raises(ValueError):
            Event(name="bad", description="", probability=-0.1)
    
    def test_simple_event_generator(self):
        events = [
            Event(name="event1", description="", probability=1.0),  # Always happens
            Event(name="event2", description="", probability=0.0),  # Never happens
        ]
        
        generator = SimpleEventGenerator(events=events)
        
        import numpy as np
        rng = np.random.default_rng(42)
        state = WorldState(timestep=0, variables={})
        
        occurred = generator.sample_events(state, rng)
        
        assert any(e.name == "event1" for e in occurred)
        assert not any(e.name == "event2" for e in occurred)
