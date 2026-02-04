"""
Simulation orchestrator.

The Simulator runs a policy through a world, recording the full trajectory
of states, actions, and events.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING, TypeVar, Generic
import copy

if TYPE_CHECKING:
    from decision_robustness.engine.world import World, WorldState
    from decision_robustness.policies.base import Policy


@dataclass
class ActionRecord:
    """Records an action taken at a specific timestep."""
    timestep: int
    action: Any
    available_actions: List[Any]
    decision_time_ms: Optional[float] = None
    decision_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EventRecord:
    """Records events that occurred at a specific timestep."""
    timestep: int
    events: List[Dict[str, Any]]


@dataclass
class SimulationResult:
    """
    Complete record of a simulation run.
    
    Contains the full trajectory including all states, actions, and events,
    along with summary statistics.
    
    Attributes:
        trajectory: List of (state_dict, action, events) tuples for each step
        initial_state: Starting state snapshot
        final_state: Ending state snapshot
        total_steps: Number of steps taken
        outcome: Final outcome ("success", "failure", "timeout", None)
        outcome_score: Numeric score (0-1)
        events_occurred: List of all events that occurred
        seed: Random seed used for this run
    """
    trajectory: List[Dict[str, Any]]
    initial_state: Dict[str, Any]
    final_state: Dict[str, Any]
    total_steps: int
    outcome: Optional[str]
    outcome_score: float
    events_occurred: List[Dict[str, Any]]
    seed: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_success(self) -> bool:
        return self.outcome == "success"
    
    def is_failure(self) -> bool:
        return self.outcome == "failure"
    
    def is_timeout(self) -> bool:
        return self.outcome == "timeout"
    
    @property
    def survival_time(self) -> int:
        """Return how many steps before failure (or total if survived)."""
        return self.total_steps
    
    def get_state_at(self, timestep: int) -> Optional[Dict[str, Any]]:
        """Get state at specific timestep."""
        if 0 <= timestep < len(self.trajectory):
            return self.trajectory[timestep].get("state")
        return None


class Simulator:
    """
    Runs a policy through a world, recording the full trajectory.
    
    The Simulator orchestrates the interaction between a World and a Policy,
    handling the main simulation loop:
    
    1. Get available actions from world
    2. Ask policy to decide
    3. Apply action to world (step)
    4. Apply stochastic events
    5. Check terminal conditions
    6. Repeat until terminal or max_steps
    """
    
    def __init__(
        self,
        world: "World",
        max_steps: int = 100,
        record_full_trajectory: bool = True,
    ):
        """
        Initialize the simulator.
        
        Args:
            world: The world to simulate
            max_steps: Maximum steps before timeout
            record_full_trajectory: Whether to store full state at each step
        """
        self.world = world
        self.max_steps = max_steps
        self.record_full_trajectory = record_full_trajectory
    
    def run(
        self,
        policy: "Policy",
        seed: Optional[int] = None,
    ) -> SimulationResult:
        """
        Run a single simulation.
        
        Args:
            policy: Decision policy to evaluate
            seed: Random seed for reproducibility
            
        Returns:
            SimulationResult containing full trajectory and outcome
        """
        # Initialize
        self.world.reset(seed)
        state = self.world.initial_state()
        initial_state = self._state_to_dict(state)
        
        trajectory = []
        all_events = []
        
        # Main simulation loop
        for step in range(self.max_steps):
            if state.is_terminal:
                break
            
            # Get available actions
            available_actions = self.world.get_available_actions(state)
            
            if not available_actions:
                # No actions available - treat as terminal
                break
            
            # Get policy decision
            from decision_robustness.policies.base import DecisionContext
            context = DecisionContext(
                state=state,
                available_actions=available_actions,
                timestep=state.timestep,
            )
            
            action = policy.decide(context)
            
            # Record pre-action state
            step_record = {
                "timestep": state.timestep,
                "state": self._state_to_dict(state) if self.record_full_trajectory else None,
                "action": self._action_to_dict(action),
                "available_actions": [self._action_to_dict(a) for a in available_actions],
            }
            
            # Apply action
            state = self.world.step(state, action)
            
            # Apply stochastic events
            state_before_events = state
            state = self.world.apply_events(state)
            
            # Record events
            events_this_step = self._get_new_events(state_before_events, state)
            step_record["events"] = events_this_step
            all_events.extend(events_this_step)
            
            # Check terminal conditions
            state = self.world.check_terminal(state)
            
            # Record post-step state
            step_record["state_after"] = self._state_to_dict(state) if self.record_full_trajectory else None
            
            trajectory.append(step_record)
        
        # Determine outcome
        if state.is_terminal:
            outcome = state.terminal_reason
        else:
            outcome = "timeout"
            # Mark as terminal due to timeout
            state = state.evolve(
                is_terminal=True,
                terminal_reason="timeout"
            )
        
        # Calculate outcome score
        outcome_score = self.world.get_outcome_score(state)
        
        return SimulationResult(
            trajectory=trajectory,
            initial_state=initial_state,
            final_state=self._state_to_dict(state),
            total_steps=len(trajectory),
            outcome=outcome,
            outcome_score=outcome_score,
            events_occurred=all_events,
            seed=seed,
            metadata={
                "max_steps": self.max_steps,
            }
        )
    
    def _state_to_dict(self, state: "WorldState") -> Dict[str, Any]:
        """Convert state to serializable dict."""
        return {
            "timestep": state.timestep,
            "variables": dict(state.variables),
            "is_terminal": state.is_terminal,
            "terminal_reason": state.terminal_reason,
            "metadata": dict(state.metadata),
        }
    
    def _action_to_dict(self, action: Any) -> Any:
        """Convert action to serializable form."""
        if hasattr(action, "to_dict"):
            return action.to_dict()
        elif hasattr(action, "__dict__"):
            return {"type": type(action).__name__, **action.__dict__}
        else:
            return action
    
    def _get_new_events(
        self,
        state_before: "WorldState",
        state_after: "WorldState"
    ) -> List[Dict[str, Any]]:
        """Extract events that occurred between two states."""
        before_events = state_before.metadata.get("events_occurred", [])
        after_events = state_after.metadata.get("events_occurred", [])
        
        # Return only new events
        return after_events[len(before_events):]
