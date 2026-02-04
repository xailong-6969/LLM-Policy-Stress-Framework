"""
World state machine and state management.

The World class provides a deterministic state machine that advances through
discrete time steps. Each step, the world state evolves based on actions taken
and stochastic events that occur.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypeVar, Generic
import copy


@dataclass(frozen=True)
class WorldState:
    """
    Immutable snapshot of the world at a point in time.
    
    This is the core data structure that policies observe and make decisions on.
    States are immutable (frozen) to ensure reproducibility and safety.
    
    Attributes:
        timestep: Current time step (0-indexed)
        variables: Dictionary of state variables (any domain-specific data)
        is_terminal: Whether this is a terminal state (success, failure, or timeout)
        terminal_reason: If terminal, why (e.g., "success", "failure", "timeout")
        metadata: Additional metadata for tracking
    """
    timestep: int
    variables: Dict[str, Any]
    is_terminal: bool = False
    terminal_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a variable from state."""
        return self.variables.get(key, default)
    
    def is_failed(self) -> bool:
        """Check if state represents a failure."""
        return self.is_terminal and self.terminal_reason == "failure"
    
    def is_success(self) -> bool:
        """Check if state represents a success."""
        return self.is_terminal and self.terminal_reason == "success"
    
    def evolve(self, **updates) -> WorldState:
        """
        Create a new state with updated variables.
        
        Returns a new WordState with the specified updates applied.
        The original state remains unchanged (immutability).
        """
        new_variables = dict(self.variables)
        
        # Separate special fields from variable updates
        special_fields = {}
        for key in ["timestep", "is_terminal", "terminal_reason", "metadata"]:
            if key in updates:
                special_fields[key] = updates.pop(key)
        
        # Apply variable updates
        new_variables.update(updates)
        
        return WorldState(
            timestep=special_fields.get("timestep", self.timestep),
            variables=new_variables,
            is_terminal=special_fields.get("is_terminal", self.is_terminal),
            terminal_reason=special_fields.get("terminal_reason", self.terminal_reason),
            metadata=special_fields.get("metadata", dict(self.metadata)),
        )


StateT = TypeVar("StateT", bound=WorldState)


class World(ABC, Generic[StateT]):
    """
    Abstract base class for world simulators.
    
    A World defines:
    - How to create an initial state
    - How the state evolves given an action
    - What events can occur (stochastically)
    - When the world reaches terminal states
    
    Subclass this to create domain-specific simulators.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize world with optional random seed.
        
        Args:
            seed: Random seed for reproducibility
        """
        self._seed = seed
        self._rng = self._create_rng(seed)
    
    def _create_rng(self, seed: Optional[int] = None):
        """Create a random number generator."""
        import numpy as np
        return np.random.default_rng(seed)
    
    def reset(self, seed: Optional[int] = None) -> None:
        """Reset the RNG with a new seed."""
        if seed is not None:
            self._seed = seed
        self._rng = self._create_rng(self._seed)
    
    @property
    def rng(self):
        """Access the random number generator."""
        return self._rng
    
    @abstractmethod
    def initial_state(self) -> StateT:
        """
        Create and return the initial world state.
        
        Returns:
            The starting state for a simulation run.
        """
        pass
    
    @abstractmethod
    def step(self, state: StateT, action: Any) -> StateT:
        """
        Advance the world by one timestep given an action.
        
        This is the deterministic core of the world. Given the same state
        and action, this should always produce the same result (before
        stochastic events are applied).
        
        Args:
            state: Current world state
            action: Action taken by the policy
            
        Returns:
            New world state after the action
        """
        pass
    
    @abstractmethod
    def get_available_actions(self, state: StateT) -> List[Any]:
        """
        Get list of valid actions for the current state.
        
        Args:
            state: Current world state
            
        Returns:
            List of valid actions the policy can choose from
        """
        pass
    
    def apply_events(self, state: StateT) -> StateT:
        """
        Apply stochastic events to the state.
        
        Override this to add random events (team member quits, bugs discovered,
        scope creep, etc.). By default, no events are applied.
        
        Args:
            state: Current world state
            
        Returns:
            New state after events are applied
        """
        return state
    
    def check_terminal(self, state: StateT) -> StateT:
        """
        Check if state is terminal and update accordingly.
        
        Override this to define success/failure conditions.
        By default, no terminal conditions are checked.
        
        Args:
            state: Current world state
            
        Returns:
            State with terminal flag set if applicable
        """
        return state
    
    def get_outcome_score(self, state: StateT) -> float:
        """
        Calculate an outcome score for the final state.
        
        Higher scores are better. Override this to define domain-specific
        success metrics.
        
        Args:
            state: Final world state
            
        Returns:
            Numeric score representing outcome quality
        """
        if state.is_success():
            return 1.0
        elif state.is_failed():
            return 0.0
        else:
            return 0.5  # Neutral for non-terminal states
    
    def describe_state(self, state: StateT) -> str:
        """
        Return a human-readable description of the state.
        
        Args:
            state: World state to describe
            
        Returns:
            String description
        """
        return f"Step {state.timestep}: {state.variables}"
