"""
Policy interface - Abstract base class for decision policies.

A Policy maps world states to actions. This is the core interface
that allows plugging in any decision source: rule-based, LLM-backed,
RL-trained, or human.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from decision_robustness.engine.world import WorldState


class ActionType(Enum):
    """Common action type categories."""
    PROCEED = auto()      # Continue current path
    DELAY = auto()        # Wait/defer
    AGGRESSIVE = auto()   # Push harder
    CONSERVATIVE = auto() # Pull back
    INVEST = auto()       # Spend resources
    CUT = auto()          # Reduce scope/resources
    PIVOT = auto()        # Change direction


@dataclass
class Action:
    """
    Represents an action a policy can take.
    
    Attributes:
        name: Unique identifier for the action
        action_type: Category of action
        description: Human-readable description
        parameters: Additional action-specific parameters
        cost: Resource cost of this action (if applicable)
        risk_level: Risk level 0-1 (higher = riskier)
    """
    name: str
    action_type: ActionType = ActionType.PROCEED
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    cost: float = 0.0
    risk_level: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "action_type": self.action_type.name,
            "description": self.description,
            "parameters": self.parameters,
            "cost": self.cost,
            "risk_level": self.risk_level,
        }
    
    def __repr__(self) -> str:
        return f"Action({self.name})"
    
    def __eq__(self, other) -> bool:
        if isinstance(other, Action):
            return self.name == other.name
        return False
    
    def __hash__(self) -> int:
        return hash(self.name)


@dataclass
class DecisionContext:
    """
    Context provided to policy for making a decision.
    
    Attributes:
        state: Current world state
        available_actions: List of valid actions to choose from
        timestep: Current timestep
        history: Optional list of previous (state, action) pairs
        metadata: Additional context information
    """
    state: "WorldState"
    available_actions: List[Action]
    timestep: int
    history: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_action_by_name(self, name: str) -> Optional[Action]:
        """Get an action by its name."""
        for action in self.available_actions:
            if action.name == name:
                return action
        return None
    
    @property
    def state_variables(self) -> Dict[str, Any]:
        """Convenience accessor for state variables."""
        return self.state.variables


class Policy(ABC):
    """
    Abstract base class for decision policies.
    
    A policy takes a decision context and returns an action.
    All policies must implement the `decide` method.
    
    Policies can be:
    - Rule-based (deterministic conditions)
    - LLM-backed (using language models)
    - RL-trained (learned policies)
    - Random (baseline)
    - Human (interactive)
    """
    
    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__
        self._decision_count = 0
    
    @abstractmethod
    def decide(self, context: DecisionContext) -> Action:
        """
        Make a decision given the current context.
        
        Args:
            context: Decision context with state and available actions
            
        Returns:
            The chosen action
        """
        pass
    
    def reset(self) -> None:
        """Reset any internal state (called between simulation runs)."""
        self._decision_count = 0
    
    @property
    def decision_count(self) -> int:
        """Number of decisions made since last reset."""
        return self._decision_count
    
    def _record_decision(self) -> None:
        """Internal method to track decision count."""
        self._decision_count += 1


class RandomPolicy(Policy):
    """
    Baseline policy that chooses randomly from available actions.
    
    Useful as a baseline for comparison.
    """
    
    def __init__(self, seed: Optional[int] = None, name: str = "RandomPolicy"):
        super().__init__(name)
        import numpy as np
        self._rng = np.random.default_rng(seed)
    
    def decide(self, context: DecisionContext) -> Action:
        self._record_decision()
        idx = self._rng.integers(0, len(context.available_actions))
        return context.available_actions[idx]
    
    def reset(self) -> None:
        super().reset()
        # Don't reset RNG to maintain reproducibility across runs


class ConstantPolicy(Policy):
    """
    Policy that always chooses a specific action type or name.
    
    Useful for testing specific strategies.
    """
    
    def __init__(
        self,
        prefer_action_type: Optional[ActionType] = None,
        prefer_action_name: Optional[str] = None,
        fallback: str = "first",  # "first", "last", "random"
        name: str = "ConstantPolicy",
    ):
        super().__init__(name)
        self.prefer_action_type = prefer_action_type
        self.prefer_action_name = prefer_action_name
        self.fallback = fallback
        
        import numpy as np
        self._rng = np.random.default_rng()
    
    def decide(self, context: DecisionContext) -> Action:
        self._record_decision()
        
        # Try to find preferred action by name
        if self.prefer_action_name:
            action = context.get_action_by_name(self.prefer_action_name)
            if action:
                return action
        
        # Try to find preferred action by type
        if self.prefer_action_type:
            for action in context.available_actions:
                if action.action_type == self.prefer_action_type:
                    return action
        
        # Fallback
        if self.fallback == "first":
            return context.available_actions[0]
        elif self.fallback == "last":
            return context.available_actions[-1]
        else:  # random
            idx = self._rng.integers(0, len(context.available_actions))
            return context.available_actions[idx]
