"""
Stochastic event system for world simulation.

Events represent random occurrences that can affect the world state.
They are the source of uncertainty in simulations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from decision_robustness.engine.world import WorldState


@dataclass
class Event:
    """
    Represents a stochastic event that can occur in the world.
    
    Attributes:
        name: Unique identifier for the event
        description: Human-readable description
        probability: Base probability of occurrence per timestep (0-1)
        severity: Impact magnitude (0-1, higher = more severe)
        is_irreversible: Whether the event cannot be undone
        cooldown: Minimum timesteps before event can occur again
        metadata: Additional event-specific data
    """
    name: str
    description: str
    probability: float
    severity: float = 0.5
    is_irreversible: bool = False
    cooldown: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not 0 <= self.probability <= 1:
            raise ValueError(f"Probability must be 0-1, got {self.probability}")
        if not 0 <= self.severity <= 1:
            raise ValueError(f"Severity must be 0-1, got {self.severity}")


@dataclass
class EventOccurrence:
    """
    Records when an event occurred during simulation.
    
    Attributes:
        event: The event that occurred
        timestep: When it occurred
        state_before: State snapshot before event
        state_after: State snapshot after event
        details: Event-specific occurrence details
    """
    event: Event
    timestep: int
    state_before: Optional[Dict[str, Any]] = None
    state_after: Optional[Dict[str, Any]] = None
    details: Dict[str, Any] = field(default_factory=dict)


class EventGenerator(ABC):
    """
    Abstract base class for event generation systems.
    
    An EventGenerator determines which events occur at each timestep
    and how they modify the world state.
    """
    
    @abstractmethod
    def get_events(self) -> List[Event]:
        """Return list of all possible events."""
        pass
    
    @abstractmethod
    def sample_events(
        self,
        state: "WorldState",
        rng: np.random.Generator
    ) -> List[Event]:
        """
        Sample which events occur this timestep.
        
        Args:
            state: Current world state
            rng: Random number generator
            
        Returns:
            List of events that occurred
        """
        pass
    
    @abstractmethod
    def apply_event(
        self,
        state: "WorldState",
        event: Event,
        rng: np.random.Generator
    ) -> "WorldState":
        """
        Apply an event to the world state.
        
        Args:
            state: Current world state
            event: Event to apply
            rng: Random number generator
            
        Returns:
            New state after event is applied
        """
        pass


class SimpleEventGenerator(EventGenerator):
    """
    Basic event generator with independent event probabilities.
    
    Events are sampled independently based on their base probabilities,
    with optional modifiers based on state.
    """
    
    def __init__(
        self,
        events: List[Event],
        probability_modifiers: Optional[Dict[str, Callable[["WorldState"], float]]] = None,
        event_handlers: Optional[Dict[str, Callable[["WorldState", Event, np.random.Generator], "WorldState"]]] = None,
    ):
        """
        Initialize the event generator.
        
        Args:
            events: List of possible events
            probability_modifiers: Optional dict mapping event names to functions
                that modify probability based on state (return multiplier)
            event_handlers: Optional dict mapping event names to custom handlers
        """
        self._events = {e.name: e for e in events}
        self._probability_modifiers = probability_modifiers or {}
        self._event_handlers = event_handlers or {}
        self._cooldowns: Dict[str, int] = {}  # Track cooldowns per event
    
    def get_events(self) -> List[Event]:
        return list(self._events.values())
    
    def _get_effective_probability(self, event: Event, state: "WorldState") -> float:
        """Calculate effective probability considering state modifiers."""
        base_prob = event.probability
        
        if event.name in self._probability_modifiers:
            modifier = self._probability_modifiers[event.name](state)
            base_prob *= modifier
        
        return min(1.0, max(0.0, base_prob))
    
    def sample_events(
        self,
        state: "WorldState",
        rng: np.random.Generator
    ) -> List[Event]:
        """Sample events for this timestep."""
        occurred = []
        timestep = state.timestep
        
        for event in self._events.values():
            # Check cooldown
            if event.name in self._cooldowns:
                if timestep < self._cooldowns[event.name]:
                    continue
            
            # Sample with effective probability
            prob = self._get_effective_probability(event, state)
            if rng.random() < prob:
                occurred.append(event)
                
                # Set cooldown if applicable
                if event.cooldown > 0:
                    self._cooldowns[event.name] = timestep + event.cooldown
        
        return occurred
    
    def apply_event(
        self,
        state: "WorldState",
        event: Event,
        rng: np.random.Generator
    ) -> "WorldState":
        """Apply event using custom handler or default behavior."""
        if event.name in self._event_handlers:
            return self._event_handlers[event.name](state, event, rng)
        
        # Default: just mark that event occurred in metadata
        new_metadata = dict(state.metadata)
        events_occurred = new_metadata.get("events_occurred", [])
        events_occurred.append({
            "name": event.name,
            "timestep": state.timestep,
            "severity": event.severity,
        })
        new_metadata["events_occurred"] = events_occurred
        
        return state.evolve(metadata=new_metadata)
    
    def reset_cooldowns(self) -> None:
        """Reset all cooldown trackers."""
        self._cooldowns.clear()


class CompositeEventGenerator(EventGenerator):
    """
    Combines multiple event generators.
    
    Useful for separating different categories of events
    (e.g., team events, technical events, external events).
    """
    
    def __init__(self, generators: List[EventGenerator]):
        self._generators = generators
    
    def get_events(self) -> List[Event]:
        events = []
        for gen in self._generators:
            events.extend(gen.get_events())
        return events
    
    def sample_events(
        self,
        state: "WorldState",
        rng: np.random.Generator
    ) -> List[Event]:
        occurred = []
        for gen in self._generators:
            occurred.extend(gen.sample_events(state, rng))
        return occurred
    
    def apply_event(
        self,
        state: "WorldState",
        event: Event,
        rng: np.random.Generator
    ) -> "WorldState":
        # Find the generator that owns this event
        for gen in self._generators:
            if event in gen.get_events():
                return gen.apply_event(state, event, rng)
        
        # Fallback: no-op
        return state
