"""
Project World Configuration.

A simulation of a software project with realistic dynamics:
- Technical debt accumulates with rushed decisions
- Team morale affects productivity
- Bugs emerge from debt
- Deadlines create pressure
- Events can disrupt progress
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np

from decision_robustness.engine.world import World, WorldState
from decision_robustness.engine.events import Event, SimpleEventGenerator


@dataclass(frozen=True)
class ProjectState(WorldState):
    """
    State of a software project.
    
    Inherits from WorldState, using variables dict for project state.
    """
    pass


class ProjectWorld(World[ProjectState]):
    """
    Simulates a software project over time.
    
    Models the dynamics of:
    - Progress toward completion
    - Technical debt accumulation
    - Team morale
    - Bug discovery/fixing
    - Budget consumption
    
    The goal is to complete the project (progress=100) without:
    - Running out of budget
    - Team morale collapsing (morale=0)
    - Technical debt overwhelming the project (debt>90 with bugs>20)
    """
    
    # Project configuration
    DEFAULT_CONFIG = {
        "initial_progress": 0,
        "initial_debt": 10,
        "initial_morale": 75,
        "initial_budget": 100,
        "initial_bugs": 0,
        "max_steps": 50,
        "base_progress_rate": 5,
        "debt_bug_threshold": 40,
    }
    
    def __init__(
        self,
        seed: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(seed)
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self._event_generator = self._create_event_generator()
    
    def _create_event_generator(self) -> SimpleEventGenerator:
        """Create the event system for the project."""
        events = [
            Event(
                name="team_member_quits",
                description="A team member quits unexpectedly",
                probability=0.05,
                severity=0.7,
                is_irreversible=True,
                cooldown=10,
            ),
            Event(
                name="scope_creep",
                description="Stakeholder requests new features",
                probability=0.08,
                severity=0.4,
            ),
            Event(
                name="critical_bug_discovered",
                description="A critical bug is discovered in production",
                probability=0.03,
                severity=0.8,
            ),
            Event(
                name="dependency_update_breaks",
                description="A dependency update breaks the build",
                probability=0.04,
                severity=0.5,
                cooldown=5,
            ),
            Event(
                name="morale_boost",
                description="Team celebration or good news",
                probability=0.05,
                severity=0.3,  # Positive event
            ),
        ]
        
        # Probability modifiers based on state
        def debt_modifier(state: WorldState) -> float:
            debt = state.get("debt", 0)
            # More debt = more bugs discovered
            return 1.0 + (debt / 100)
        
        def morale_modifier(state: WorldState) -> float:
            morale = state.get("morale", 50)
            # Low morale = more quits
            if morale < 30:
                return 2.0
            elif morale < 50:
                return 1.5
            return 1.0
        
        # Event handlers
        def handle_quit(state: WorldState, event: Event, rng) -> WorldState:
            # Losing a team member hurts progress rate and morale
            return state.evolve(
                morale=max(0, state.get("morale", 50) - 20),
                _productivity_modifier=state.get("_productivity_modifier", 1.0) * 0.75,
            )
        
        def handle_scope_creep(state: WorldState, event: Event, rng) -> WorldState:
            # Scope creep increases work but doesn't reduce progress
            progress = state.get("progress", 0)
            return state.evolve(
                progress=max(0, progress - 5),
                morale=max(0, state.get("morale", 50) - 5),
            )
        
        def handle_critical_bug(state: WorldState, event: Event, rng) -> WorldState:
            return state.evolve(
                bugs=state.get("bugs", 0) + 5,
                morale=max(0, state.get("morale", 50) - 10),
            )
        
        def handle_dependency_break(state: WorldState, event: Event, rng) -> WorldState:
            return state.evolve(
                progress=max(0, state.get("progress", 0) - 3),
                debt=min(100, state.get("debt", 0) + 5),
            )
        
        def handle_morale_boost(state: WorldState, event: Event, rng) -> WorldState:
            return state.evolve(
                morale=min(100, state.get("morale", 50) + 10),
            )
        
        return SimpleEventGenerator(
            events=events,
            probability_modifiers={
                "critical_bug_discovered": debt_modifier,
                "team_member_quits": morale_modifier,
            },
            event_handlers={
                "team_member_quits": handle_quit,
                "scope_creep": handle_scope_creep,
                "critical_bug_discovered": handle_critical_bug,
                "dependency_update_breaks": handle_dependency_break,
                "morale_boost": handle_morale_boost,
            },
        )
    
    def initial_state(self) -> ProjectState:
        """Create initial project state."""
        return ProjectState(
            timestep=0,
            variables={
                "progress": self.config["initial_progress"],
                "debt": self.config["initial_debt"],
                "morale": self.config["initial_morale"],
                "budget": self.config["initial_budget"],
                "bugs": self.config["initial_bugs"],
                "_productivity_modifier": 1.0,
            },
            is_terminal=False,
            metadata={
                "events_occurred": [],
            },
        )
    
    def step(self, state: ProjectState, action: Any) -> ProjectState:
        """Advance project by one week given an action."""
        progress = state.get("progress", 0)
        debt = state.get("debt", 0)
        morale = state.get("morale", 50)
        budget = state.get("budget", 100)
        bugs = state.get("bugs", 0)
        productivity = state.get("_productivity_modifier", 1.0)
        
        # Base progress (affected by morale and productivity)
        morale_factor = 0.5 + (morale / 200)  # 0.5 to 1.0
        base_progress = self.config["base_progress_rate"] * morale_factor * productivity
        
        # Action effects
        action_name = action.name if hasattr(action, 'name') else str(action)
        
        if action_name == "SHIP_NOW":
            # Rush to ship - high progress but accumulates debt
            progress_delta = base_progress * 1.5
            debt_delta = 8
            morale_delta = -3
            budget_delta = -3
            bugs_delta = int(debt / 20)  # Bugs proportional to debt
            
        elif action_name == "REFACTOR":
            # Reduce debt at cost of progress
            progress_delta = base_progress * 0.3
            debt_delta = -15
            morale_delta = 2  # Devs like clean code
            budget_delta = -2
            bugs_delta = -3
            
        elif action_name == "HIRE":
            # Invest in team growth
            progress_delta = base_progress * 0.5  # Short-term slowdown
            debt_delta = 2
            morale_delta = 5
            budget_delta = -8
            bugs_delta = 0
            productivity = productivity * 1.2  # Long-term gain
            
        elif action_name == "CUT_SCOPE":
            # Reduce requirements
            progress_delta = base_progress * 0.8 + 10  # Instant progress boost
            debt_delta = 3
            morale_delta = -5  # Team doesn't like cutting features
            budget_delta = -1
            bugs_delta = 0
            
        elif action_name == "FIX_BUGS":
            # Focus on quality
            progress_delta = base_progress * 0.4
            debt_delta = -5
            morale_delta = 3
            budget_delta = -2
            bugs_delta = -max(5, bugs // 2)
            
        elif action_name == "DELAY":
            # Wait and plan
            progress_delta = base_progress * 0.2
            debt_delta = 0
            morale_delta = -2  # Team gets bored
            budget_delta = -1
            bugs_delta = 0
            
        else:  # Default: steady progress
            progress_delta = base_progress
            debt_delta = 3
            morale_delta = 0
            budget_delta = -2
            bugs_delta = int(debt / 30)
        
        # Debt causes more bugs naturally
        if debt > self.config["debt_bug_threshold"]:
            bugs_delta += int((debt - self.config["debt_bug_threshold"]) / 20)
        
        # Apply changes
        new_state = state.evolve(
            timestep=state.timestep + 1,
            progress=min(100, max(0, progress + progress_delta)),
            debt=min(100, max(0, debt + debt_delta)),
            morale=min(100, max(0, morale + morale_delta)),
            budget=max(0, budget + budget_delta),
            bugs=max(0, bugs + bugs_delta),
            _productivity_modifier=productivity,
        )
        
        return new_state
    
    def get_available_actions(self, state: ProjectState) -> List[Any]:
        """Get available actions for current state."""
        from decision_robustness.examples.project_simulator.decisions import create_project_actions
        return create_project_actions()
    
    def apply_events(self, state: ProjectState) -> ProjectState:
        """Apply random events."""
        events = self._event_generator.sample_events(state, self.rng)
        
        for event in events:
            state = self._event_generator.apply_event(state, event, self.rng)
        
        return state
    
    def check_terminal(self, state: ProjectState) -> ProjectState:
        """Check for terminal conditions."""
        progress = state.get("progress", 0)
        morale = state.get("morale", 50)
        budget = state.get("budget", 100)
        debt = state.get("debt", 0)
        bugs = state.get("bugs", 0)
        
        # Success: project completed
        if progress >= 100:
            return state.evolve(
                is_terminal=True,
                terminal_reason="success",
            )
        
        # Failure: out of budget
        if budget <= 0:
            return state.evolve(
                is_terminal=True,
                terminal_reason="failure",
                metadata={**state.metadata, "failure_cause": "budget_exhausted"},
            )
        
        # Failure: morale collapse
        if morale <= 0:
            return state.evolve(
                is_terminal=True,
                terminal_reason="failure",
                metadata={**state.metadata, "failure_cause": "morale_collapse"},
            )
        
        # Failure: overwhelmed by bugs and debt
        if debt >= 90 and bugs >= 20:
            return state.evolve(
                is_terminal=True,
                terminal_reason="failure",
                metadata={**state.metadata, "failure_cause": "technical_collapse"},
            )
        
        return state
    
    def get_outcome_score(self, state: ProjectState) -> float:
        """Calculate outcome score (0-1)."""
        if state.is_success():
            # Success: score based on remaining resources
            budget = state.get("budget", 0)
            morale = state.get("morale", 50)
            debt = state.get("debt", 50)
            
            # Higher score for finishing with resources
            base_score = 0.7
            budget_bonus = (budget / 100) * 0.1
            morale_bonus = (morale / 100) * 0.1
            debt_penalty = (debt / 100) * 0.1
            
            return min(1.0, base_score + budget_bonus + morale_bonus - debt_penalty)
        
        elif state.is_failed():
            # Failure: score based on progress made
            progress = state.get("progress", 0)
            return (progress / 100) * 0.3  # Max 0.3 for failures
        
        else:
            # Timeout: partial credit
            progress = state.get("progress", 0)
            return 0.3 + (progress / 100) * 0.3
    
    def describe_state(self, state: ProjectState) -> str:
        """Human-readable state description."""
        return (
            f"Week {state.timestep}: "
            f"Progress={state.get('progress', 0):.0f}%, "
            f"Debt={state.get('debt', 0):.0f}, "
            f"Morale={state.get('morale', 0):.0f}, "
            f"Budget={state.get('budget', 0):.0f}, "
            f"Bugs={state.get('bugs', 0)}"
        )
