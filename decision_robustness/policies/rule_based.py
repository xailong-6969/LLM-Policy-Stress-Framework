"""
Rule-based policy implementation.

Provides a way to define policies using explicit conditional rules.
Rules are evaluated in order, and the first matching rule's action is taken.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from decision_robustness.policies.base import (
    Policy,
    Action,
    ActionType,
    DecisionContext,
)


@dataclass
class Rule:
    """
    A single decision rule.
    
    A rule consists of a condition and an action. When the condition
    evaluates to True, the action is taken.
    
    Attributes:
        name: Rule identifier for debugging
        condition: Function that takes DecisionContext and returns bool
        action_name: Name of action to take when condition matches
        priority: Higher priority rules are evaluated first
        description: Human-readable description of the rule
    """
    name: str
    condition: Callable[[DecisionContext], bool]
    action_name: str
    priority: int = 0
    description: str = ""
    
    def matches(self, context: DecisionContext) -> bool:
        """Check if this rule's condition matches the context."""
        try:
            return self.condition(context)
        except Exception:
            return False
    
    def get_action(self, context: DecisionContext) -> Optional[Action]:
        """Get the action for this rule, if available."""
        return context.get_action_by_name(self.action_name)


class RuleBasedPolicy(Policy):
    """
    Policy that makes decisions based on ordered rules.
    
    Rules are evaluated by priority (highest first), then by order added.
    The first rule whose condition matches determines the action.
    
    Example:
        policy = RuleBasedPolicy([
            Rule(
                name="low_morale_conservative",
                condition=lambda ctx: ctx.state.get("morale") < 30,
                action_name="DELAY",
                priority=100,
                description="When morale is low, delay to recover"
            ),
            Rule(
                name="high_debt_refactor",
                condition=lambda ctx: ctx.state.get("tech_debt") > 70,
                action_name="REFACTOR",
                priority=50,
                description="High tech debt triggers refactoring"
            ),
            Rule(
                name="default_proceed",
                condition=lambda ctx: True,  # Always matches
                action_name="SHIP",
                priority=0,
                description="Default: keep shipping"
            ),
        ])
    """
    
    def __init__(
        self,
        rules: List[Rule],
        fallback_action_name: Optional[str] = None,
        name: str = "RuleBasedPolicy",
    ):
        """
        Initialize rule-based policy.
        
        Args:
            rules: List of rules to evaluate
            fallback_action_name: Action to take if no rules match
            name: Policy name
        """
        super().__init__(name)
        self._rules = sorted(rules, key=lambda r: -r.priority)  # Sort by priority desc
        self._fallback_action_name = fallback_action_name
        self._last_matched_rule: Optional[Rule] = None
    
    @property
    def rules(self) -> List[Rule]:
        return self._rules.copy()
    
    @property
    def last_matched_rule(self) -> Optional[Rule]:
        """The rule that matched in the last decision."""
        return self._last_matched_rule
    
    def add_rule(self, rule: Rule) -> None:
        """Add a new rule and re-sort by priority."""
        self._rules.append(rule)
        self._rules.sort(key=lambda r: -r.priority)
    
    def remove_rule(self, rule_name: str) -> bool:
        """Remove a rule by name. Returns True if found and removed."""
        original_len = len(self._rules)
        self._rules = [r for r in self._rules if r.name != rule_name]
        return len(self._rules) < original_len
    
    def decide(self, context: DecisionContext) -> Action:
        self._record_decision()
        self._last_matched_rule = None
        
        # Evaluate rules in priority order
        for rule in self._rules:
            if rule.matches(context):
                action = rule.get_action(context)
                if action:
                    self._last_matched_rule = rule
                    return action
        
        # No rule matched, use fallback
        if self._fallback_action_name:
            action = context.get_action_by_name(self._fallback_action_name)
            if action:
                return action
        
        # Last resort: return first available action
        return context.available_actions[0]
    
    def explain_decision(self, context: DecisionContext) -> Dict[str, Any]:
        """
        Explain why a particular decision was made.
        
        Returns which rules matched and which was selected.
        """
        matching_rules = []
        for rule in self._rules:
            if rule.matches(context):
                matching_rules.append({
                    "name": rule.name,
                    "description": rule.description,
                    "action": rule.action_name,
                    "priority": rule.priority,
                })
        
        return {
            "matching_rules": matching_rules,
            "selected_rule": matching_rules[0] if matching_rules else None,
            "total_rules": len(self._rules),
        }


# Convenience functions for building rules

def when(condition: Callable[[DecisionContext], bool]) -> Callable:
    """
    Fluent interface for building rules.
    
    Example:
        rule = when(lambda ctx: ctx.state.get("debt") > 50).then("REFACTOR")
    """
    class RuleBuilder:
        def __init__(self, condition: Callable[[DecisionContext], bool]):
            self.condition = condition
            self.rule_name = "unnamed_rule"
            self.rule_priority = 0
            self.rule_description = ""
        
        def named(self, name: str) -> "RuleBuilder":
            self.rule_name = name
            return self
        
        def with_priority(self, priority: int) -> "RuleBuilder":
            self.rule_priority = priority
            return self
        
        def described_as(self, description: str) -> "RuleBuilder":
            self.rule_description = description
            return self
        
        def then(self, action_name: str) -> Rule:
            return Rule(
                name=self.rule_name,
                condition=self.condition,
                action_name=action_name,
                priority=self.rule_priority,
                description=self.rule_description,
            )
    
    return RuleBuilder(condition)


# Common condition helpers

def state_lt(key: str, value: Any) -> Callable[[DecisionContext], bool]:
    """Condition: state[key] < value"""
    return lambda ctx: ctx.state.get(key, 0) < value


def state_gt(key: str, value: Any) -> Callable[[DecisionContext], bool]:
    """Condition: state[key] > value"""
    return lambda ctx: ctx.state.get(key, 0) > value


def state_eq(key: str, value: Any) -> Callable[[DecisionContext], bool]:
    """Condition: state[key] == value"""
    return lambda ctx: ctx.state.get(key) == value


def state_between(key: str, low: Any, high: Any) -> Callable[[DecisionContext], bool]:
    """Condition: low <= state[key] <= high"""
    return lambda ctx: low <= ctx.state.get(key, 0) <= high


def always() -> Callable[[DecisionContext], bool]:
    """Condition that always matches (for default rules)."""
    return lambda ctx: True
