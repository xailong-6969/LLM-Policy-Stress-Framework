"""
Tests for policy implementations.
"""

import pytest
from decision_robustness.policies.base import (
    Policy,
    Action,
    ActionType,
    DecisionContext,
    RandomPolicy,
    ConstantPolicy,
)
from decision_robustness.policies.rule_based import (
    RuleBasedPolicy,
    Rule,
    when,
    state_lt,
    state_gt,
    always,
)
from decision_robustness.engine.world import WorldState


def make_context(variables, actions=None):
    """Helper to create test context."""
    state = WorldState(
        timestep=0,
        variables=variables,
    )
    
    if actions is None:
        actions = [
            Action(name="action1"),
            Action(name="action2"),
            Action(name="action3"),
        ]
    
    return DecisionContext(
        state=state,
        available_actions=actions,
        timestep=0,
    )


class TestAction:
    """Tests for Action class."""
    
    def test_action_creation(self):
        action = Action(
            name="test",
            action_type=ActionType.AGGRESSIVE,
            description="A test action",
        )
        
        assert action.name == "test"
        assert action.action_type == ActionType.AGGRESSIVE
    
    def test_action_equality(self):
        action1 = Action(name="test")
        action2 = Action(name="test")
        action3 = Action(name="other")
        
        assert action1 == action2
        assert action1 != action3


class TestDecisionContext:
    """Tests for DecisionContext."""
    
    def test_get_action_by_name(self):
        context = make_context({"value": 50})
        
        action = context.get_action_by_name("action2")
        assert action is not None
        assert action.name == "action2"
        
        missing = context.get_action_by_name("nonexistent")
        assert missing is None
    
    def test_state_variables_access(self):
        context = make_context({"a": 1, "b": 2})
        
        assert context.state_variables["a"] == 1
        assert context.state_variables["b"] == 2


class TestRandomPolicy:
    """Tests for RandomPolicy."""
    
    def test_random_selection(self):
        policy = RandomPolicy(seed=42)
        context = make_context({"value": 50})
        
        decisions = [policy.decide(context).name for _ in range(100)]
        
        # Should see variety
        unique = set(decisions)
        assert len(unique) > 1
    
    def test_seed_reproducibility(self):
        policy1 = RandomPolicy(seed=42)
        policy2 = RandomPolicy(seed=42)
        context = make_context({"value": 50})
        
        decisions1 = [policy1.decide(context).name for _ in range(10)]
        decisions2 = [policy2.decide(context).name for _ in range(10)]
        
        assert decisions1 == decisions2


class TestConstantPolicy:
    """Tests for ConstantPolicy."""
    
    def test_prefer_by_name(self):
        policy = ConstantPolicy(prefer_action_name="action2")
        context = make_context({"value": 50})
        
        action = policy.decide(context)
        assert action.name == "action2"
    
    def test_prefer_by_type(self):
        actions = [
            Action(name="a", action_type=ActionType.AGGRESSIVE),
            Action(name="b", action_type=ActionType.CONSERVATIVE),
        ]
        
        policy = ConstantPolicy(prefer_action_type=ActionType.CONSERVATIVE)
        context = make_context({"value": 50}, actions)
        
        action = policy.decide(context)
        assert action.name == "b"
    
    def test_fallback(self):
        policy = ConstantPolicy(
            prefer_action_name="nonexistent",
            fallback="first",
        )
        context = make_context({"value": 50})
        
        action = policy.decide(context)
        assert action.name == "action1"


class TestRuleBasedPolicy:
    """Tests for RuleBasedPolicy."""
    
    def test_rule_matching(self):
        rules = [
            Rule(
                name="low_value",
                condition=lambda ctx: ctx.state.get("value") < 30,
                action_name="action1",
                priority=100,
            ),
            Rule(
                name="high_value",
                condition=lambda ctx: ctx.state.get("value") > 70,
                action_name="action2",
                priority=100,
            ),
            Rule(
                name="default",
                condition=lambda ctx: True,
                action_name="action3",
                priority=0,
            ),
        ]
        
        policy = RuleBasedPolicy(rules)
        
        # Low value
        low_ctx = make_context({"value": 20})
        assert policy.decide(low_ctx).name == "action1"
        
        # High value
        high_ctx = make_context({"value": 80})
        assert policy.decide(high_ctx).name == "action2"
        
        # Middle value
        mid_ctx = make_context({"value": 50})
        assert policy.decide(mid_ctx).name == "action3"
    
    def test_priority_ordering(self):
        rules = [
            Rule(
                name="low_priority",
                condition=always(),
                action_name="action1",
                priority=10,
            ),
            Rule(
                name="high_priority",
                condition=always(),
                action_name="action2",
                priority=100,
            ),
        ]
        
        policy = RuleBasedPolicy(rules)
        context = make_context({"value": 50})
        
        # High priority should win
        assert policy.decide(context).name == "action2"
    
    def test_explain_decision(self):
        rules = [
            Rule(
                name="test_rule",
                condition=always(),
                action_name="action1",
                description="Always matches",
            ),
        ]
        
        policy = RuleBasedPolicy(rules)
        context = make_context({"value": 50})
        
        explanation = policy.explain_decision(context)
        
        assert len(explanation["matching_rules"]) == 1
        assert explanation["selected_rule"]["name"] == "test_rule"


class TestRuleHelpers:
    """Tests for rule helper functions."""
    
    def test_when_builder(self):
        rule = (
            when(lambda ctx: ctx.state.get("x") > 10)
            .named("high_x")
            .with_priority(50)
            .described_as("When x is high")
            .then("action1")
        )
        
        assert rule.name == "high_x"
        assert rule.priority == 50
        assert rule.action_name == "action1"
    
    def test_state_lt(self):
        condition = state_lt("value", 50)
        
        low_ctx = make_context({"value": 30})
        high_ctx = make_context({"value": 70})
        
        assert condition(low_ctx) is True
        assert condition(high_ctx) is False
    
    def test_state_gt(self):
        condition = state_gt("value", 50)
        
        low_ctx = make_context({"value": 30})
        high_ctx = make_context({"value": 70})
        
        assert condition(low_ctx) is False
        assert condition(high_ctx) is True
