"""
Example policies for the project simulator.

Demonstrates different decision-making strategies.
"""

from decision_robustness.policies.base import Policy, DecisionContext
from decision_robustness.policies.rule_based import (
    RuleBasedPolicy,
    Rule,
    when,
    state_lt,
    state_gt,
    always,
)


class AggressivePolicy(RuleBasedPolicy):
    """
    Aggressive policy that prioritizes shipping fast.
    
    This policy always pushes for speed, only backing off
    when absolutely necessary.
    """
    
    def __init__(self):
        rules = [
            # Only cut scope if desperate
            Rule(
                name="desperate_cut",
                condition=lambda ctx: (
                    ctx.state.get("budget") < 20 and
                    ctx.state.get("progress") < 80
                ),
                action_name="CUT_SCOPE",
                priority=100,
                description="Cut scope when budget critically low",
            ),
            # Fix bugs if overwhelming
            Rule(
                name="bug_crisis",
                condition=lambda ctx: ctx.state.get("bugs") > 15,
                action_name="FIX_BUGS",
                priority=90,
                description="Fix bugs when they're overwhelming",
            ),
            # Default: ship ship ship
            Rule(
                name="default_ship",
                condition=always(),
                action_name="SHIP_NOW",
                priority=0,
                description="Always prioritize shipping",
            ),
        ]
        super().__init__(rules, name="AggressivePolicy")


class ConservativePolicy(RuleBasedPolicy):
    """
    Conservative policy that prioritizes sustainability.
    
    This policy focuses on maintaining healthy metrics
    and only ships when conditions are favorable.
    """
    
    def __init__(self):
        rules = [
            # High debt? Refactor first
            Rule(
                name="manage_debt",
                condition=lambda ctx: ctx.state.get("debt") > 50,
                action_name="REFACTOR",
                priority=100,
                description="Keep debt under control",
            ),
            # Low morale? Boost it
            Rule(
                name="manage_morale",
                condition=lambda ctx: ctx.state.get("morale") < 50,
                action_name="HIRE",
                priority=90,
                description="Invest in team when morale drops",
            ),
            # Bugs piling up? Fix them
            Rule(
                name="manage_bugs",
                condition=lambda ctx: ctx.state.get("bugs") > 5,
                action_name="FIX_BUGS",
                priority=80,
                description="Keep bugs under control",
            ),
            # Good conditions? Ship carefully
            Rule(
                name="careful_ship",
                condition=lambda ctx: (
                    ctx.state.get("debt") < 40 and
                    ctx.state.get("morale") > 60
                ),
                action_name="SHIP_NOW",
                priority=50,
                description="Ship when conditions are good",
            ),
            # Default: delay and plan
            Rule(
                name="default_delay",
                condition=always(),
                action_name="DELAY",
                priority=0,
                description="Wait when uncertain",
            ),
        ]
        super().__init__(rules, name="ConservativePolicy")


class BalancedPolicy(RuleBasedPolicy):
    """
    Balanced policy that adapts to circumstances.
    
    This policy tries to maintain a balance between
    shipping speed and project health.
    """
    
    def __init__(self):
        rules = [
            # Critical: morale collapse incoming
            Rule(
                name="morale_emergency",
                condition=lambda ctx: ctx.state.get("morale") < 30,
                action_name="HIRE",
                priority=100,
                description="Emergency morale intervention",
            ),
            # Critical: budget running out
            Rule(
                name="budget_emergency",
                condition=lambda ctx: ctx.state.get("budget") < 15,
                action_name="CUT_SCOPE",
                priority=95,
                description="Emergency scope cut",
            ),
            # High debt is dangerous
            Rule(
                name="high_debt",
                condition=lambda ctx: ctx.state.get("debt") > 70,
                action_name="REFACTOR",
                priority=90,
                description="Debt getting dangerous",
            ),
            # Too many bugs
            Rule(
                name="bug_problem",
                condition=lambda ctx: ctx.state.get("bugs") > 10,
                action_name="FIX_BUGS",
                priority=85,
                description="Bugs need attention",
            ),
            # Behind schedule? Push harder
            Rule(
                name="behind_schedule",
                condition=lambda ctx: (
                    ctx.state.get("progress") < ctx.timestep * 2 and
                    ctx.state.get("debt") < 60
                ),
                action_name="SHIP_NOW",
                priority=70,
                description="Need to catch up",
            ),
            # Good position? Keep shipping
            Rule(
                name="on_track",
                condition=lambda ctx: ctx.state.get("morale") > 50,
                action_name="SHIP_NOW",
                priority=50,
                description="Making good progress",
            ),
            # Default: maintain
            Rule(
                name="default",
                condition=always(),
                action_name="FIX_BUGS",
                priority=0,
                description="Default maintenance",
            ),
        ]
        super().__init__(rules, name="BalancedPolicy")
