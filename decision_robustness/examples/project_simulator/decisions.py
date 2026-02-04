"""
Project decision types.

Defines the actions available in the project simulator.
"""

from decision_robustness.policies.base import Action, ActionType


class ProjectAction(Action):
    """Action specific to project simulation."""
    pass


def create_project_actions():
    """Create the set of available project actions."""
    return [
        ProjectAction(
            name="SHIP_NOW",
            action_type=ActionType.AGGRESSIVE,
            description="Rush to ship features, accumulating technical debt",
            risk_level=0.7,
            cost=3,
        ),
        ProjectAction(
            name="REFACTOR",
            action_type=ActionType.INVEST,
            description="Spend time cleaning up code and reducing debt",
            risk_level=0.2,
            cost=2,
        ),
        ProjectAction(
            name="HIRE",
            action_type=ActionType.INVEST,
            description="Invest in hiring to increase team capacity",
            risk_level=0.4,
            cost=8,
        ),
        ProjectAction(
            name="CUT_SCOPE",
            action_type=ActionType.CUT,
            description="Reduce project scope to accelerate delivery",
            risk_level=0.5,
            cost=1,
        ),
        ProjectAction(
            name="FIX_BUGS",
            action_type=ActionType.CONSERVATIVE,
            description="Focus on fixing bugs and improving quality",
            risk_level=0.2,
            cost=2,
        ),
        ProjectAction(
            name="DELAY",
            action_type=ActionType.DELAY,
            description="Wait and plan before taking action",
            risk_level=0.1,
            cost=1,
        ),
    ]
