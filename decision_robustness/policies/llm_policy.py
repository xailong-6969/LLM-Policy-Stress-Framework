"""
LLM-backed policy implementation.

Uses language models (OpenAI, Anthropic, etc.) to make decisions
based on natural language understanding of the state.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from decision_robustness.policies.base import (
    Policy,
    Action,
    DecisionContext,
)


@dataclass
class LLMConfig:
    """
    Configuration for LLM policy.
    
    Attributes:
        provider: "openai" or "anthropic"
        model: Model name (e.g., "gpt-4", "claude-3-opus-20240229")
        temperature: Sampling temperature (0 = deterministic)
        max_tokens: Maximum response tokens
        api_key: API key (defaults to env var)
        system_prompt: System prompt for the LLM
    """
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 256
    api_key: Optional[str] = None
    system_prompt: str = ""
    
    def __post_init__(self):
        if self.provider not in ["openai", "anthropic"]:
            raise ValueError(f"Unknown provider: {self.provider}")
        
        if not self.api_key:
            if self.provider == "openai":
                self.api_key = os.getenv("OPENAI_API_KEY")
            elif self.provider == "anthropic":
                self.api_key = os.getenv("ANTHROPIC_API_KEY")


class LLMPolicy(Policy):
    """
    Policy that uses an LLM to make decisions.
    
    The LLM receives a formatted prompt describing the current state
    and available actions, and must return the name of the chosen action.
    
    Example:
        policy = LLMPolicy(
            config=LLMConfig(
                provider="openai",
                model="gpt-4o-mini",
                temperature=0.0,
            ),
            state_formatter=lambda s: f"Progress: {s.get('progress')}%, Debt: {s.get('debt')}%",
        )
    """
    
    DEFAULT_SYSTEM_PROMPT = """You are a decision-making agent evaluating project management scenarios.
Your goal is to make decisions that maximize project success while managing risk.

Given the current state and available actions, respond with ONLY the action name you choose.
Do not include any explanation or additional text - just the action name exactly as shown."""
    
    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        state_formatter: Optional[Callable[[Dict[str, Any]], str]] = None,
        action_formatter: Optional[Callable[[Action], str]] = None,
        response_parser: Optional[Callable[[str, List[Action]], Action]] = None,
        name: str = "LLMPolicy",
    ):
        """
        Initialize LLM policy.
        
        Args:
            config: LLM configuration
            state_formatter: Function to format state for prompt
            action_formatter: Function to format actions for prompt
            response_parser: Function to parse LLM response into action
            name: Policy name
        """
        super().__init__(name)
        self.config = config or LLMConfig()
        self._state_formatter = state_formatter or self._default_state_formatter
        self._action_formatter = action_formatter or self._default_action_formatter
        self._response_parser = response_parser or self._default_response_parser
        self._client = None
        self._decision_history: List[Dict[str, Any]] = []
    
    def _get_client(self):
        """Lazy initialization of API client."""
        if self._client is None:
            if self.config.provider == "openai":
                try:
                    from openai import OpenAI
                    self._client = OpenAI(api_key=self.config.api_key)
                except ImportError:
                    raise ImportError("openai package required. Install with: pip install openai")
            elif self.config.provider == "anthropic":
                try:
                    import anthropic
                    self._client = anthropic.Anthropic(api_key=self.config.api_key)
                except ImportError:
                    raise ImportError("anthropic package required. Install with: pip install anthropic")
        return self._client
    
    def _default_state_formatter(self, variables: Dict[str, Any]) -> str:
        """Default state formatting."""
        lines = ["Current State:"]
        for key, value in sorted(variables.items()):
            if isinstance(value, float):
                lines.append(f"  - {key}: {value:.1f}")
            else:
                lines.append(f"  - {key}: {value}")
        return "\n".join(lines)
    
    def _default_action_formatter(self, action: Action) -> str:
        """Default action formatting."""
        desc = f" - {action.description}" if action.description else ""
        return f"  • {action.name}{desc}"
    
    def _default_response_parser(self, response: str, actions: List[Action]) -> Action:
        """Parse LLM response to find matching action."""
        response = response.strip().upper()
        
        # Try exact match
        for action in actions:
            if action.name.upper() == response:
                return action
        
        # Try partial match
        for action in actions:
            if action.name.upper() in response or response in action.name.upper():
                return action
        
        # Fallback to first action
        return actions[0]
    
    def _build_prompt(self, context: DecisionContext) -> str:
        """Build the decision prompt."""
        state_desc = self._state_formatter(context.state.variables)
        
        action_lines = ["Available Actions:"]
        for action in context.available_actions:
            action_lines.append(self._action_formatter(action))
        actions_desc = "\n".join(action_lines)
        
        return f"""{state_desc}

{actions_desc}

Choose the best action for this situation. Respond with ONLY the action name."""
    
    def _call_llm(self, prompt: str) -> str:
        """Make LLM API call."""
        client = self._get_client()
        
        system = self.config.system_prompt or self.DEFAULT_SYSTEM_PROMPT
        
        if self.config.provider == "openai":
            response = client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            return response.choices[0].message.content.strip()
        
        elif self.config.provider == "anthropic":
            response = client.messages.create(
                model=self.config.model,
                system=system,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            return response.content[0].text.strip()
        
        raise ValueError(f"Unknown provider: {self.config.provider}")
    
    def decide(self, context: DecisionContext) -> Action:
        self._record_decision()
        
        prompt = self._build_prompt(context)
        
        try:
            response = self._call_llm(prompt)
            action = self._response_parser(response, context.available_actions)
        except Exception as e:
            # On error, fall back to first action
            action = context.available_actions[0]
            response = f"ERROR: {e}"
        
        # Record for analysis
        self._decision_history.append({
            "timestep": context.timestep,
            "prompt": prompt,
            "response": response,
            "action": action.name,
        })
        
        return action
    
    def reset(self) -> None:
        super().reset()
        self._decision_history.clear()
    
    @property
    def decision_history(self) -> List[Dict[str, Any]]:
        return self._decision_history.copy()


class MockLLMPolicy(Policy):
    """
    Mock LLM policy for testing without API calls.
    
    Uses simple heuristics or random selection to simulate LLM behavior.
    """
    
    def __init__(
        self,
        preference_map: Optional[Dict[str, str]] = None,
        seed: Optional[int] = None,
        name: str = "MockLLMPolicy",
    ):
        """
        Initialize mock policy.
        
        Args:
            preference_map: Maps state conditions to action names
            seed: Random seed for non-deterministic choices
            name: Policy name
        """
        super().__init__(name)
        self._preference_map = preference_map or {}
        
        import numpy as np
        self._rng = np.random.default_rng(seed)
    
    def decide(self, context: DecisionContext) -> Action:
        self._record_decision()
        
        # Check preferences
        for condition, action_name in self._preference_map.items():
            # Simple key:value condition parsing
            if ":" in condition:
                key, value = condition.split(":", 1)
                state_value = str(context.state.get(key, ""))
                if value in state_value or state_value == value:
                    action = context.get_action_by_name(action_name)
                    if action:
                        return action
        
        # Random fallback
        idx = self._rng.integers(0, len(context.available_actions))
        return context.available_actions[idx]
