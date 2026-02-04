# Decision Robustness: A Distribution-Level Framework for Evaluating AI Decision Policies

**Abstract:** We present a framework for evaluating AI decision-making systems based on outcome distributions rather than single-run accuracy. By simulating thousands of parallel futures with stochastic events, we measure policy robustness through survival curves, collapse probability, and sensitivity analysis. This approach addresses a critical gap in AI evaluation: understanding how policies perform under uncertainty and delayed consequences.

---

## 1. Introduction

### The Problem with Point Evaluations

Current AI benchmarks evaluate models on discrete question-answer pairs:
- MMLU: "What is the capital of France?" → "Paris" ✓
- HumanEval: "Implement quicksort" → Code ✓
- LMSYS Arena: "Which response is better?" → A vs B

These evaluations share a fundamental limitation: **they measure single outcomes, not outcome distributions**.

For AI systems making consequential decisions (agents, planners, autonomous systems), the relevant question is not "Was this one decision correct?" but rather:

> **"What is the distribution of outcomes when this policy interacts with an uncertain environment over time?"**

### Why Distributions Matter

Consider an AI project manager deciding between "SHIP_NOW" vs "REFACTOR":

| Evaluation Type | Question | Answer |
|-----------------|----------|--------|
| **Point** | "Is SHIP_NOW correct here?" | Maybe |
| **Distribution** | "What happens across 1000 futures if we SHIP_NOW?" | 67% fail, 33% succeed, mean survival 23 steps |

The distribution reveals:
- Tail risks (catastrophic outcomes)
- Sensitivity to randomness
- Degradation patterns over time

---

## 2. Framework Architecture

### 2.1 World Simulation Engine

A **World** is a stochastic state machine:

```
WorldState = {
    timestep: int,
    variables: {key: value, ...},
    is_terminal: bool,
    terminal_reason: "success" | "failure" | "timeout"
}
```

Each world can:
1. **Step**: Advance state given an action
2. **Apply Events**: Inject stochastic perturbations
3. **Check Terminal**: Detect success/failure conditions
4. **Score Outcome**: Quantify quality of final state

### 2.2 Policy Interface

A **Policy** maps decision contexts to actions:

```python
class Policy(ABC):
    def decide(self, context: DecisionContext) -> Action:
        """Given current state, return chosen action."""
        pass
```

Supported policy types:
- **Rule-based**: Priority-ordered condition-action rules
- **LLM-backed**: Prompt → API call → Parse response
- **Random**: Baseline for comparison
- **Hybrid**: Rules with LLM fallback

### 2.3 Swarm Execution

The **SwarmExecutor** runs N parallel simulations:

```
For each seed in [base_seed, base_seed + n_worlds]:
    world = create_world(seed)
    while not world.is_terminal:
        action = policy.decide(world.state)
        world.step(action)
        world.apply_events()
    record(outcome)

Aggregate → Distribution
```

Each seed produces a different future due to:
- Different random event occurrences
- Different event timings
- Different event severities

---

## 3. Robustness Metrics

### 3.1 Survival Analysis

**Kaplan-Meier Survival Curve**: P(still running) at each timestep

```
S(t) = ∏_{t_i ≤ t} (1 - d_i/n_i)

where:
  d_i = failures at time t_i
  n_i = at-risk population at t_i
```

Key outputs:
- Median survival time
- Survival probability at horizon T
- Hazard rate over time

### 3.2 Collapse Metrics

**Collapse Probability**: P(terminal_reason = "failure")

```
P(collapse) = (# failures) / (# total runs)
```

Decomposed by timing:
- **Early collapse**: Failures in first 20% of max steps
- **Late collapse**: Failures in last 20%
- **Conditional collapse**: P(fail | survived to step t)

### 3.3 Regret Analysis

**Regret**: Distance from optimal outcome

```
regret_i = optimal_score - actual_score_i
```

Distribution statistics:
- Mean regret
- Tail regret (worst 10%)
- Cumulative regret over time

### 3.4 Sensitivity Analysis

**Brittleness Score**: How much does outcome variance depend on randomness?

```
brittleness = CV(scores) × P(failure | high noise)
            = (σ/μ) × sensitivity_factor
```

Where:
- CV = coefficient of variation of scores
- sensitivity_factor captures response to perturbations

---

## 4. Example Domain: Project Simulator

### State Space

| Variable | Range | Semantics |
|----------|-------|-----------|
| progress | 0-100 | % complete |
| debt | 0-100 | Technical debt |
| morale | 0-100 | Team morale |
| budget | 0-∞ | Remaining funds |
| bugs | 0-∞ | Bug count |

### Action Space

| Action | Effect | Risk |
|--------|--------|------|
| SHIP_NOW | +progress, +debt | High |
| REFACTOR | -debt, +morale | Low |
| HIRE | +capacity, -budget | Medium |
| CUT_SCOPE | +progress, -morale | Medium |
| FIX_BUGS | -bugs, +morale | Low |
| DELAY | minimal | Low |

### Stochastic Events

| Event | Probability | Effect |
|-------|-------------|--------|
| team_member_quits | 5% × morale_modifier | -morale, -capacity |
| scope_creep | 8% | -progress, -morale |
| critical_bug | 3% × debt_modifier | +bugs, -morale |
| dependency_breaks | 4% | -progress, +debt |
| morale_boost | 5% | +morale |

### Terminal Conditions

- **Success**: progress ≥ 100
- **Failure**: budget ≤ 0 OR morale ≤ 0 OR (debt ≥ 90 AND bugs ≥ 20)
- **Timeout**: timestep > max_steps

---

## 5. Experimental Results (Illustrative)

### Policy Comparison on Project Simulator

| Policy | Success Rate | Mean Survival | Brittleness | Grade |
|--------|--------------|---------------|-------------|-------|
| Aggressive | 33% | 23 steps | 0.58 | D |
| Conservative | 71% | 42 steps | 0.31 | B |
| Balanced | 64% | 38 steps | 0.35 | B |

### Key Findings

1. **Aggressive policies have high tail risk**: Despite occasionally achieving faster success, 67% of futures result in failure.

2. **Conservative policies sacrifice speed for stability**: Higher survival rate but longer time-to-completion.

3. **Brittleness correlates with failure mode diversity**: Aggressive policies fail for different reasons in different futures.

---

## 6. Distributed Execution via Hivemind

For large-scale evaluations, the framework supports distributed execution using Hivemind DHT:

```
┌─────────────────┐
│  Master Node    │
│  (publishes     │
│   task)         │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌───────┐
│Worker1│ │Worker2│
│(seeds │ │(seeds │
│1-500) │ │501-1000)
└───┬───┘ └───┬───┘
    │         │
    └────┬────┘
         ▼
┌─────────────────┐
│  Aggregate via  │
│  DHT            │
└─────────────────┘
```

This enables:
- Running 10,000+ parallel simulations
- Crowd-sourced robustness testing
- Integration with Gensyn ecosystem

---

## 7. Related Work

| Approach | Focus | Limitation |
|----------|-------|------------|
| LMSYS Arena | Human preference | Single-turn, no trajectory |
| RL Evaluation | Cumulative reward | Requires reward shaping |
| Policy Gradient | Training signal | Optimization, not measurement |
| Monte Carlo Sim | Financial risk | Not AI-specific |
| **This Work** | Outcome distributions | Requires world simulation |

---

## 8. Future Directions

1. **More domains**: Governance, trading, operations, healthcare
2. **LLM policy comparison**: GPT-4 vs Claude vs Llama on robustness
3. **Delphi integration**: Prediction markets on robustness scores
4. **Automated world generation**: Generate simulation environments from descriptions
5. **Causal analysis**: Why does policy X fail more than policy Y?

---

## 9. Conclusion

Single-outcome evaluation is insufficient for AI systems making sequential decisions under uncertainty. Distribution-level evaluation—measuring survival, collapse, regret, and sensitivity across thousands of futures—provides the robustness signal needed for safe deployment.

The Decision Robustness Framework operationalizes this paradigm with:
- Pluggable world simulations
- Flexible policy interfaces
- Comprehensive robustness metrics
- Distributed execution support

**The question is not "Was this decision correct?" but "How does this policy's outcome distribution look across all possible futures?"**

---

## References

1. Kaplan, E. L., & Meier, P. (1958). Nonparametric estimation from incomplete observations.
2. Hanson, R. (2003). Combinatorial information market design.
3. OpenAI. (2023). GPT-4 Technical Report.
4. Anthropic. (2024). Claude 3 Model Card.
5. Gensyn. (2024). RL-Swarm: Decentralized Reinforcement Learning.
