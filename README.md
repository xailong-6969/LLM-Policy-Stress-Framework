# LLM Policy Stress Framework

Stress testing LLM decision policies across many possible futures to measure robustness under uncertainty and delayed consequences.

## What this is
An experimental framework for evaluating AI decision policies by running them through large numbers of parallel simulated environments and analyzing outcome distributions.

## What this is NOT
- Not an LLM benchmark
- Not a prompt comparison tool
- Not a prediction market
- Not an LLM arena

## Core idea
Single outcomes lie. Distributions don’t.

## Status
Early stage research framework. Currently includes:
- World simulation engine (in progress)
- Policy interface (rule-based initially)
- Swarmed evaluation runner
- Robustness metrics (failure probability, collapse time)
