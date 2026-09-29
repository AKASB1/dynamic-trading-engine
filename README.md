# Dynamic Portfolio & Execution Engine

An event-driven trading engine for portfolio allocation, transaction-cost-aware rebalancing, execution simulation, and online decision policies.

**Status:** implementation scaffold.

## Scope

- portfolio state and constraints
- signal / forecast input interface
- risk model
- rolling portfolio optimization
- transaction-cost-aware rebalancing
- order generation and execution simulation
- adaptive policy interface
- paper-trading adapter later
- decision and execution attribution

## Proposed stack

Python 3.12 · NumPy · Polars · CVXPY · OSQP/HiGHS · Pydantic · asyncio

## Decision loop

```text
Market State
    │
    ▼
Forecast / Signal
    │
    ▼
Risk + Constraints
    │
    ▼
Portfolio Optimizer
    │
    ▼
Target Holdings
    │
    ▼
Execution Policy
    │
    ▼
Orders / Fills
    └──────────────► next state
```

## Repository layout

```text
src/dynamic_trading_engine/
  state/
  forecasts/
  risk/
  optimization/
  execution/
  policies/
  broker/
  analytics/
tests/
configs/
simulations/
```

See [IMPLEMENTATION.md](IMPLEMENTATION.md).

## Reference projects

- [cvxgrp/cvxportfolio](https://github.com/cvxgrp/cvxportfolio) — convex portfolio optimization and transaction-cost modeling
- [AI4Finance-Foundation/FinRL](https://github.com/AI4Finance-Foundation/FinRL) — RL-based trading environments and workflows
- [nautechsystems/nautilus_trader](https://github.com/nautechsystems/nautilus_trader) — event-driven trading-system architecture

## License

MIT

## Available now

Importable local primitives cover portfolio state, equal-weight targets, order generation, simple market fills, and turnover. Run `PYTHONPATH=src python -m unittest discover -s tests`. Convex optimization and the full execution simulator are planned.
