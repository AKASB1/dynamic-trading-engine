# Implementation plan

## Boundary with quant-research-engine

`quant-research-engine` produces validated signals and research artifacts. This project starts from a stream of market states and signals and focuses on decisions, rebalancing, and execution.

## Portfolio optimizer

Start with a convex single-period problem:

```text
maximize    expected_return
          - risk_penalty
          - transaction_cost
subject to  budget
            position bounds
            turnover bound
            optional sector/factor exposure bounds
```

Expose solver choice through an adapter. The default path should work with open-source solvers; commercial solvers can be optional.

## Execution simulator

Model:

- market and limit orders
- configurable spread
- slippage
- latency
- partial fills
- participation cap

## Policy interface

A policy maps current state and target portfolio to orders. Implement simple TWAP/POV baselines before adaptive policies.

## Delivery order

1. portfolio state model
2. convex optimizer
3. order/fill event model
4. execution simulator
5. rolling rebalance loop
6. risk and transaction-cost models
7. baseline execution policies
8. adaptive policy interface
9. paper-trading adapter
