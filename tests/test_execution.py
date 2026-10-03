"""Checks 10 (Almgren-Chriss) and 11 (execution simulator), and the simulator leg of check 3."""

import math

import numpy as np
import pytest

from dynamic_trading_engine.contracts.costs import CostConfig, fill_cost
from dynamic_trading_engine.execution.ac import (
    ACParams,
    ac_closed_form,
    ac_linear_solve,
    ac_qp,
    expected_cost_variance,
    twap_schedule,
)
from dynamic_trading_engine.execution.simulator import (
    ExecConfig,
    ExecPolicy,
    ParentOrder,
    ac_policy,
    draws,
    flat_profile,
    market_policy,
    pov_policy,
    simulate,
    simulate_path_reference,
    twap_policy,
    u_profile,
    vwap_policy,
)
from dynamic_trading_engine.rng import stream

KA1 = ACParams(X=1e6, T=5, N=5, sigma=0.95, gamma=2.5e-7, eta=2.5e-6, lam=1e-6)
KA2 = ACParams(X=1, T=1, N=4, sigma=0.3, gamma=0.1, eta=0.5, lam=2)
ANS1 = dict(
    kappa=0.607076163247063,
    x=[541955.554373922, 289854.219409935, 147897.487821723, 62141.801605766],
    E=848726.986303794,
    V=364128572058.141,
    obj=1212855.55836193,
)
ANS2 = dict(
    kappa=0.607060859211631,
    x=[0.730481151084272, 0.477819559501258, 0.236184573137504],
    E=0.538765372560291,
    V=0.0183981914126893,
    obj=0.575561755385669,
)


def _rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


@pytest.mark.parametrize("p,ans", [(KA1, ANS1), (KA2, ANS2)])
def test_ac_known_answers(p, ans):
    s = ac_closed_form(p)
    assert _rel(s.kappa, ans["kappa"]) <= 1e-9
    for got, want in zip(s.x[1:-1], ans["x"]):
        assert _rel(got, want) <= 1e-9
    assert _rel(s.E, ans["E"]) <= 1e-9
    assert _rel(s.V, ans["V"]) <= 1e-9
    assert _rel(s.E + p.lam * s.V, ans["obj"]) <= 1e-9


@pytest.mark.parametrize("p", [KA1, KA2])
def test_ac_qp_and_linear_solve_agree(p):
    s = ac_closed_form(p)
    for x in (ac_qp(p), ac_linear_solve(p)):
        assert np.max(np.abs(x - s.x)) <= 1e-6 * p.X
        E, V = expected_cost_variance(p, x)
        assert _rel(E + p.lam * V, s.E + p.lam * s.V) <= 1e-6


def test_ac_limits_and_front_loading():
    p0 = ACParams(1.0, 1.0, 10, 0.3, 0.1, 0.5, 1e-14)
    s = ac_closed_form(p0)
    assert np.max(np.abs(s.x - twap_schedule(p0))) <= 1e-5
    pz = ACParams(1.0, 1.0, 10, 0.3, 0.1, 0.5, 0.0)
    sz = ac_closed_form(pz)
    assert sz.kappa == 0.0 and np.array_equal(sz.x, twap_schedule(pz))
    for lam in (0.5, 2.0, 8.0):
        p = ACParams(1.0, 1.0, 10, 0.3, 0.1, 0.5, lam)
        s = ac_closed_form(p)
        Et, Vt = expected_cost_variance(p, twap_schedule(p))
        assert Et + lam * Vt > s.E + lam * s.V
    firsts = [
        ac_closed_form(ACParams(1.0, 1.0, 10, 0.3, 0.1, 0.5, lam)).n[0] for lam in (0.1, 1, 10, 100)
    ]
    assert all(a < b for a, b in zip(firsts, firsts[1:]))


AC_CFG = ExecConfig(
    impact="direct",
    eta_ac=KA2.eta,
    gamma_ac=KA2.gamma,
    half_spread_bps=0.0,
    commission_bps=0.0,
    participation_cap=None,
    volume_noise_sd=0.0,
)


def _ac_order(p: ACParams, side=1):
    # sigma_bar * S0 = sigma_ac; S0 = 100 so that sigma_bar = sigma / 100
    S0 = 100.0
    return ParentOrder(
        side, p.X, S0, p.sigma / S0, 1e9, tuple([p.T / p.N] * p.N), flat_profile(p.N, 1e9, p.T)
    )


def test_simulator_reproduces_ac_mean_and_variance():
    p = KA2
    order = _ac_order(p)
    pol = ac_policy(order, p.lam, p.eta, p.gamma)
    xi, zv = draws(7, 20000, p.N)
    out = simulate(order, pol, AC_CFG, xi, zv)
    s = ac_closed_form(p)
    IS = out["IS"]
    se = IS.std(ddof=1) / math.sqrt(len(IS))
    assert abs(IS.mean() - s.E) <= 3 * se
    assert abs(IS.var(ddof=1) / s.V - 1) <= 0.05
    tse = out["timing"].std(ddof=1) / math.sqrt(len(IS))
    assert abs(out["timing"].mean()) <= 3 * tse
    n = s.n
    tau = p.T / p.N
    perm = p.gamma * (p.X**2 - np.sum(n**2)) / 2
    temp = p.eta / tau * np.sum(n**2)
    assert np.max(np.abs(out["permanent"] - perm)) <= 1e-9 * abs(perm)
    assert np.max(np.abs(out["temporary"] - temp)) <= 1e-9 * abs(temp)


def _order(g, K=10, side=1, Q=None):
    adv = float(g.uniform(2e5, 2e6))
    T = 1.0
    prof = u_profile(K, adv, T) if g.random() < 0.5 else flat_profile(K, adv, T)
    Q = Q if Q is not None else float(adv * g.uniform(0.005, 0.2))
    return ParentOrder(
        side,
        Q,
        float(g.uniform(20, 200)),
        float(g.uniform(0.01, 0.03)),
        adv,
        tuple([T / K] * K),
        prof,
    )


def test_twap_vwap_slices_and_pov_rate():
    g = stream(1, "test.exec.slices")
    order = _order(g, Q=1e4)
    cfg = ExecConfig(participation_cap=None, volume_noise_sd=0.3)
    xi, zv = draws(1, 50, order.K)
    tw = simulate(order, twap_policy(order), cfg, xi, zv, record=True)
    assert np.allclose(tw["fills"], order.Q / order.K)
    vw = simulate(order, vwap_policy(order), cfg, xi, zv, record=True)
    v = np.asarray(order.profile)
    assert np.allclose(vw["fills"], order.Q * v / v.sum())  # the expected profile, not the realized
    rate = 0.05
    pv = simulate(order, pov_policy(rate), cfg, xi, zv, record=True)
    V = np.asarray(order.profile) * np.exp(0.3 * zv - 0.5 * 0.09)
    cum = np.cumsum(pv["fills"], axis=1)
    for i in range(50):
        for k in range(order.K):
            before = cum[i, k - 1] if k else 0.0
            want = min(rate * V[i, k], order.Q - before)
            assert pv["fills"][i, k] == pytest.approx(want, rel=1e-12, abs=1e-9)


def test_cap_never_exceeded_and_latency():
    g = stream(2, "test.exec.cap")
    for _ in range(20):
        order = _order(g, Q=None)
        order = ParentOrder(
            order.side,
            order.adv * 0.3,
            order.S0,
            order.sigma_bar,
            order.adv,
            order.tau,
            order.profile,
        )
        cap = float(g.uniform(0.02, 0.2))
        d = int(g.integers(0, 3))
        cfg = ExecConfig(participation_cap=cap, latency=d)
        xi, zv = draws(int(g.integers(1, 1000)), 30, order.K)
        V = np.asarray(order.profile) * np.exp(0.3 * zv - 0.5 * 0.09)
        for pol in (market_policy(order), twap_policy(order), vwap_policy(order), pov_policy(0.3)):
            out = simulate(order, pol, cfg, xi, zv, record=True)
            assert np.all(out["fills"] <= cap * V * (1 + 1e-12))
            first = np.argmax(out["fills"] > 0, axis=1)
            assert np.all(first == d)
            assert np.all(out["completion"] <= 1 + 1e-12)


def test_partial_fill_carry_and_opportunity():
    g = stream(3, "test.exec.carry")
    order = _order(g, Q=None)
    order = ParentOrder(
        1, order.adv * 0.5, order.S0, order.sigma_bar, order.adv, order.tau, order.profile
    )
    xi, zv = draws(3, 40, order.K)
    with_carry = simulate(
        order, market_policy(order), ExecConfig(participation_cap=0.1), xi, zv, record=True
    )
    no_carry = simulate(
        order,
        market_policy(order),
        ExecConfig(participation_cap=0.1, carry=False),
        xi,
        zv,
        record=True,
    )
    assert np.all(no_carry["fills"][:, 1:] == 0)  # without carry the shortfall is dropped
    assert np.all(with_carry["completion"] >= no_carry["completion"])
    unf = 1 - with_carry["completion"]
    assert np.all((unf > 1e-12) <= (np.abs(with_carry["opportunity"]) > 0))
    assert np.all(with_carry["completion"] < 1)  # 0.1 * total volume < 0.5 ADV


def test_perturbation_changes_no_earlier_decision():
    """Decisions at interval k depend only on what is known at its start: changing the realized
    prices and volumes of intervals k and later changes no decision up to k."""
    g = stream(4, "test.exec.perturb")
    order = _order(g)
    for pol in (
        market_policy(order),
        twap_policy(order),
        vwap_policy(order),
        ac_policy(order, 1e-6, 1e-6, 0.0),
    ):
        xi, zv = draws(4, 10, order.K)
        a = simulate(order, pol, ExecConfig(), xi, zv, record=True)
        for k in range(order.K):
            xi2, zv2 = xi.copy(), zv.copy()
            xi2[:, k:] = g.standard_normal(xi2[:, k:].shape)
            zv2[:, k:] = g.standard_normal(zv2[:, k:].shape)
            b = simulate(order, pol, ExecConfig(), xi2, zv2, record=True)
            assert np.array_equal(a["decided"][:, : k + 1], b["decided"][:, : k + 1])
            assert np.array_equal(a["fills"][:, :k], b["fills"][:, :k])


def test_decomposition_exact_on_random_runs():
    """1000 random runs (100 configurations x 10 paths): all policies, buys and sells, both
    impact models, spread, commission, latency, cap, volume noise, permanent impact; the
    components sum to IS within 1e-12 * Q * S0, and the vectorized simulator equals the
    path-by-path reference."""
    g = stream(5, "test.exec.decomp")
    runs = 0
    for c in range(100):
        side = 1 if g.random() < 0.5 else -1
        order = _order(g, K=int(g.integers(1, 15)), side=side)
        cfg = ExecConfig(
            impact=("sqrt", "linear")[int(g.integers(0, 2))],
            y=float(g.uniform(0.2, 1.0)),
            gamma_ac=float(g.uniform(0, 1e-5)),
            half_spread_bps=float(g.uniform(0, 5)),
            commission_bps=float(g.uniform(0, 2)),
            commission_per_share=float(g.uniform(0, 0.01)),
            min_commission=float(g.uniform(0, 2)),
            participation_cap=None if g.random() < 0.3 else float(g.uniform(0.02, 0.3)),
            volume_noise_sd=float(g.uniform(0, 0.6)),
            carry=bool(g.random() < 0.7),
            latency=int(g.integers(0, 3)),
        )
        pols = [
            market_policy(order),
            twap_policy(order),
            vwap_policy(order),
            pov_policy(float(g.uniform(0.02, 0.3))),
        ]
        pols.append(ac_policy(order, float(10 ** g.uniform(-8, -4)), 1e-6, 0.0))
        pol = pols[c % len(pols)]
        xi, zv = draws(c, 10, order.K)
        out = simulate(order, pol, cfg, xi, zv)
        total = sum(
            out[k]
            for k in ("timing", "permanent", "spread", "temporary", "opportunity", "commission")
        )
        assert np.max(np.abs(total - out["IS"])) <= 1e-12 * order.Q * order.S0
        for i in range(10):
            ref = simulate_path_reference(order, pol, cfg, xi[i], zv[i])
            assert abs(ref["IS"] - out["IS"][i]) <= 1e-12 * order.Q * order.S0
            runs += 1
    assert runs >= 1000


def test_cost_parity_simulator_leg():
    """The market policy with one interval of one bar, beta 0.5, no noise, no cap equals the
    contract's fill cost at the reference price (1e-10 relative)."""
    g = stream(6, "test.exec.parity")
    costs = CostConfig(commission_per_share=0.003)
    for _ in range(200):
        P = float(g.uniform(5, 500))
        sig = float(g.uniform(0.005, 0.05))
        V = float(10 ** g.uniform(4, 7))
        q = float(10 ** g.uniform(0, 5))
        side = 1 if g.random() < 0.5 else -1
        order = ParentOrder(side, q, P, sig, V, (1.0,), (V,))
        cfg = ExecConfig(
            impact="sqrt",
            y=0.5,
            half_spread_bps=2.0,
            commission_bps=1.0,
            commission_per_share=0.003,
            participation_cap=None,
            volume_noise_sd=0.0,
        )
        out = simulate(order, market_policy(order), cfg, np.zeros((1, 1)), np.zeros((1, 1)))
        fc = fill_cost(side * q, P, sig, V, costs)
        want = fc.spread_cost + fc.impact_cost + fc.commission
        assert abs(out["IS"][0] - want) <= 1e-10 * want


def test_policy_types():
    order = _order(stream(7, "x"))
    assert isinstance(market_policy(order), ExecPolicy)
    assert pov_policy(0.1).rate == 0.1
