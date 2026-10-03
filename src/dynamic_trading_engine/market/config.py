"""Market configuration (assumed parameters; every number is documented in docs/data.md)."""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass

from dynamic_trading_engine.contracts.canonical import canonical_dict, config_hash


@dataclass(frozen=True)
class ShiftConfig:
    at_bar: int = 630
    vol_mult: float = 1.0
    corr_mult: float = 1.0
    ic_mult: float = 1.0
    mu_shift_annual: float = 0.0


@dataclass(frozen=True)
class MarketConfig:
    name: str = "base"
    n_instruments: int = 30
    n_bars: int = 1260
    calendar: str = "equity_daily"
    ppy: int = 252
    start: str = "2018-01-02"
    warmup_bars: int = 260
    # factors
    factors: bool = True
    market_vol_annual: float = 0.16
    premium_annual: float = 0.05
    n_sectors: int = 3
    sector_vol_annual: float = 0.10
    beta_lo: float = 0.6
    beta_hi: float = 1.4
    # idiosyncratic part and tails
    idio_vol_lo: float = 0.15
    idio_vol_hi: float = 0.35
    equal_idio_vol: bool = False
    tails: str = "gaussian"  # or "student_t" (Tier 2, assumed tail model)
    t_dof: float = 5.0
    jump_rate_annual: float = 0.0  # Tier 2: idiosyncratic jumps per instrument and year
    jump_size: float = 4.0  # jump standard deviation in units of the bar's idiosyncratic sd
    gap_share: float = 0.25
    # regime
    regime: bool = True
    stress_vol_mult: float = 2.0
    p_calm_to_stress: float = 0.01
    p_stress_to_calm: float = 0.05
    # planted predictability
    ic: float = 0.02
    persistence: float = 0.9
    observable_noise: float = 1.0
    # liquidity and prices
    adv_lo: float = 500_000.0
    adv_hi: float = 5_000_000.0
    price_lo: float = 20.0
    price_hi: float = 200.0
    volume_ar: float = 0.7
    volume_sd: float = 0.3
    lot_size: float = 1.0
    # events
    events: bool = True
    delist_hazard_annual: float = 0.01
    delist_return_lo: float = -0.6
    delist_return_hi: float = 0.1
    split_rate_annual: float = 0.03
    dividend_yield_annual: float = 0.015
    late_listing_share: float = 0.1
    # distribution shift (None: no shift)
    shift: ShiftConfig | None = None

    def __post_init__(self):
        if self.calendar != "equity_daily":
            raise ValueError("only the equity_daily calendar is generated")
        if self.tails not in ("gaussian", "student_t"):
            raise ValueError("tails must be gaussian or student_t")
        if not (0 <= self.gap_share < 1 and -1 < self.ic < 1 and 0 <= self.persistence < 1):
            raise ValueError("gap_share, ic, persistence out of range")
        if self.n_instruments < 1 or self.n_bars < 2 or self.warmup_bars >= self.n_bars:
            raise ValueError("bad sizes")

    def generator_params(self) -> dict:
        return canonical_dict(self)

    def hash(self) -> str:
        return config_hash(self)


def market_config_from_dict(d: dict) -> MarketConfig:
    d = dict(d)
    if d.get("shift") is not None:
        d["shift"] = ShiftConfig(**d["shift"])
    names = {f.name for f in dataclasses.fields(MarketConfig)}
    unknown = sorted(set(d) - names)
    if unknown:
        raise ValueError(f"unknown market configuration fields: {unknown}")
    return MarketConfig(**d)


CONFIG_DIR = os.path.join("configs", "market")


def load_market_config(name_or_path: str, root: str | None = None) -> MarketConfig:
    """Load ``configs/market/<name>.json`` (relative to ``root`` or the repository root)."""
    path = name_or_path
    if not path.endswith(".json"):
        base = root if root is not None else repo_root()
        path = os.path.join(base, CONFIG_DIR, name_or_path + ".json")
    with open(path, encoding="utf-8") as fh:
        return market_config_from_dict(json.load(fh))


def repo_root() -> str:
    """The repository root: the nearest parent of this file with a ``configs`` directory, or
    the current directory (an installed package run from a checkout)."""
    env = os.environ.get("DTE_ROOT")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        if os.path.isdir(os.path.join(here, "configs", "market")):
            return here
        here = os.path.dirname(here)
    return os.getcwd()
