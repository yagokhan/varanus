"""
varanus/risk.py — Portfolio Risk Manager (v5.0, Step 8).

V5 changes vs v4:
  - leverage_map: updated to V5 four-tier map (including 5x Power Setup)
    Delegates to model.get_leverage() to avoid duplication.
  - max_portfolio_leverage: 2.5 → 3.5 (Power Setup accommodation)
  - portfolio_stop_pct: aligned to HPO hard constraint (-15%)
  - power_setup_max_concurrent: max 2 Power Setup trades open simultaneously
  - get_position_size() updated: stacks ps_scalar + hv_scalar + leverage
    (delegates to backtest._compute_position_size logic for consistency)
  - is_correlated_to_open(): unchanged (imported by backtest.py)
"""

from __future__ import annotations

import logging

import pandas as pd

from varanus.universe import HIGH_VOL_SUBTIER
from varanus.model    import get_leverage, is_power_setup

logger = logging.getLogger(__name__)

# ── V5 Risk Config ─────────────────────────────────────────────────────────────

RISK_CONFIG: dict = {
    "initial_capital":            5_000.0,

    # V5: raised from 2.5 to accommodate Power Setup (5x lever) entries
    "max_portfolio_leverage":     3.5,
    "max_concurrent_positions":   4,

    # Correlation guard (unchanged from v4)
    "corr_block_threshold":       0.75,
    "corr_lookback_days":         20,

    # V5 position sizing scalars (stackable)
    "position_size_scalar": {
        "standard":      1.00,
        "high_vol":      0.75,    # TAO, ASTR, KITE, ICP
        "power_setup":   1.25,    # confidence >= 0.95 — v5 new
    },

    # V5: aligned to HPO hard constraint (-15%)
    "daily_loss_limit_pct":       0.05,   # Halt all signals if -5% in 24h
    "portfolio_stop_pct":         0.15,   # Halt all signals if -15% from peak

    # V5: cap simultaneous Power Setup positions to bound tail risk
    "power_setup_max_concurrent": 2,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Portfolio health & circuit breaker
# ═══════════════════════════════════════════════════════════════════════════════

def check_portfolio_health(
    equity_curve: pd.Series,
    cfg:          dict = RISK_CONFIG,
) -> dict:
    """
    Evaluate portfolio-level circuit breakers.

    Takes a pd.Series of portfolio equity indexed by strictly monotonic timestamps.

    Returns:
        dict with keys:
            current_equity  — latest equity value
            daily_loss_pct  — % change over last 24h
            drawdown_pct    — % from all-time peak (negative = loss)
            halt_signals    — True if any circuit breaker is tripped
            halt_reason     — human-readable reason string
    """
    if len(equity_curve) == 0:
        return {
            "current_equity": cfg["initial_capital"],
            "daily_loss_pct": 0.0,
            "drawdown_pct":   0.0,
            "halt_signals":   False,
            "halt_reason":    "",
        }

    current      = equity_curve.iloc[-1]
    peak         = equity_curve.cummax().iloc[-1]
    current_time = equity_curve.index[-1]
    cutoff_time  = current_time - pd.Timedelta(days=1)

    past_24h  = equity_curve[equity_curve.index >= cutoff_time]
    day_start = past_24h.iloc[0] if len(past_24h) > 0 else equity_curve.iloc[0]

    daily_loss = (current - day_start) / day_start if day_start > 0 else 0.0
    drawdown   = (current - peak)      / peak      if peak      > 0 else 0.0

    halt   = False
    reason = ""
    if daily_loss <= -cfg["daily_loss_limit_pct"] and drawdown <= -cfg["portfolio_stop_pct"]:
        halt   = True
        reason = "Daily loss AND peak drawdown limits both breached"
    elif daily_loss <= -cfg["daily_loss_limit_pct"]:
        halt   = True
        reason = f"Daily loss limit breached ({daily_loss * 100:.1f}%)"
    elif drawdown <= -cfg["portfolio_stop_pct"]:
        halt   = True
        reason = f"Peak drawdown limit breached ({drawdown * 100:.1f}%)"

    return {
        "current_equity": current,
        "daily_loss_pct": round(daily_loss * 100, 2),
        "drawdown_pct":   round(drawdown   * 100, 2),
        "halt_signals":   halt,
        "halt_reason":    reason,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Position sizing
# ═══════════════════════════════════════════════════════════════════════════════

def get_position_size(
    confidence: float,
    capital:    float,
    asset:      str,
    cfg:        dict = RISK_CONFIG,
) -> float:
    """
    Calculate position size in USD (V5 stacking logic).

    Scalars applied in sequence:
      1. base = capital / max_concurrent_positions
      2. × power_setup_size_scalar (1.25) if confidence >= 0.95
      3. × high_vol_size_scalar (0.75) if asset in HIGH_VOL_SUBTIER
      4. × leverage (1x / 2x / 3x / 5x from V5 confidence tier)

    Both scalars can apply simultaneously (e.g. Power Setup + high-vol → 1.25 × 0.75 × 5x).
    """
    lev      = get_leverage(confidence)
    ps_flag  = is_power_setup(confidence)
    hv_flag  = asset in HIGH_VOL_SUBTIER

    scalars  = cfg["position_size_scalar"]
    base     = capital / cfg["max_concurrent_positions"]
    ps_mult  = scalars["power_setup"] if ps_flag else 1.0
    hv_mult  = scalars["high_vol"]    if hv_flag else 1.0

    return base * ps_mult * hv_mult * lev


# ═══════════════════════════════════════════════════════════════════════════════
# Portfolio leverage helpers
# ═══════════════════════════════════════════════════════════════════════════════

def compute_portfolio_leverage(open_trades: dict, capital: float) -> float:
    """
    Return current portfolio leverage = total notional / capital.
    Returns 0.0 when no positions are open or capital is zero.
    """
    if capital <= 0 or not open_trades:
        return 0.0
    total_notional = sum(t["position_usd"] for t in open_trades.values())
    return total_notional / capital


def would_breach_leverage(
    open_trades: dict,
    capital:     float,
    new_sig:     dict,
    cfg:         dict = RISK_CONFIG,
) -> bool:
    """Return True if adding the new signal would exceed max_portfolio_leverage."""
    if capital <= 0:
        return True
    new_size         = get_position_size(
        new_sig.get("confidence", 0.0), capital, new_sig.get("asset", ""), cfg
    )
    current_notional = sum(t["position_usd"] for t in open_trades.values())
    return (current_notional + new_size) / capital > cfg["max_portfolio_leverage"]


def count_power_setups(open_trades: dict) -> int:
    """Return the number of currently open Power Setup (5x) positions."""
    return sum(1 for t in open_trades.values() if t.get("power_setup", False))


def would_breach_power_setup_cap(
    open_trades: dict,
    confidence:  float,
    cfg:         dict = RISK_CONFIG,
) -> bool:
    """
    True if the candidate trade is a Power Setup AND the concurrent Power Setup
    limit is already reached.
    """
    if not is_power_setup(confidence):
        return False
    return count_power_setups(open_trades) >= cfg["power_setup_max_concurrent"]


# ═══════════════════════════════════════════════════════════════════════════════
# Correlation guard (imported by backtest.py)
# ═══════════════════════════════════════════════════════════════════════════════

def is_correlated_to_open(
    asset:       str,
    open_trades: dict,
    data:        dict,
    cfg:         dict = RISK_CONFIG,
) -> bool:
    """
    Block entry if the candidate asset's recent returns are too highly correlated
    with any currently open position (|corr| >= corr_block_threshold).

    Requires at least 20 overlapping bars to compute; skips if data is insufficient.
    """
    if not open_trades or asset not in data:
        return False

    lookback      = cfg["corr_lookback_days"] * 6   # 6 × 4h bars per day
    asset_returns = data[asset]["close"].pct_change().dropna().tail(lookback)

    for open_asset in open_trades:
        if open_asset not in data or open_asset == asset:
            continue
        open_returns = data[open_asset]["close"].pct_change().dropna().tail(lookback)
        combined     = pd.concat([asset_returns, open_returns], axis=1, join="inner").dropna()
        if len(combined) < 20:
            continue
        corr = combined.iloc[:, 0].corr(combined.iloc[:, 1])
        if pd.notna(corr) and abs(corr) >= cfg["corr_block_threshold"]:
            logger.debug(
                "Correlation block: %s vs %s corr=%.3f (thresh=%.2f)",
                asset, open_asset, corr, cfg["corr_block_threshold"],
            )
            return True

    return False
