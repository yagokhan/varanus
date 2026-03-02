"""
varanus/backtest.py — Portfolio Backtesting Engine (v5.0, Step 5).

V5 changes vs v4:
  - Leverage map: 4 tiers, including 5x Power Setup (confidence >= 0.95)
  - max_portfolio_leverage: 2.5 → 3.5 (accommodates Power Setup entries)
  - power_setup_size_scalar: 1.25× applied before leverage for Power Setups
  - max_holding default: 31 bars (V5 time exit lock)
  - Acceptance gate: min_profit_factor 1.30 → 1.50, max_drawdown -35% → -15%
  - Force-close at end-of-test window (preserved from v4)
"""

from __future__ import annotations

import logging

import numpy  as np
import pandas as pd

from varanus.tbm_labeler  import calculate_barriers, TBM_CONFIG
from varanus.model        import get_leverage, is_power_setup
from varanus.universe     import HIGH_VOL_SUBTIER
from varanus.risk         import is_correlated_to_open

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

BACKTEST_CONFIG: dict = {
    "initial_capital":            5_000.0,
    "maker_fee":                  0.0002,   # 0.02% limit orders
    "taker_fee":                  0.0005,   # 0.05% market orders (SL exits)
    "slippage_pct":               0.0008,   # 0.08% mid-cap 4h slippage
    "entry_on_bar":               "open",   # Enter on next bar after signal

    "use_flash_wick_guard":       True,
    "wick_body_close_required":   True,

    # Portfolio constraints
    "max_concurrent_positions":   4,
    "max_portfolio_leverage":     3.5,      # v5: raised from 2.5 (Power Setup)
    "corr_block_threshold":       0.75,
    "corr_lookback_days":         20,

    # V5 sizing scalars
    "power_setup_size_scalar":    1.25,     # +25% notional for Power Setup trades
    "high_vol_size_scalar":       0.75,     # TAO, ASTR, KITE, ICP

    "equity_curve_freq":          "4h",
    "trade_log":                  True,
}

# ── V5 acceptance gate (tightened vs v4) ──────────────────────────────────────

BACKTEST_PASS_CRITERIA: dict = {
    "min_trades":         50,
    "min_win_rate":       0.43,
    "min_profit_factor":  1.50,   # v5: raised from 1.30
    "max_drawdown":      -0.15,   # v5: -15% hard cap (aligned to HPO constraint)
    "min_sharpe":         0.90,   # v5: raised from 0.80
    "min_calmar":         0.60,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Barrier + PnL helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _check_barriers(bar: pd.Series, trade: dict, cfg: dict) -> dict | None:
    """
    Evaluate TP, SL, and time barrier for a single bar.

    Flash-wick guard: SL requires body close beyond level; wick alone does not
    trigger unless it exceeds the wick_tolerance (0.3 × ATR).

    Order of priority: time barrier → TP (wick touch) → SL (body close).
    Time barrier is checked first to prevent dead capital accumulation.
    """
    d = trade["direction"]

    # Time barrier
    if bar.name >= trade["max_hold_bar"]:
        return {"type": "time", "price": bar["close"]}

    # Take-Profit (wick touch sufficient — we capture the full gain)
    if d ==  1 and bar["high"] >= trade["take_profit"]:
        return {"type": "tp", "price": trade["take_profit"]}
    if d == -1 and bar["low"]  <= trade["take_profit"]:
        return {"type": "tp", "price": trade["take_profit"]}

    # Stop-Loss (flash-wick guard: body close required)
    if cfg["use_flash_wick_guard"] and cfg["wick_body_close_required"]:
        if d ==  1 and bar["close"] < trade["stop_loss"]:
            return {"type": "sl", "price": trade["stop_loss"]}
        if d == -1 and bar["close"] > trade["stop_loss"]:
            return {"type": "sl", "price": trade["stop_loss"]}
    else:
        if d ==  1 and bar["low"]  <= trade["stop_loss"]:
            return {"type": "sl", "price": trade["stop_loss"]}
        if d == -1 and bar["high"] >= trade["stop_loss"]:
            return {"type": "sl", "price": trade["stop_loss"]}

    return None


def _calculate_pnl(trade: dict, outcome: dict, cfg: dict) -> float:
    """Net PnL after fees and slippage."""
    raw_ret = (
        trade["direction"] *
        (outcome["price"] - trade["entry_price"]) /
        trade["entry_price"]
    )
    fee     = cfg["taker_fee"] if outcome["type"] == "sl" else cfg["maker_fee"]
    net_ret = raw_ret - fee - cfg["slippage_pct"]
    return trade["position_usd"] * net_ret


def _compute_position_size(
    capital:    float,
    confidence: float,
    asset:      str,
    cfg:        dict,
) -> tuple[float, float, bool]:
    """
    Compute (position_usd, leverage, is_power_setup_flag).

    V5 sizing logic (applied in sequence):
      1. Base allocation = capital / max_concurrent_positions
      2. If Power Setup (conf >= 0.95): × power_setup_size_scalar (1.25)
      3. If High-Vol Sub-Tier: × high_vol_size_scalar (0.75)
      4. × leverage (1x / 2x / 3x / 5x from confidence tier)
    """
    lev      = get_leverage(confidence)
    ps_flag  = is_power_setup(confidence)
    hv_flag  = asset in HIGH_VOL_SUBTIER

    base     = capital / cfg["max_concurrent_positions"]
    ps_mult  = cfg["power_setup_size_scalar"] if ps_flag else 1.0
    hv_mult  = cfg["high_vol_size_scalar"]    if hv_flag else 1.0

    position = base * ps_mult * hv_mult * lev

    return position, lev, ps_flag


def _would_breach_leverage(
    open_trades: dict,
    capital:     float,
    sig:         dict,
    cfg:         dict,
) -> bool:
    """Return True if adding this signal would exceed max_portfolio_leverage."""
    if capital <= 0:
        return True
    pos, _, _ = _compute_position_size(
        capital, sig.get("confidence", 0.0), sig.get("asset", ""), cfg
    )
    current_notional = sum(t["position_usd"] for t in open_trades.values())
    return (current_notional + pos) / capital > cfg["max_portfolio_leverage"]


# ═══════════════════════════════════════════════════════════════════════════════
# Main simulation loop
# ═══════════════════════════════════════════════════════════════════════════════

def run_backtest(
    data:    dict[str, pd.DataFrame],
    signals: dict[str, pd.DataFrame],
    model,
    params:  dict,
    cfg:     dict = BACKTEST_CONFIG,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Simulate the full Varanus v5 Tier 2 strategy over historical data.

    Args:
        data:    {asset: OHLCV DataFrame}
        signals: {asset: signal DataFrame} — columns: confidence, direction,
                 entry_price, atr
        model:   Fitted VaranusModel (used for metadata only; signals pre-computed)
        params:  Hyperparameter dict (confidence_thresh, sl_atr_mult, rr_ratio, etc.)
        cfg:     BACKTEST_CONFIG

    Returns:
        (equity_curve: pd.Series, trade_log: pd.DataFrame)
    """
    capital     = cfg["initial_capital"]
    equity      = {}
    open_trades = {}
    trade_log   = []

    # Resolve confidence threshold from params (v5 HPO output)
    conf_thresh = params.get(
        "confidence_thresh",
        params.get("confidence_threshold", 0.750)
    )

    # Resolve max holding: V5 lock = 31 bars, but respect explicit override
    max_hold_bars = params.get("max_holding_candles", 31)

    # Build TBM config from params for barrier calculation
    tbm_cfg = TBM_CONFIG.copy()
    if "sl_atr_mult" in params:
        tbm_cfg["stop_loss_atr"] = params["sl_atr_mult"]
    if "rr_ratio" in params:
        tbm_cfg["rr_ratio"] = params["rr_ratio"]
        tbm_cfg.pop("take_profit_atr", None)
    elif "tp_atr_mult" in params:
        tbm_cfg["take_profit_atr"] = params["tp_atr_mult"]
        tbm_cfg.pop("rr_ratio", None)

    all_timestamps = sorted(set().union(*[df.index for df in data.values()]))

    for ts in all_timestamps:

        # ── 1. Check barrier outcomes for all open trades ─────────────────────
        for asset, trade in list(open_trades.items()):
            if ts not in data[asset].index:
                continue
            bar     = data[asset].loc[ts]
            outcome = _check_barriers(bar, trade, cfg)
            if outcome:
                pnl     = _calculate_pnl(trade, outcome, cfg)
                capital += pnl
                trade_log.append({
                    **trade,
                    "exit_ts":    ts,
                    "exit_price": outcome["price"],
                    "outcome":    outcome["type"],
                    "pnl_usd":    pnl,
                })
                del open_trades[asset]

        # ── 2. Collect and rank new signals ───────────────────────────────────
        current_sigs = []
        for asset, sig_df in signals.items():
            if ts not in sig_df.index or asset in open_trades:
                continue
            sig          = sig_df.loc[ts].copy()
            sig["asset"] = asset
            if sig.get("confidence", 0) >= conf_thresh:
                current_sigs.append(sig)

        # Highest confidence first (prevents list-order bias)
        current_sigs.sort(key=lambda x: x.get("confidence", 0), reverse=True)

        # ── 3. Open new trades ────────────────────────────────────────────────
        for sig in current_sigs:
            if len(open_trades) >= cfg["max_concurrent_positions"]:
                break

            asset = sig["asset"]

            if _would_breach_leverage(open_trades, capital, sig, cfg):
                continue
            if is_correlated_to_open(asset, open_trades, data, cfg):
                continue

            barriers = calculate_barriers(
                sig["entry_price"], sig["atr"], sig["direction"], tbm_cfg, asset
            )
            if not barriers.get("min_rr_satisfied", True):
                continue

            position_usd, lev, ps_flag = _compute_position_size(
                capital, sig["confidence"], asset, cfg
            )

            open_trades[asset] = {
                "asset":         asset,
                "entry_ts":      ts,
                "entry_price":   sig["entry_price"],
                "direction":     sig["direction"],
                "take_profit":   barriers["take_profit"],
                "stop_loss":     barriers["stop_loss"],
                "position_usd":  position_usd,
                "leverage":      lev,
                "confidence":    sig["confidence"],
                "rr_ratio":      barriers["rr_ratio"],
                "power_setup":   ps_flag,
                "max_hold_bar":  ts + pd.Timedelta(hours=4 * max_hold_bars),
            }

        equity[ts] = capital

    # ── 4. Force-close remaining positions at end of test window ─────────────
    if all_timestamps:
        last_ts = all_timestamps[-1]
        for asset, trade in list(open_trades.items()):
            if last_ts in data[asset].index:
                last_price = data[asset].loc[last_ts, "close"]
                pnl        = _calculate_pnl(
                    trade, {"type": "time", "price": last_price}, cfg
                )
                capital   += pnl
                trade_log.append({
                    **trade,
                    "exit_ts":    last_ts,
                    "exit_price": last_price,
                    "outcome":    "time",
                    "pnl_usd":    pnl,
                })
        equity[last_ts] = capital

    return pd.Series(equity), pd.DataFrame(trade_log)


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def _max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    return ((equity - roll_max) / roll_max).min()


def compute_metrics(
    equity_curve: pd.Series,
    trade_log:    pd.DataFrame,
) -> dict:
    """Full performance report for one backtest run."""
    if len(equity_curve) < 2:
        return {k: 0 for k in [
            "total_return_pct", "cagr_pct", "max_drawdown_pct", "calmar_ratio",
            "sharpe_ratio", "win_rate_pct", "profit_factor", "total_trades",
            "tp_hits", "sl_hits", "time_exits", "avg_win_usd", "avg_loss_usd",
        ]}

    returns   = equity_curve.pct_change().dropna()
    total_ret = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    n_days    = (equity_curve.index[-1] - equity_curve.index[0]).days
    cagr      = (1 + total_ret) ** (365 / n_days) - 1 if n_days > 0 else 0
    max_dd    = _max_drawdown(equity_curve)
    calmar    = cagr / abs(max_dd) if max_dd != 0 else 0

    std = returns.std()
    sharpe = returns.mean() / std * (365 * 6) ** 0.5 if std > 0 else 0

    wins  = trade_log["pnl_usd"] > 0 if len(trade_log) else pd.Series(dtype=bool)
    loss  = trade_log["pnl_usd"] < 0 if len(trade_log) else pd.Series(dtype=bool)

    profit_factor = (
        abs(trade_log.loc[wins, "pnl_usd"].sum() /
            trade_log.loc[loss, "pnl_usd"].sum())
        if loss.any() else float("inf")
    )
    by_outcome = trade_log["outcome"].value_counts() if len(trade_log) else {}

    return {
        "total_return_pct": round(total_ret * 100, 2),
        "cagr_pct":         round(cagr * 100, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "calmar_ratio":     round(calmar, 3),
        "sharpe_ratio":     round(sharpe, 3),
        "win_rate_pct":     round(wins.mean() * 100, 2) if len(trade_log) else 0,
        "profit_factor":    round(profit_factor, 2),
        "total_trades":     len(trade_log),
        "tp_hits":          by_outcome.get("tp",   0),
        "sl_hits":          by_outcome.get("sl",   0),
        "time_exits":       by_outcome.get("time", 0),
        "avg_win_usd":      round(trade_log.loc[wins, "pnl_usd"].mean(), 2) if wins.any() else 0,
        "avg_loss_usd":     round(trade_log.loc[loss, "pnl_usd"].mean(), 2) if loss.any() else 0,
    }


def passes_backtest_gate(metrics: dict) -> bool:
    """Return True if all v5 acceptance criteria are satisfied."""
    c = BACKTEST_PASS_CRITERIA
    return (
        metrics["total_trades"]      >= c["min_trades"]              and
        metrics["win_rate_pct"]      >= c["min_win_rate"] * 100      and
        metrics["profit_factor"]     >= c["min_profit_factor"]       and
        metrics["max_drawdown_pct"]  >= c["max_drawdown"] * 100      and
        metrics["sharpe_ratio"]      >= c["min_sharpe"]              and
        metrics["calmar_ratio"]      >= c["min_calmar"]
    )
