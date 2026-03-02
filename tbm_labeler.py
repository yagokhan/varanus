"""
varanus/tbm_labeler.py — Triple-Barrier Labeling (v5.0, Step 3).

V5 changes vs v4:
  - max_holding_candles: 30 → 31 (LOCKED — not an Optuna parameter)
  - TP derivation: if params contains 'rr_ratio' + 'sl_atr_mult',
    TP = sl_atr_mult × rr_ratio × ATR (R:R exact by construction)
  - High-vol overrides use additive delta on top of Optuna-chosen base

Label encoding:
     1  →  TP hit first   (win)
     0  →  Time barrier at bar 31 (neutral)
    -1  →  SL hit first   (loss)

Flash-Wick Guard (preserved from v4):
    Mid-caps engineer wicks that sweep stop-losses before reversing.
    SL requires a candle body close beyond the stop level.
    A severe wick (> 0.3×ATR beyond stop) is treated as a hit regardless.

Entry convention: entry price = close[i] on the signal bar.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from varanus.universe import is_high_vol

logger = logging.getLogger(__name__)

# ── V5 Config ──────────────────────────────────────────────────────────────────

TBM_CONFIG: dict = {
    "atr_window":           14,

    # Barrier multipliers — default placeholders.
    # Overridden by Optuna params: 'sl_atr_mult' + 'rr_ratio' (v5)
    # or 'tp_atr_mult' + 'sl_atr_mult' (v4-compat fallback).
    "take_profit_atr":      3.4235,   # = 0.835 × 4.1 (illustrative default)
    "stop_loss_atr":        0.835,

    # ── V5 LOCK: time barrier at 31 bars. DO NOT pass 'max_holding' to Optuna. ──
    "max_holding_candles":  31,       # 31 × 4h = 5.2 days. FIXED.

    "min_rr_ratio":         3.5,      # Minimum acceptable R:R (lower bound of v5 search)
    "flash_wick_guard":     True,

    # High-vol sub-tier overrides (additive deltas on top of Optuna-chosen base)
    "high_vol_overrides": {
        "sl_atr_delta": 0.30,   # high-vol SL = base + 0.30
        # TP is always sl_effective × rr_ratio (no separate TP delta needed)
    },
}

FLASH_WICK_GUARD: dict = {
    "enabled":                        True,
    "require_body_close_beyond_stop": True,
    "wick_tolerance_atr_ratio":       0.3,    # Wick may pierce up to 0.3×ATR past SL
    "confirmation_candles":           1,
}


# ═══════════════════════════════════════════════════════════════════════════════
# ATR helper
# ═══════════════════════════════════════════════════════════════════════════════

def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Simple ATR: rolling mean of True Range."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ═══════════════════════════════════════════════════════════════════════════════
# Barrier calculation
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_barriers(
    entry:     float,
    atr:       float,
    direction: int,
    cfg:       dict = TBM_CONFIG,
    asset:     str  = "",
) -> dict:
    """
    Compute TP/SL barriers for a single trade setup.

    V5 logic:
      If cfg contains 'rr_ratio', TP is derived: TP = sl_dist × rr_ratio.
      If not, falls back to cfg['take_profit_atr'] (v4-compat).

    High-vol assets receive an additive SL delta:
      sl_effective = sl_atr × ATR + delta × ATR (from high_vol_overrides).
      TP is always sl_effective × rr_ratio (maintains exact R:R).

    Args:
        entry:     Entry price (close of signal bar).
        atr:       ATR(14) at the entry bar.
        direction: +1 for long, -1 for short.
        cfg:       TBM_CONFIG or effective config dict built by label_trades().
        asset:     Base currency string (triggers high-vol overrides).

    Returns:
        dict: take_profit, stop_loss, rr_ratio, min_rr_satisfied
    """
    hv      = is_high_vol(asset)
    hv_cfg  = cfg.get("high_vol_overrides", {})

    # Determine effective SL multiplier
    sl_base = cfg.get("stop_loss_atr", TBM_CONFIG["stop_loss_atr"])
    if hv:
        sl_base = sl_base + hv_cfg.get("sl_atr_delta", 0.30)

    sl_dist = sl_base * atr

    # Determine effective TP
    rr = cfg.get("rr_ratio")
    if rr is not None:
        # V5: R:R-derived TP
        tp_dist = sl_dist * rr
    else:
        # Fallback: explicit TP ATR multiplier (v4-compat)
        tp_mul  = hv_cfg.get("take_profit_atr", cfg["take_profit_atr"]) if hv else cfg["take_profit_atr"]
        tp_dist = tp_mul * atr

    take_profit = entry + direction * tp_dist
    stop_loss   = entry - direction * sl_dist
    realized_rr = tp_dist / sl_dist if sl_dist > 0 else 0.0

    return {
        "take_profit":      take_profit,
        "stop_loss":        stop_loss,
        "rr_ratio":         round(realized_rr, 3),
        "min_rr_satisfied": realized_rr >= cfg.get("min_rr_ratio", TBM_CONFIG["min_rr_ratio"]),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Core labeler
# ═══════════════════════════════════════════════════════════════════════════════

def label_trades(
    df:       pd.DataFrame,
    signals:  pd.Series,
    cfg:      dict = TBM_CONFIG,
    asset:    str  = "",
    params:   Optional[dict] = None,
) -> pd.Series:
    """
    Apply Triple-Barrier Labeling to generate {-1, 0, 1} labels.

    V5 param handling:
      If params contains both 'sl_atr_mult' and 'rr_ratio':
        → sl = sl_atr_mult × ATR
        → tp = sl × rr_ratio  (R:R exact)
      If params contains 'tp_atr_mult' + 'sl_atr_mult' (v4-compat):
        → sl = sl_atr_mult × ATR
        → tp = tp_atr_mult × ATR
      'max_holding_candles' is locked at 31 for v5. If params contains
      'max_holding', it is IGNORED (use cfg['max_holding_candles'] = 31).

    Args:
        df:      OHLCV DataFrame (DatetimeIndex).
        signals: Direction series {-1, 0, 1} aligned to df.index.
        cfg:     TBM_CONFIG or override dict.
        asset:   Base currency — used for high-vol barrier overrides.
        params:  Optional Optuna param overrides.

    Returns:
        pd.Series of int8 {-1, 0, 1} aligned to df.index.
    """
    # Build effective config
    eff                       = cfg.copy()
    eff["high_vol_overrides"] = cfg.get("high_vol_overrides", {}).copy()

    if params:
        if "sl_atr_mult" in params:
            eff["stop_loss_atr"] = params["sl_atr_mult"]

        if "rr_ratio" in params:
            # V5: TP derived from SL × R:R (do NOT read tp_atr_mult separately)
            eff["rr_ratio"] = params["rr_ratio"]
            # Clear any stale explicit TP multiplier to force R:R derivation
            eff.pop("take_profit_atr", None)
        elif "tp_atr_mult" in params:
            # v4-compat fallback
            eff["take_profit_atr"] = params["tp_atr_mult"]
            eff.pop("rr_ratio", None)

        # V5 LOCK: max_holding_candles is ALWAYS 31. Ignore any 'max_holding' param.
        # (Do not override eff["max_holding_candles"] from params.)

    atr_series = _atr(df, eff["atr_window"])
    max_hold   = eff["max_holding_candles"]
    flash_wick = eff.get("flash_wick_guard", True)
    wick_tol   = FLASH_WICK_GUARD["wick_tolerance_atr_ratio"]

    high_arr  = df["high"].values
    low_arr   = df["low"].values
    close_arr = df["close"].values
    atr_arr   = atr_series.values
    sig_arr   = signals.reindex(df.index).fillna(0).values.astype(int)
    n         = len(df)

    labels = np.zeros(n, dtype=np.int8)

    for i in range(n):
        direction = sig_arr[i]
        if direction == 0:
            continue

        atr_val = atr_arr[i]
        if np.isnan(atr_val) or atr_val <= 0:
            continue

        entry_price = close_arr[i]
        barriers    = calculate_barriers(entry_price, atr_val, direction, eff, asset)

        if not barriers["min_rr_satisfied"]:
            continue

        tp       = barriers["take_profit"]
        sl       = barriers["stop_loss"]
        wick_ext = wick_tol * atr_val
        outcome  = 0

        for j in range(i + 1, min(i + max_hold + 1, n)):
            h = high_arr[j]
            l = low_arr[j]
            c = close_arr[j]

            # TP check (wick touch sufficient)
            if direction == 1 and h >= tp:
                outcome = 1
                break
            if direction == -1 and l <= tp:
                outcome = 1
                break

            # SL check (flash-wick guard: body close required)
            if flash_wick:
                if direction == 1:
                    sl_hit = (c < sl) or (l < sl - wick_ext)
                else:
                    sl_hit = (c > sl) or (h > sl + wick_ext)
            else:
                sl_hit = (direction == 1 and l <= sl) or (direction == -1 and h >= sl)

            if sl_hit:
                outcome = -1
                break

        labels[i] = outcome

    return pd.Series(labels, index=df.index, dtype=np.int8, name="label")


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-asset labeler
# ═══════════════════════════════════════════════════════════════════════════════

def label_trades_all_assets(
    data_dict:    dict[str, pd.DataFrame],
    signals_dict: dict[str, pd.Series],
    cfg:          dict = TBM_CONFIG,
    params:       Optional[dict] = None,
) -> pd.DataFrame:
    """
    Apply TBM labeling across all assets, return combined DataFrame.
    """
    records = []
    for asset, df in data_dict.items():
        sigs   = signals_dict.get(asset, pd.Series(0, index=df.index))
        lbl    = label_trades(df, sigs, cfg=cfg, asset=asset, params=params)
        active = (sigs != 0) | (lbl != 0)
        if not active.any():
            continue
        tmp            = lbl[active].reset_index()
        tmp.columns    = ["timestamp", "label"]
        tmp["asset"]   = asset
        records.append(tmp)

    if not records:
        return pd.DataFrame(columns=["timestamp", "asset", "label"])

    out = pd.concat(records, ignore_index=True)
    return out.sort_values("timestamp").reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Diagnostics
# ═══════════════════════════════════════════════════════════════════════════════

def barrier_stats(
    labels:  pd.Series,
    signals: Optional[pd.Series] = None,
) -> dict:
    """Summarise label distribution for a single asset."""
    if signals is not None:
        active   = signals[signals != 0].index
        filtered = labels.reindex(active)
    else:
        filtered = labels[labels != 0]

    total   = len(labels)
    n_sig   = len(filtered)
    wins    = int((filtered == 1).sum())
    losses  = int((filtered == -1).sum())
    neutral = int((filtered == 0).sum())

    win_rate     = wins    / n_sig if n_sig > 0 else 0.0
    loss_rate    = losses  / n_sig if n_sig > 0 else 0.0
    neutral_rate = neutral / n_sig if n_sig > 0 else 0.0
    rr_implied   = win_rate / loss_rate if loss_rate > 0 else float("inf")

    return {
        "total_bars":    total,
        "signal_bars":   n_sig,
        "win_count":     wins,
        "loss_count":    losses,
        "neutral_count": neutral,
        "win_rate":      round(win_rate,     4),
        "loss_rate":     round(loss_rate,    4),
        "neutral_rate":  round(neutral_rate, 4),
        "rr_implied":    round(rr_implied,   3),
    }
