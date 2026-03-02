"""
validate_steps.py — Varanus v5 Module Smoke-Tests (Steps 3–9).

Usage:
    cd /home/yagokhan
    PYTHONPATH=/home/yagokhan python varanus/validate_steps.py

Each step creates a minimal synthetic dataset (or uses the real cache),
exercises the core API, and asserts the output contracts.
Prints PASS / FAIL for each step.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# ── Ensure project root is on sys.path ────────────────────────────────────────
ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

PASS = "✅ PASS"
FAIL = "❌ FAIL"
results: dict[str, str] = {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame with DatetimeIndex."""
    rng   = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.0002, 0.015, n))
    high  = close * (1 + rng.uniform(0, 0.02, n))
    low   = close * (1 - rng.uniform(0, 0.02, n))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    vol   = rng.uniform(1e5, 1e7, n)
    idx   = pd.date_range("2022-01-01", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": vol}, index=idx)


def _run(label: str, fn):
    """Execute fn(); catch exceptions and record PASS/FAIL."""
    try:
        fn()
        results[label] = PASS
    except Exception:
        results[label] = FAIL
        print(f"\n  [{label}] Exception:\n")
        traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════════════════
# Step 0 — Package init
# ═══════════════════════════════════════════════════════════════════════════════

def _step0():
    import varanus
    assert hasattr(varanus, "__version__"), "missing __version__"
    assert varanus.__version__.startswith("5"), f"expected v5, got {varanus.__version__}"
    print(f"    version={varanus.__version__}  tier={varanus.__tier__}")

_run("STEP 0 — package init", _step0)


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1 — Universe
# ═══════════════════════════════════════════════════════════════════════════════

def _step1():
    from varanus.universe import (
        TIER2_UNIVERSE, HIGH_VOL_SUBTIER, EXCLUSION_RULES,
        is_high_vol, get_size_scalar,
    )
    assert len(TIER2_UNIVERSE) == 20, f"expected 20 assets, got {len(TIER2_UNIVERSE)}"
    assert all(a in TIER2_UNIVERSE for a in HIGH_VOL_SUBTIER), "HIGH_VOL missing from universe"
    # high-vol asset scalar < 1
    assert get_size_scalar("TAO", 0.7) < 1.0, "TAO should get size reduction"
    # power setup scalar > 1
    assert get_size_scalar("BNB", 0.96) > 1.0, "Power Setup should get scalar > 1"
    print(f"    universe={len(TIER2_UNIVERSE)} assets  high_vol={HIGH_VOL_SUBTIER}")

_run("STEP 1 — universe", _step1)


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2 — PA Features
# ═══════════════════════════════════════════════════════════════════════════════

def _step2():
    from varanus.pa_features import build_features, FEATURE_COLS
    df   = _make_ohlcv(500)   # 500 bars: enough to warm up all rolling windows (100-bar ATR%ile)
    feat = build_features(df, df, asset="BNB")
    assert set(FEATURE_COLS).issubset(feat.columns), \
        f"missing columns: {set(FEATURE_COLS) - set(feat.columns)}"
    assert len(feat) > 0, "build_features returned empty DataFrame"
    assert len(feat) <= len(df), "output longer than input"
    print(f"    features={list(feat.columns)[:4]}… rows={len(feat)} / {len(df)}")

_run("STEP 2 — pa_features", _step2)


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3 — TBM Labeler
# ═══════════════════════════════════════════════════════════════════════════════

def _step3():
    from varanus.tbm_labeler import label_trades, calculate_barriers, TBM_CONFIG

    df      = _make_ohlcv(300)
    signals = pd.Series(0, index=df.index, dtype=int)
    signals.iloc[50] = 1
    signals.iloc[100] = -1

    params = {"sl_atr_mult": 0.85, "rr_ratio": 4.1}
    labels = label_trades(df, signals, cfg=TBM_CONFIG, asset="BNB", params=params)

    assert set(labels.unique()).issubset({-1, 0, 1}), "labels must be in {-1, 0, 1}"
    assert len(labels) == len(df)

    # Barrier calculation
    bar = calculate_barriers(100.0, 1.0, 1, TBM_CONFIG)
    assert "take_profit" in bar and "stop_loss" in bar and "rr_ratio" in bar
    print(f"    label distribution: {labels.value_counts().to_dict()}")

_run("STEP 3 — tbm_labeler", _step3)


# ═══════════════════════════════════════════════════════════════════════════════
# Step 4 — Model
# ═══════════════════════════════════════════════════════════════════════════════

def _step4():
    from varanus.model import VaranusModel, get_leverage, is_power_setup, FEATURE_LIST

    # Leverage map
    assert get_leverage(0.76)  == 1.0, "Base tier should be 1x"
    assert get_leverage(0.86)  == 2.0, "Standard tier should be 2x"
    assert get_leverage(0.93)  == 3.0, "High conviction should be 3x"
    assert get_leverage(0.96)  == 5.0, "Power Setup should be 5x"
    assert is_power_setup(0.95), "0.95 should be Power Setup"
    assert not is_power_setup(0.94), "0.94 should NOT be Power Setup"

    # Fit + predict on synthetic data
    rng    = np.random.default_rng(0)
    n      = 200
    X      = pd.DataFrame(rng.normal(size=(n, len(FEATURE_LIST))), columns=FEATURE_LIST)
    y      = pd.Series(rng.choice([-1, 0, 1], size=n))
    model  = VaranusModel()
    model.fit(X[:150], y[:150], X[150:], y[150:])
    preds  = model.predict(X[150:], confidence_thresh=0.50)
    proba  = model.predict_proba(X[150:])

    assert preds.shape == (50,), f"predict shape mismatch: {preds.shape}"
    assert proba.shape == (50, 3), f"predict_proba shape mismatch: {proba.shape}"
    assert set(np.unique(preds)).issubset({-1, 0, 1})
    print(f"    leverage map OK | pred unique={np.unique(preds)}")

_run("STEP 4 — model", _step4)


# ═══════════════════════════════════════════════════════════════════════════════
# Step 5 — Backtest
# ═══════════════════════════════════════════════════════════════════════════════

def _step5():
    from varanus.backtest import (
        run_backtest, compute_metrics, passes_backtest_gate,
        BACKTEST_CONFIG, BACKTEST_PASS_CRITERIA,
    )
    from varanus.model import VaranusModel

    # V5 config checks
    assert BACKTEST_CONFIG["max_portfolio_leverage"] == 3.5
    assert BACKTEST_CONFIG["power_setup_size_scalar"] == 1.25
    assert BACKTEST_PASS_CRITERIA["min_profit_factor"] == 1.50
    assert BACKTEST_PASS_CRITERIA["max_drawdown"] == -0.15

    # Minimal synthetic backtest
    df  = _make_ohlcv(200)
    sig = pd.DataFrame({
        "confidence": [0.80] * len(df),
        "direction":  [1]    * len(df),
        "entry_price": df["close"].values,
        "atr":         (df["high"] - df["low"]).values,
    }, index=df.index)

    data    = {"BNB": df}
    signals = {"BNB": sig}
    model   = VaranusModel()

    equity, trade_log = run_backtest(
        data    = data,
        signals = signals,
        model   = model,
        params  = {"confidence_thresh": 0.75, "sl_atr_mult": 0.85, "rr_ratio": 4.1},
    )
    assert isinstance(equity,    pd.Series)
    assert isinstance(trade_log, pd.DataFrame)

    metrics = compute_metrics(equity, trade_log)
    assert "profit_factor" in metrics
    print(f"    trades={metrics['total_trades']}  PF={metrics['profit_factor']}"
          f"  DD={metrics['max_drawdown_pct']}%")

_run("STEP 5 — backtest", _step5)


# ═══════════════════════════════════════════════════════════════════════════════
# Step 6 — Walk-Forward
# ═══════════════════════════════════════════════════════════════════════════════

def _step6():
    from varanus.walk_forward import (
        generate_rolling_folds, _validate_fold_integrity, WFV_CONFIG, FoldIndices,
    )

    n = 5000   # large enough for 8-fold × 40% train ≥ min_train_candles=400
    folds = generate_rolling_folds(n, n_folds=8, gap_candles=24)
    assert len(folds) == 8, f"expected 8 folds, got {len(folds)}"

    for f in folds:
        assert isinstance(f, FoldIndices)

    _validate_fold_integrity(folds, n)   # takes the full list

    # Check non-overlapping OOS test windows
    test_stops = [f.test.stop for f in folds]
    assert test_stops == sorted(test_stops), "test windows not monotonically increasing"
    assert folds[-1].test.stop <= n, f"last fold overflows: {folds[-1].test.stop} > {n}"

    print(f"    8 folds OK | train={folds[0].train_bars}  val={folds[0].val_bars}"
          f"  test={folds[0].test_bars}")

_run("STEP 6 — walk_forward", _step6)


# ═══════════════════════════════════════════════════════════════════════════════
# Step 7 — Optimizer
# ═══════════════════════════════════════════════════════════════════════════════

def _step7():
    from varanus.optimizer import V5_SEARCH_SPACE, V5_TIME_EXIT_BARS, OPTUNA_CONFIG

    # V5 time exit lock
    assert V5_TIME_EXIT_BARS == 31, "Time exit must be 31"
    assert "max_holding_candles" not in V5_SEARCH_SPACE, \
        "max_holding_candles must NOT be an Optuna search param"

    # Search ranges
    ct = V5_SEARCH_SPACE["confidence_thresh"]
    sl = V5_SEARCH_SPACE["sl_atr_mult"]
    rr = V5_SEARCH_SPACE["rr_ratio"]

    assert ct["low"]  == 0.750 and ct["high"] == 0.880
    assert sl["low"]  == 0.700 and sl["high"] == 1.200
    assert rr["low"]  == 3.500 and rr["high"] == 5.000

    assert OPTUNA_CONFIG["n_trials"] == 300
    print(f"    search space keys={list(V5_SEARCH_SPACE.keys())}")

_run("STEP 7 — optimizer config", _step7)


# ═══════════════════════════════════════════════════════════════════════════════
# Step 8 — Risk
# ═══════════════════════════════════════════════════════════════════════════════

def _step8():
    from varanus.risk import (
        RISK_CONFIG, check_portfolio_health, get_position_size,
        compute_portfolio_leverage, would_breach_leverage,
        count_power_setups, would_breach_power_setup_cap,
        is_correlated_to_open,
    )

    # V5 config checks
    assert RISK_CONFIG["max_portfolio_leverage"] == 3.5
    assert RISK_CONFIG["portfolio_stop_pct"]     == 0.15
    assert RISK_CONFIG["power_setup_max_concurrent"] == 2

    # Position sizing — Power Setup (5x + 1.25 scalar)
    pos = get_position_size(0.96, 10_000, "BNB")
    assert pos > 0, "position should be > 0"
    # Standard 1x
    pos1x = get_position_size(0.76, 10_000, "BNB")
    assert pos > pos1x, "Power Setup position should exceed base-tier position"

    # Health check
    idx    = pd.date_range("2024-01-01", periods=100, freq="4h", tz="UTC")
    equity = pd.Series([5000 - i * 10 for i in range(100)], index=idx)
    health = check_portfolio_health(equity)
    assert "halt_signals" in health
    assert health["drawdown_pct"] < 0

    # Power Setup cap
    fake_ps_trades = {
        "BNB": {"position_usd": 100, "power_setup": True},
        "SOL": {"position_usd": 100, "power_setup": True},
    }
    assert count_power_setups(fake_ps_trades) == 2
    assert would_breach_power_setup_cap(fake_ps_trades, 0.97), \
        "should be blocked: 2 PS trades already open"

    print(f"    RISK_CONFIG max_lev={RISK_CONFIG['max_portfolio_leverage']}  "
          f"port_stop={RISK_CONFIG['portfolio_stop_pct']}")

_run("STEP 8 — risk", _step8)


# ═══════════════════════════════════════════════════════════════════════════════
# Step 9 — Alerts
# ═══════════════════════════════════════════════════════════════════════════════

def _step9():
    from varanus.alerts import (
        send_alert, send_exit_alert, send_halt_alert, send_no_signal_alert,
        _confidence_label,
    )

    assert _confidence_label(0.96)  == "⚡ POWER SETUP"
    assert _confidence_label(0.93)  == "🔴 HIGH CONVICTION"
    assert _confidence_label(0.87)  == "🟡 STANDARD"
    assert _confidence_label(0.76)  == "⚪ BASE"

    trade = {
        "timestamp_utc": "2024-01-01T00:00:00Z",
        "asset":         "BNB",
        "direction":     "LONG",
        "confidence":    0.96,
        "leverage":      5.0,
        "entry_price":   300.0,
        "take_profit":   320.0,
        "stop_loss":     295.0,
        "rr_ratio":      4.0,
        "atr_14":        2.5,
        "mss":           "BULLISH",
        "fvg_valid":     True,
        "sweep_confirmed": True,
        "rvol":          1.8,
        "rsi":           55.0,
        "htf_bias":      "BULL",
        "position_usd":  1250.0,
        "port_lev":      2.5,
        "power_setup":   True,
    }

    # dry_run=True — should print without raising
    send_alert(trade, "fake_token", "fake_chat", dry_run=True)
    send_halt_alert(
        {"daily_loss_pct": -6.0, "drawdown_pct": -16.0,
         "current_equity": 4200.0, "halt_reason": "test"},
        "fake_token", "fake_chat", dry_run=True,
    )
    print("    entry/halt dry-run alerts printed successfully")

_run("STEP 9 — alerts", _step9)


# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 55)
print("  Varanus v5.0 — Validation Summary")
print("═" * 55)
all_pass = True
for step, status in results.items():
    print(f"  {status}  {step}")
    if status != PASS:
        all_pass = False
print("═" * 55)

if all_pass:
    print("  ALL STEPS PASS — v5.0 implementation validated ✅")
else:
    print("  ⚠  Some steps FAILED — check output above for details")
    sys.exit(1)
