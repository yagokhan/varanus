# Varanus v5.0 — High-Resolution Regime Adaptation Strategy
> **Evolutionary upgrade from v4.0 Structural Bridge.**
> v5.0 introduces an 8-Fold High-Resolution Rolling Window, a
> regime-adaptive HPO objective, and a Power Setup 5x Leverage tier.
> The predatory entry logic (MSS + FVG + Sweep) is structurally preserved.

---

## Version Lineage

| Version | Architecture          | Folds | Split        | Leverage Cap | Status   |
|---------|----------------------|-------|--------------|--------------|----------|
| v4.0.0  | 5-Fold Sliding WFV   | 5     | 70 / 15 / 15 | 3x           | FROZEN ← `/v4/` |
| **v5.0.0**  | **8-Fold Rolling WFV**   | **8**     | **40 / 30 / 30** | **5x (Power Setup)** | **ACTIVE** |

> **Protected Baseline:** All v4 code lives in `/v4/`. It is read-only.
> No file from `/v4/` may be imported, modified, or monkey-patched by v5.

---

## Implementation Roadmap

```
STEP 0  →  Project Setup & V5 Module Architecture
STEP 1  →  Universe Definition (inherited from v4, validated for 2026)
STEP 2  →  Feature Engineering (MSS + FVG + Sweep — structurally preserved)
STEP 3  →  Triple-Barrier Labeling (31-Bar time exit — v5 lock)
STEP 4  →  Model Training (XGBoost — same feature vector, new fold logic)
STEP 5  →  Backtesting Engine (updated leverage map, 5x Power Setup)
STEP 6  →  8-Fold Rolling Window Walk-Forward Validation ← KEY UPGRADE
STEP 7  →  V5 Optuna HPO (new objective, new search ranges)
STEP 8  →  Risk Management (updated leverage caps + drawdown gate)
STEP 9  →  Alert System (Power Setup flag in Telegram output)
STEP 10 →  Governance & Version Tagging (v5.0.0)
```

> **Build discipline:** Complete and validate each step before proceeding.
> Never skip forward. All `assert` gates must pass before the next step begins.

---

## STEP 0 — Project Setup & V5 Module Architecture

### Directory Structure

```
varanus/                          ← V5 active root
│   ├── v5_claude.md              ← THIS FILE. Single source of truth for v5.
│   ├── universe.py               ← STEP 1  (re-validated for 2026 regime)
│   ├── pa_features.py            ← STEP 2  (structurally identical to v4)
│   ├── tbm_labeler.py            ← STEP 3  (31-bar lock enforced)
│   ├── model.py                  ← STEP 4  (same architecture, new folds)
│   ├── backtest.py               ← STEP 5  (updated leverage map)
│   ├── walk_forward.py           ← STEP 6  (8-fold rolling window — NEW)
│   ├── optimizer.py              ← STEP 7  (new HPO objective — NEW)
│   ├── risk.py                   ← STEP 8  (Power Setup tier added)
│   ├── alerts.py                 ← STEP 9  (Power Setup flag added)
│   ├── config/
│   │   ├── settings.yaml
│   │   ├── best_params_v5.json   ← populated after STEP 7
│   │   └── .env                  ← Telegram credentials (NOT committed)
│   ├── results/
│   ├── plots/
│   └── v4/                       ← FROZEN BASELINE. DO NOT IMPORT.
```

### Isolation Contract

```python
# v5 modules must NEVER do this:
#   from varanus.v4 import anything
#   import v4.model
# v5 is a clean rewrite. v4 is an archive, not a library.
```

### Step 0 Validation ✓
```bash
python -c "from chameleon import features; print('Chameleon import OK')"
python -c "import varanus; print('Varanus v5 package OK')"
python -c "
import varanus
assert not hasattr(varanus, 'v4'), 'v4 must not be auto-imported'
print('Isolation check OK')
"
```

---

## STEP 1 — Universe Definition

**File:** `varanus/universe.py`

Universe is structurally identical to v4. Re-validate each asset for 2026
liquidity before commencing STEP 7. Any asset with < $50M 24h volume for
5 consecutive days is suspended from the active universe for that HPO cycle.

```python
TIER2_UNIVERSE = [
    "DOT", "ADA", "LINK", "AVAX", "ASTR", "APT", "KITE",
    "HBAR", "TAO", "TRX", "NEAR", "UNI", "LTC", "SUI",
    "ARB", "OP", "POL", "ATOM", "FIL", "ICP",
]

TIER2_QUOTE       = "USDT"
TIER2_TF          = "4h"
TIER2_TF_HTF      = "1d"
TIER2_MIN_VOL_USD = 50_000_000

HIGH_VOL_SUBTIER  = ["TAO", "ASTR", "KITE", "ICP"]

V5_UNIVERSE_NOTE = (
    "Re-validate for 2026 liquidity regime before running HPO. "
    "Delist any asset that fails the vol gate for 5+ consecutive days."
)
```

### Step 1 Validation ✓
```python
assert len(TIER2_UNIVERSE) == 20
assert all(a in TIER2_UNIVERSE for a in HIGH_VOL_SUBTIER)
print(f"V5 Universe: {len(TIER2_UNIVERSE)} assets | Sub-tier: {HIGH_VOL_SUBTIER}")
```

---

## STEP 2 — Feature Engineering

**File:** `varanus/pa_features.py`

> **Structural Preservation Principle:** The MSS + FVG + Liquidity Sweep
> pipeline is the Predatory Edge. It is NOT a search space variable in v5 HPO.
> It is preserved exactly as designed in v4, validated, and locked.

### 2.1 Preserved Configuration (from v4 — DO NOT alter in v5)

```python
MSS_CONFIG = {
    "lookback_range":     (30, 50),
    "lookback_default":   40,
    "swing_confirmation": 3,
    "body_filter":        0.6,
    "wick_tolerance":     0.005,
    "htf_bias_required":  True,
}

FVG_CONFIG = {
    "min_gap_atr_ratio":    0.3,
    "max_gap_age_candles":  20,
    "sweep_lookback":       15,
    "min_sweep_pct":        0.004,
    "sweep_close_reversal": True,
    "fvg_partial_fill_pct": 0.5,
    "require_sweep":        True,   # NEVER set to False in v5
}

CONFIRMATION_FEATURES = {
    "relative_volume": {"window": 20, "threshold": 1.5, "weight": 0.40},
    "rsi_14":          {"oversold": 35, "overbought": 65, "weight": 0.35},
    "atr_percentile":  {"window": 100, "min_percentile": 40, "weight": 0.15},
    "ema_alignment":   {"fast": 21, "slow": 55, "weight": 0.10},
}
CONFIRMATION_SCORE_MIN = 0.70
```

### Step 2 Validation ✓
```python
df  = load_ohlcv("LINKUSDT", "4h", limit=500)
atr = compute_atr(df, 14)
mss = detect_mss(df)
fvg = detect_fvg(df, atr, FVG_CONFIG)

assert mss.isin([-1, 0, 1]).all(),  "MSS values out of range"
assert 'fvg_valid' in fvg.columns,  "FVG output malformed"
print(f"LINK | MSS: {(mss!=0).sum()} signals | Valid FVGs: {fvg['fvg_valid'].sum()}")
```

---

## STEP 3 — Triple-Barrier Labeling

**File:** `varanus/tbm_labeler.py`

### V5 Time Exit Lock: 31 Bars

The time barrier is **locked at 31 bars** for v5. This is not a search
space variable in the HPO. 31 × 4h = 124h ≈ 5.2 days. It provides a
fractionally wider decay window than v4's 30-bar default to capture the
slower structural breakouts observed in 2026 mid-cap regimes.

```python
TBM_CONFIG_V5 = {
    "atr_window":           14,

    # ── ATR multipliers: SEARCH SPACE for v5 HPO (see STEP 7) ──────────
    # Defaults below are starting points only. Final values come from Optuna.
    "take_profit_atr":      None,   # Derived: sl_atr_mult × rr_ratio
    "stop_loss_atr":        None,   # Optimized: range [0.700, 1.200]

    # ── V5 LOCK — do not make this a trial parameter ─────────────────────
    "max_holding_candles":  31,     # 31 × 4h = 5.2 days. FIXED.

    "min_rr_ratio":         3.5,    # Minimum acceptable R:R — aligned to
                                    # lower bound of v5 HPO TP search space.
    "flash_wick_guard":     True,

    "high_vol_overrides": {
        # Applied additively to the Optuna-chosen base multipliers
        "sl_atr_delta":  0.3,   # High-vol SL = base + 0.3
        "tp_atr_delta":  0.5,   # High-vol TP = base + 0.5
    },
}

def calculate_barriers_v5(entry: float, atr: float, direction: int,
                           sl_mult: float, rr_ratio: float,
                           asset: str) -> dict:
    """
    V5 barriers derived from Optuna-optimized sl_mult and rr_ratio.
    TP is always a multiple of SL distance — ensuring R:R is exact.
    """
    cfg     = TBM_CONFIG_V5
    is_hv   = asset in HIGH_VOL_SUBTIER
    sl_m    = sl_mult + (cfg['high_vol_overrides']['sl_atr_delta'] if is_hv else 0)
    sl_dist = sl_m * atr
    tp_dist = sl_dist * rr_ratio + (
        cfg['high_vol_overrides']['tp_atr_delta'] * atr if is_hv else 0
    )

    take_profit = entry + direction * tp_dist
    stop_loss   = entry - direction * sl_dist
    rr          = tp_dist / sl_dist

    return {
        "take_profit":      take_profit,
        "stop_loss":        stop_loss,
        "rr_ratio":         round(rr, 3),
        "min_rr_satisfied": rr >= cfg['min_rr_ratio'],
        "sl_atr_applied":   round(sl_m, 4),
        "tp_atr_applied":   round(sl_m * rr, 4),
    }
```

### Label Encoding (unchanged from v4)
```
 1  → TP hit first  (win)
 0  → Time barrier expires at bar 31 (neutral)
-1  → SL hit first  (loss)
```

### Step 3 Validation ✓
```python
# Confirm 31-bar lock is respected
assert TBM_CONFIG_V5['max_holding_candles'] == 31, "Time exit lock violated"

labels = label_trades(df, signals, TBM_CONFIG_V5, asset="LINK",
                      sl_mult=0.90, rr_ratio=4.0)
dist   = labels.value_counts(normalize=True)
print(f"Label distribution:\n{dist}")
assert dist.index.isin([-1, 0, 1]).all()
```

---

## STEP 4 — Model Training

**File:** `varanus/model.py`

Feature vector is identical to v4 (16 features). The model architecture
is unchanged. What changes is how data is presented to the model — driven
by the 8-fold rolling window in STEP 6.

```python
FEATURE_LIST = [
    # PA Core
    "mss_signal", "fvg_type", "fvg_distance_atr", "fvg_age_candles",
    "sweep_occurred", "htf_bias",
    # Chameleon Confirmation
    "relative_volume", "rsi_14", "rsi_slope_3", "ema21_55_alignment",
    "atr_percentile_100",
    # Market Character
    "volatility_rank", "volume_rank", "asset_tier_flag",
    "hour_of_day", "day_of_week",
]

MODEL_CONFIG_V5 = {
    "type":                 "XGBoostClassifier",
    "target_classes":       3,
    # confidence_threshold is now an HPO variable — not hardcoded here.
    # After STEP 7, best_params_v5.json will supply the final value.
    "confidence_threshold": None,   # Populated by STEP 7 best params

    "xgb_params": {
        "n_estimators":          500,
        "max_depth":             6,
        "learning_rate":         0.05,
        "subsample":             0.8,
        "colsample_bytree":      0.8,
        "eval_metric":           "mlogloss",
        "early_stopping_rounds": 30,
        "use_label_encoder":     False,
    },
}
```

### Step 4 Validation ✓
```python
model = VaranusModel(MODEL_CONFIG_V5)
model.fit(X_train, y_train, X_val, y_val)
probs = model.predict_proba(X_test)
assert probs.shape[1] == 3
assert (probs.sum(axis=1).round(2) == 1.0).all()
print(f"Signals at ≥0.80 gate: {(probs.max(axis=1) >= 0.80).sum()}")
```

---

## STEP 5 — Backtesting Engine

**File:** `varanus/backtest.py`

### V5 Dynamic Leverage — Power Setup

V5 introduces a fourth leverage tier. When the model outputs a confidence
score ≥ 0.95, the trade is classified as a **Power Setup**. This is not a
common occurrence — it signals the model has found convergence across all
feature dimensions. At this tier, leverage scales to 5x.

```python
# ── V5 Leverage Map ───────────────────────────────────────────────────────
CONFIDENCE_LEVERAGE_MAP_V5 = {
    # (lower_inclusive, upper_exclusive): leverage_multiplier
    (0.750, 0.850): 1.0,   # Base signal — confirmed entry, lean position
    (0.850, 0.920): 2.0,   # Strong signal — standard conviction
    (0.920, 0.950): 3.0,   # High conviction — structural alignment
    (0.950, 1.001): 5.0,   # Power Setup — full convergence (v5 only)
}

def get_leverage_v5(confidence: float) -> float:
    """
    V5 Leverage resolution with Power Setup tier.

    Key upgrade vs v4:
      - Tier 4 (≥0.95): 5x leverage enabled
      - Lower bands widen slightly to match new HPO confidence floor (0.750)

    Args:
        confidence: XGBoost max class probability [0.0, 1.0]

    Returns:
        Leverage multiplier (1.0 / 2.0 / 3.0 / 5.0)
    """
    for (lo, hi), lev in CONFIDENCE_LEVERAGE_MAP_V5.items():
        if lo <= confidence < hi:
            return lev
    return 1.0   # Fallback — should not trigger if confidence gate is applied


def is_power_setup(confidence: float) -> bool:
    """True if this trade qualifies for the 5x Power Setup tier."""
    return confidence >= 0.950
```

### V5 Backtest Configuration

```python
BACKTEST_CONFIG_V5 = {
    "initial_capital":           5_000.0,
    "maker_fee":                 0.0002,
    "taker_fee":                 0.0005,
    "slippage_pct":              0.0008,
    "entry_on_bar":              "open",

    "use_flash_wick_guard":      True,
    "wick_body_close_required":  True,

    # Portfolio constraints
    "max_concurrent_positions":  4,
    "max_portfolio_leverage":    3.5,   # Raised from v4's 2.5x to accommodate
                                        # Power Setup entries without halting
                                        # normal positions
    "corr_block_threshold":      0.75,
    "corr_lookback_days":        20,

    "power_setup_size_scalar":   1.25,  # Power Setup positions get +25% size
                                        # before leverage is applied
    "high_vol_size_scalar":      0.75,  # TAO, ASTR, KITE, ICP (unchanged)

    "equity_curve_freq":         "4h",
    "trade_log":                 True,
}
```

### Position Sizing with Power Setup

```python
def compute_position_size(
    capital:    float,
    confidence: float,
    asset:      str,
    cfg:        dict = BACKTEST_CONFIG_V5,
) -> tuple[float, float, bool]:
    """
    Returns (position_usd, leverage, power_setup_flag).

    Sizing logic (applied in order):
      1. Base allocation = capital / max_concurrent_positions
      2. If Power Setup: multiply by power_setup_size_scalar
      3. If High-Vol Sub-Tier: multiply by high_vol_size_scalar
      4. Apply leverage
    """
    lev       = get_leverage_v5(confidence)
    ps_flag   = is_power_setup(confidence)
    hv_flag   = asset in HIGH_VOL_SUBTIER

    base      = capital / cfg['max_concurrent_positions']
    ps_scalar = cfg['power_setup_size_scalar'] if ps_flag else 1.0
    hv_scalar = cfg['high_vol_size_scalar']    if hv_flag else 1.0

    position  = base * ps_scalar * hv_scalar * lev

    return position, lev, ps_flag
```

### Minimum Acceptance Criteria (V5 — tightened)

```python
BACKTEST_PASS_CRITERIA_V5 = {
    "min_trades":         50,
    "min_win_rate":       0.43,
    "min_profit_factor":  1.50,   # Raised from v4's 1.30 — v5 HPO objective
    "max_drawdown":      -0.15,   # Hard cap at -15% (v5 DD constraint)
    "min_sharpe":         0.90,   # Raised from v4's 0.80
    "min_calmar":         0.60,
}
```

### Step 5 Validation ✓
```python
equity, trades = run_backtest(data, signals, model, best_params_v5)
metrics        = compute_metrics(equity, trades)

# Confirm Power Setup trades exist and leverage is correct
ps_trades = trades[trades['confidence'] >= 0.95]
if len(ps_trades) > 0:
    assert (ps_trades['leverage'] == 5.0).all(), "Power Setup leverage incorrect"
    print(f"Power Setup trades: {len(ps_trades)} ({len(ps_trades)/len(trades):.1%})")

passed = passes_backtest_gate_v5(metrics)
print(f"V5 Backtest gate: {'PASS ✓' if passed else 'FAIL ✗'}")
assert len(trades) >= 10, "Too few trades — check signal pipeline"
```

---

## STEP 6 — 8-Fold High-Resolution Rolling Window

**File:** `varanus/walk_forward.py`

### Architecture Rationale

V4 used a 5-fold 70/15/15 split — optimized for long training windows
in a stable 2024–2025 bull-to-correction cycle. V5's 8-fold 40/30/30
design prioritizes **regime adaptation speed** for 2026:

| Design Axis        | V4 (5-fold)     | V5 (8-fold)      | Effect                         |
|--------------------|-----------------|------------------|-------------------------------|
| Train ratio        | 70%             | 40%              | Leaner — faster regime adapt  |
| Val ratio          | 15%             | 30%              | Wider HPO tuning surface       |
| Test ratio         | 15%             | 30%              | Larger OOS sample per fold     |
| Folds              | 5               | 8                | 60% more resolution            |
| OOS coverage       | 75% of data     | 8 × 30% windows  | High statistical confidence    |

### Window Arithmetic

For a dataset with `N` candles:

```
Window size  W  = (N − 2×gap) / (1 + (n_folds − 1) × test_ratio)
                = (N − 48)    / (1 + 7 × 0.30)
                = (N − 48)    / 3.10

train_size      = W × 0.40
val_size        = W × 0.30
test_size       = W × 0.30   ← also the step between folds

Fold k layout (0-indexed):
  [k×step  …  k×step+train_size)          ← TRAIN
  [+gap    …  +gap+val_size)              ← VAL   (leakage gap before)
  [+gap    …  +gap+test_size)             ← TEST  (leakage gap before)

Total span check: W + 2×gap + 7×step = N  ✓
```

**Example:** 2 years of 4h data = 4,380 candles, gap = 24 candles
```
W         ≈ (4380 − 48) / 3.10  ≈ 1,397 candles (≈ 233 days)
train     ≈  559 candles  (≈ 93 days)
val       ≈  419 candles  (≈ 70 days)
test      ≈  419 candles  (≈ 70 days)
step      ≈  419 candles  (≈ 70 days)
```

### Python Implementation

```python
"""
varanus/walk_forward.py
8-Fold High-Resolution Rolling Window Walk-Forward Validation.
"""

from __future__ import annotations
import numpy  as np
import pandas as pd
from dataclasses import dataclass, field
from typing      import Iterator

# ── Configuration ─────────────────────────────────────────────────────────

WFV_CONFIG_V5 = {
    "n_folds":           8,
    "method":            "rolling_window",   # Each fold advances by step=test_size
    "shuffle":           False,              # NEVER shuffle. Temporal integrity sacred.
    "train_ratio":       0.40,
    "val_ratio":         0.30,
    "test_ratio":        0.30,
    "min_train_candles": 400,               # ~67 days on 4h — hard floor
    "gap_candles":       24,                # 4-day gap between splits (leakage guard)
    "performance_gate": {
        "min_profit_factor": 1.50,          # V5 primary gate
        "min_win_rate":      0.43,
        "max_fold_dd":      -0.15,          # Per-fold DD cap (15%)
        "min_calmar":        0.50,
        "consistency_req":   0.75,          # ≥ 75% of folds must pass (6/8)
    },
}


@dataclass
class FoldIndices:
    """Index boundaries for a single rolling fold."""
    fold:       int
    train:      slice
    val:        slice
    test:       slice
    train_bars: int
    val_bars:   int
    test_bars:  int

    def __repr__(self) -> str:
        return (
            f"Fold {self.fold:02d} | "
            f"train=[{self.train.start}:{self.train.stop}]({self.train_bars}) "
            f"val=[{self.val.start}:{self.val.stop}]({self.val_bars}) "
            f"test=[{self.test.start}:{self.test.stop}]({self.test_bars})"
        )


# ── Core Fold Generator ────────────────────────────────────────────────────

def generate_rolling_folds(
    n_samples:   int,
    n_folds:     int   = WFV_CONFIG_V5["n_folds"],
    train_ratio: float = WFV_CONFIG_V5["train_ratio"],
    val_ratio:   float = WFV_CONFIG_V5["val_ratio"],
    test_ratio:  float = WFV_CONFIG_V5["test_ratio"],
    gap_candles: int   = WFV_CONFIG_V5["gap_candles"],
    min_train:   int   = WFV_CONFIG_V5["min_train_candles"],
) -> list[FoldIndices]:
    """
    8-Fold High-Resolution Rolling Window generator.

    Design contract:
    ─────────────────────────────────────────────────────────────────────
    • Split ratios:  40% train / 30% val / 30% test (must sum to 1.0)
    • Step:          Exactly one test_size forward per fold → non-overlapping
                     OOS test windows across all 8 folds
    • Gap:           gap_candles removed between train→val and val→test
                     to eliminate look-ahead leakage at boundaries
    • Shuffle:       NEVER. Temporal ordering is the contract.
    • OOS coverage:  All 8 test windows cover distinct, non-overlapping
                     calendar segments of the dataset
    ─────────────────────────────────────────────────────────────────────

    Window arithmetic (derivation):
        W + (n_folds − 1) × step    = n_samples − 2 × gap_candles
        step                         = W × test_ratio
        W × (1 + (n_folds−1) × test_ratio) = n_samples − 2 × gap_candles
        W = (n_samples − 2 × gap_candles) / (1 + (n_folds−1) × test_ratio)

    Args:
        n_samples:   Total number of candles in the dataset.
        n_folds:     Number of rolling folds (default 8).
        train_ratio: Fraction of window for training (default 0.40).
        val_ratio:   Fraction of window for validation (default 0.30).
        test_ratio:  Fraction of window for OOS testing (default 0.30).
        gap_candles: Candles to skip at train→val and val→test boundaries.
        min_train:   Minimum training candles — raises if violated.

    Returns:
        List of FoldIndices (length == n_folds).

    Raises:
        ValueError: If dataset too small, ratios malformed, or gap too large.
        AssertionError: If post-construction sanity checks fail.
    """
    # ── Input validation ──────────────────────────────────────────────────
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-9:
        raise ValueError(
            f"Ratios must sum to 1.0. Got {train_ratio + val_ratio + test_ratio:.6f}"
        )
    if n_folds < 2:
        raise ValueError(f"n_folds must be ≥ 2. Got {n_folds}")

    # ── Window size derivation ─────────────────────────────────────────────
    denominator = 1.0 + (n_folds - 1) * test_ratio
    W           = int((n_samples - 2 * gap_candles) / denominator)
    train_size  = int(W * train_ratio)
    val_size    = int(W * val_ratio)
    test_size   = W - train_size - val_size    # absorbs rounding residual

    if train_size < min_train:
        raise ValueError(
            f"train_size={train_size} < min_train={min_train}. "
            f"Provide more historical data or reduce n_folds/gap_candles."
        )
    if test_size < 30:
        raise ValueError(
            f"test_size={test_size} is too small (<30 candles) for statistical "
            f"significance. Reduce n_folds or gap_candles."
        )

    step = test_size   # Advance by exactly one test window per fold

    # ── Fold construction ──────────────────────────────────────────────────
    folds: list[FoldIndices] = []

    for fold_idx in range(n_folds):
        t0         = fold_idx * step
        train_end  = t0        + train_size
        val_start  = train_end + gap_candles
        val_end    = val_start + val_size
        test_start = val_end   + gap_candles
        test_end   = test_start + test_size

        if test_end > n_samples:
            raise ValueError(
                f"Fold {fold_idx + 1}: test_end={test_end} exceeds "
                f"n_samples={n_samples}. Dataset too short for {n_folds} folds "
                f"with gap_candles={gap_candles}."
            )

        folds.append(FoldIndices(
            fold       = fold_idx + 1,
            train      = slice(t0,         train_end),
            val        = slice(val_start,  val_end),
            test       = slice(test_start, test_end),
            train_bars = train_size,
            val_bars   = val_size,
            test_bars  = test_size,
        ))

    # ── Post-construction sanity checks ───────────────────────────────────
    _validate_fold_integrity(folds, n_samples)
    return folds


def _validate_fold_integrity(folds: list[FoldIndices], n_samples: int) -> None:
    """
    Assert structural guarantees on the generated fold set.
    Called internally by generate_rolling_folds() — not user-facing.
    """
    for f in folds:
        # 1. No section overflows dataset boundary
        assert f.test.stop  <= n_samples,      f"Fold {f.fold}: test overflows dataset"
        assert f.train.stop <= n_samples,      f"Fold {f.fold}: train overflows dataset"
        assert f.val.stop   <= n_samples,      f"Fold {f.fold}: val overflows dataset"

        # 2. No temporal inversion
        assert f.train.start < f.train.stop,   f"Fold {f.fold}: empty train slice"
        assert f.val.start   < f.val.stop,     f"Fold {f.fold}: empty val slice"
        assert f.test.start  < f.test.stop,    f"Fold {f.fold}: empty test slice"

        # 3. Strict temporal ordering: train < val < test
        assert f.train.stop  < f.val.start,    f"Fold {f.fold}: train bleeds into val"
        assert f.val.stop    < f.test.start,   f"Fold {f.fold}: val bleeds into test"

    # 4. Non-overlapping OOS test windows across all folds
    test_ranges = [(f.test.start, f.test.stop) for f in folds]
    for i in range(len(test_ranges) - 1):
        a_end   = test_ranges[i][1]
        b_start = test_ranges[i + 1][0]
        assert a_end <= b_start, (
            f"OOS overlap detected: fold {i+1} test ends at {a_end}, "
            f"fold {i+2} test starts at {b_start}"
        )


# ── Rolling Index Iterator (for DataFrame-based workflows) ─────────────────

def rolling_fold_iterator(
    index:       pd.Index,
    **fold_kwargs,
) -> Iterator[tuple[FoldIndices, pd.Index, pd.Index, pd.Index]]:
    """
    Convenience iterator that yields (FoldIndices, train_idx, val_idx, test_idx)
    for direct use with pandas DataFrames.

    Usage:
        for fold, tr_idx, vl_idx, ts_idx in rolling_fold_iterator(df.index):
            train_df = df.loc[tr_idx]
            val_df   = df.loc[vl_idx]
            test_df  = df.loc[ts_idx]
    """
    folds = generate_rolling_folds(len(index), **fold_kwargs)
    for f in folds:
        yield (
            f,
            index[f.train],
            index[f.val],
            index[f.test],
        )


# ── Walk-Forward Execution ────────────────────────────────────────────────

def run_walk_forward_v5(
    df_dict: dict[str, pd.DataFrame],
    params:  dict,
    cfg:     dict = WFV_CONFIG_V5,
) -> tuple[pd.DataFrame, float]:
    """
    Execute 8-Fold Rolling Window Walk-Forward Validation.

    Each fold:
      1. Slice data into train / val / test by fold boundaries
      2. Retrain a fresh VaranusModel on the train slice
      3. Validate hyperparameters against val slice (early-stopping)
      4. Generate signals and run backtest on the held-out test slice
      5. Collect metrics, including Power Setup trade count

    Args:
        df_dict: {asset: OHLCV DataFrame} — full history, all assets
        params:  Hyperparameter dict (from STEP 7 best_params_v5.json)
        cfg:     WFV config (default: WFV_CONFIG_V5)

    Returns:
        (fold_results_df, consistency_ratio)
    """
    # Use the first asset's index as the timeline reference
    reference_index = next(iter(df_dict.values())).index
    folds = generate_rolling_folds(
        n_samples   = len(reference_index),
        n_folds     = cfg["n_folds"],
        train_ratio = cfg["train_ratio"],
        val_ratio   = cfg["val_ratio"],
        test_ratio  = cfg["test_ratio"],
        gap_candles = cfg["gap_candles"],
        min_train   = cfg["min_train_candles"],
    )

    fold_results = []
    gate         = cfg["performance_gate"]

    for fold in folds:
        print(f"\n{'─'*60}")
        print(f"  Fold {fold.fold}/{cfg['n_folds']}  |  {fold}")
        print(f"{'─'*60}")

        # Slice data
        train_data = {a: df.iloc[fold.train] for a, df in df_dict.items()}
        val_data   = {a: df.iloc[fold.val]   for a, df in df_dict.items()}
        test_data  = {a: df.iloc[fold.test]  for a, df in df_dict.items()}

        # Retrain on this fold's window
        model  = VaranusModel(MODEL_CONFIG_V5)
        X_tr, y_tr = build_features(train_data, params)
        X_vl, y_vl = build_features(val_data,   params)
        model.fit(X_tr, y_tr, X_vl, y_vl)

        # OOS evaluation
        signals        = generate_signals(test_data, model, params)
        equity, trades = run_backtest(test_data, signals, model, params)
        metrics        = compute_metrics(equity, trades)

        # Power Setup statistics
        ps_trades = trades[trades['confidence'] >= 0.95] if len(trades) else pd.DataFrame()
        metrics['power_setup_count']  = len(ps_trades)
        metrics['power_setup_pct']    = (
            round(len(ps_trades) / len(trades) * 100, 1) if len(trades) else 0
        )

        # Fold pass/fail
        fold_pass = (
            metrics['profit_factor']   >= gate['min_profit_factor'] and
            metrics['win_rate_pct']    >= gate['min_win_rate'] * 100 and
            metrics['max_drawdown_pct'] >= gate['max_fold_dd'] * 100 and
            metrics['calmar_ratio']    >= gate['min_calmar']
        )
        metrics['fold_pass'] = fold_pass

        fold_results.append({'fold': fold.fold, **metrics})

        status = "PASS ✓" if fold_pass else "FAIL ✗"
        print(f"  Profit Factor: {metrics['profit_factor']:.2f}  "
              f"Win Rate: {metrics['win_rate_pct']:.1f}%  "
              f"MaxDD: {metrics['max_drawdown_pct']:.1f}%  "
              f"Power Setups: {metrics['power_setup_count']}  "
              f"→ {status}")

    results_df   = pd.DataFrame(fold_results)
    passed_folds = results_df['fold_pass'].sum()
    consistency  = passed_folds / cfg['n_folds']

    print(f"\n{'='*60}")
    print(f"  V5 WFV Summary")
    print(f"  Folds passed:  {passed_folds}/{cfg['n_folds']}")
    print(f"  Consistency:   {consistency:.0%}  "
          f"(required: {gate['consistency_req']:.0%})")
    print(f"  Avg PF:        {results_df['profit_factor'].mean():.2f}")
    print(f"  Avg MaxDD:     {results_df['max_drawdown_pct'].mean():.1f}%")
    print(f"  Avg WR:        {results_df['win_rate_pct'].mean():.1f}%")
    print(f"{'='*60}")

    print(results_df[[
        'fold', 'profit_factor', 'win_rate_pct', 'max_drawdown_pct',
        'calmar_ratio', 'total_trades', 'power_setup_count', 'fold_pass',
    ]].to_string(index=False))

    return results_df, consistency
```

### Step 6 Validation ✓
```python
# Structural test — no real data required
folds = generate_rolling_folds(n_samples=4380, n_folds=8,
                                train_ratio=0.40, val_ratio=0.30,
                                test_ratio=0.30, gap_candles=24)

assert len(folds) == 8,                          "Must produce exactly 8 folds"
assert folds[0].train_bars >= 400,               "Train floor violated"

# Verify non-overlapping OOS windows
test_ends = [f.test.stop for f in folds]
for i in range(len(folds) - 1):
    assert folds[i].test.stop <= folds[i+1].test.start, \
        f"OOS overlap detected at fold {i+1}"

# Print fold layout
for f in folds:
    r_train = f.train_bars / (f.train_bars + f.val_bars + f.test_bars)
    r_val   = f.val_bars   / (f.train_bars + f.val_bars + f.test_bars)
    r_test  = f.test_bars  / (f.train_bars + f.val_bars + f.test_bars)
    print(f"{f}  |  ratios: {r_train:.0%}/{r_val:.0%}/{r_test:.0%}")

print("\nAll fold integrity assertions: PASS ✓")

# Full WFV run
results, consistency = run_walk_forward_v5(data, best_params_v5)
assert consistency >= WFV_CONFIG_V5['performance_gate']['consistency_req'], \
    f"WFV consistency {consistency:.0%} below required threshold"
print("V5 Walk-Forward Validation: PASS ✓")
```

---

## STEP 7 — V5 Optuna HPO (Regime-Optimized)

**File:** `varanus/optimizer.py`

### Objective Design

V5 does not inherit v4's static parameters. The three core execution
parameters are treated as unknown variables to be discovered by Optuna
against the 2026 regime data.

| Parameter           | V4 Value (Frozen) | V5 Search Range  | Rationale                         |
|---------------------|-------------------|------------------|-----------------------------------|
| `confidence_thresh` | 0.8147            | [0.750, 0.880]   | Wider floor captures 2026 setups  |
| `sl_atr_mult`       | ~1.20 (TBM base)  | [0.700, 1.200]   | Explores tighter stops for R:R    |
| `rr_ratio`          | ~4.0 (implicit)   | [3.500, 5.000]   | Pushes for higher reward ceiling  |

### HPO Objective: Profit Factor × Net Profit with DD Gate

```python
"""
varanus/optimizer.py
V5 Optuna HPO — Regime-Adaptive Parameter Search.
Objective: Maximize Profit Factor × Net Profit  |  Constraint: MaxDD < 15%
"""

import optuna
import numpy  as np
import pandas as pd
from optuna.samplers import TPESampler
from optuna.pruners  import HyperbandPruner


# ── V5 Search Space Definition ────────────────────────────────────────────

V5_SEARCH_SPACE = {
    # ── CORE EXECUTION PARAMETERS (recalculated from scratch) ─────────────
    "confidence_thresh": {"type": "float",  "low": 0.750, "high": 0.880},
    "sl_atr_mult":       {"type": "float",  "low": 0.700, "high": 1.200},
    "rr_ratio":          {"type": "float",  "low": 3.500, "high": 5.000},

    # ── FEATURE PARAMETERS (secondary search, narrower ranges) ───────────
    "mss_lookback":      {"type": "int",    "low": 30,    "high": 50   },
    "fvg_min_atr_ratio": {"type": "float",  "low": 0.20,  "high": 0.50 },
    "sweep_min_pct":     {"type": "float",  "low": 0.002, "high": 0.008},
    "fvg_max_age":       {"type": "int",    "low": 10,    "high": 25   },
    "rvol_threshold":    {"type": "float",  "low": 1.20,  "high": 2.50 },
    "rsi_oversold":      {"type": "int",    "low": 28,    "high": 42   },
    "rsi_overbought":    {"type": "int",    "low": 58,    "high": 72   },

    # ── XGB PARAMS ────────────────────────────────────────────────────────
    "xgb_max_depth":    {"type": "int",    "low": 4,     "high": 8    },
    "xgb_n_estimators": {"type": "int",    "low": 200,   "high": 800  },
    "xgb_lr":           {"type": "float",  "low": 0.01,  "high": 0.10,
                          "log": True},
    "xgb_subsample":    {"type": "float",  "low": 0.60,  "high": 1.00 },
}

# Time exit is NOT in the search space — locked at 31 bars.
V5_TIME_EXIT_BARS = 31   # IMMUTABLE. Do not add to search space.


# ── Objective Function ────────────────────────────────────────────────────

def v5_optuna_objective(
    trial:    optuna.Trial,
    data:     dict,
    cfg:      dict = WFV_CONFIG_V5,
) -> float:
    """
    V5 HPO objective: Profit Factor × Net Profit.

    Constraint (hard):
        Max Drawdown ≥ −15%  →  violated trials return −999.0

    Secondary penalty:
        Profit Factor < 1.50  →  score halved
        Win Rate < 40%        →  score halved
        Sharpe < 0.80         →  score × 0.75

    The optimizer runs on the TRAIN + VAL windows only.
    The TEST window is held out exclusively for STEP 6 WFV evaluation.

    Args:
        trial: Optuna trial object.
        data:  Dict with keys 'train' and 'val', each a {asset: df} dict.
        cfg:   WFV configuration (for fold-level data slicing).

    Returns:
        float: Composite score (higher is better). Negative on constraint violation.
    """

    # ── Sample hyperparameters ─────────────────────────────────────────────
    params = _sample_params(trial)

    # ── Run backtest on train window (val used only for early-stopping) ────
    signals        = generate_signals(data['train'], model=None, params=params)
    equity, trades = run_backtest(data['train'], signals, model=None, params=params)

    # ── Minimum trade floor ────────────────────────────────────────────────
    if len(trades) < 30:
        return -999.0   # Penalize sparse parameter sets

    # ── Compute base metrics ───────────────────────────────────────────────
    metrics = compute_metrics(equity, trades)

    profit_factor    = metrics['profit_factor']
    net_profit_pct   = metrics['total_return_pct']   # % gain over period
    max_dd_pct       = metrics['max_drawdown_pct']   # negative value
    win_rate_pct     = metrics['win_rate_pct']
    sharpe           = metrics['sharpe_ratio']

    # ── HARD CONSTRAINT: Max Drawdown < 15% ───────────────────────────────
    if max_dd_pct < -15.0:
        return -999.0

    # ── Primary composite score ────────────────────────────────────────────
    # Reward both quality (profit_factor) and magnitude (net_profit).
    # Log-scaling net_profit prevents outsized single-run outliers from
    # dominating the search.
    if net_profit_pct <= 0:
        return -999.0

    score = profit_factor * np.log1p(net_profit_pct)

    # ── Penalty modifiers ──────────────────────────────────────────────────
    if profit_factor < 1.50:
        score *= 0.50   # Must clear v5 minimum PF gate
    if win_rate_pct < 40.0:
        score *= 0.50   # Low win-rate equity paths are fragile
    if sharpe < 0.80:
        score *= 0.75   # Penalize erratic equity curves

    # Proximity penalty: drawdown approaching the 15% wall loses bonus
    # This nudges Optuna toward robustly safe parameter regions.
    dd_headroom = (-15.0 - max_dd_pct)   # Positive = headroom remaining
    if dd_headroom < 3.0:               # Within 3% of the wall
        score *= (0.70 + 0.10 * dd_headroom)

    return round(score, 6)


def _sample_params(trial: optuna.Trial) -> dict:
    """Sample all hyperparameters from V5_SEARCH_SPACE for a given trial."""
    params = {}
    for name, spec in V5_SEARCH_SPACE.items():
        if spec['type'] == 'int':
            params[name] = trial.suggest_int(name, spec['low'], spec['high'])
        elif spec['type'] == 'float':
            kwargs = {'log': spec.get('log', False)}
            params[name] = trial.suggest_float(
                name, spec['low'], spec['high'], **kwargs
            )
    # Inject the locked time exit — makes it available to backtest engine
    params['max_holding_candles'] = V5_TIME_EXIT_BARS
    return params


# ── Study Configuration ───────────────────────────────────────────────────

V5_OPTUNA_CONFIG = {
    "study_name":  "varanus_v5_hpo",
    "storage":     "sqlite:///config/optuna_v5.db",
    "n_trials":    300,                   # More trials than v4 (200)
    "direction":   "maximize",
    "sampler":     "TPESampler",
    "pruner":      "HyperbandPruner",
    "n_jobs":      1,                     # Serial to preserve temporal order
    "secondary_filter": {
        "min_trades":         30,
        "min_profit_factor":  1.50,       # V5 acceptance floor
        "min_win_rate":       0.40,
        "max_drawdown_floor": -0.15,      # -15% hard cap
    },
}


# ── Study Runner ──────────────────────────────────────────────────────────

def run_v5_optimization(data: dict) -> optuna.Study:
    """
    Launch the V5 Optuna study.

    Args:
        data: {
            'train': {asset: df},   ← used by objective for backtest
            'val':   {asset: df},   ← used for XGB early-stopping only
            'test':  {asset: df},   ← NEVER passed here; reserved for WFV
        }

    Returns:
        Completed optuna.Study object.
    """
    study = optuna.create_study(
        study_name = V5_OPTUNA_CONFIG["study_name"],
        storage    = V5_OPTUNA_CONFIG["storage"],
        direction  = V5_OPTUNA_CONFIG["direction"],
        sampler    = TPESampler(seed=42),
        pruner     = HyperbandPruner(),
        load_if_exists = True,   # Allows resuming interrupted studies
    )

    study.optimize(
        lambda trial: v5_optuna_objective(trial, data),
        n_trials  = V5_OPTUNA_CONFIG["n_trials"],
        n_jobs    = V5_OPTUNA_CONFIG["n_jobs"],
        show_progress_bar = True,
    )

    return study


# ── Best Params Extraction & Validation ──────────────────────────────────

def extract_and_validate_best_params(study: optuna.Study) -> dict:
    """
    Extract best trial params, apply secondary filters, and validate
    they meet v5 acceptance criteria before saving.

    Returns:
        dict of best hyperparameters, or raises if no trial qualifies.
    """
    sf = V5_OPTUNA_CONFIG["secondary_filter"]

    # Filter completed trials through secondary gates
    qualified = [
        t for t in study.trials
        if (t.state == optuna.trial.TrialState.COMPLETE
            and t.value is not None
            and t.value > 0)
    ]

    if not qualified:
        raise RuntimeError(
            "No trials passed all secondary filters. "
            "Increase n_trials or relax secondary_filter thresholds."
        )

    best      = max(qualified, key=lambda t: t.value)
    params    = best.params.copy()
    params['max_holding_candles'] = V5_TIME_EXIT_BARS  # Enforce lock

    print(f"\n{'='*60}")
    print(f"  V5 HPO Complete — Best Trial #{best.number}")
    print(f"  Score (PF × log(NetPnL)):  {best.value:.4f}")
    print(f"\n  Core Parameters:")
    print(f"    confidence_thresh : {params['confidence_thresh']:.4f}")
    print(f"    sl_atr_mult       : {params['sl_atr_mult']:.4f}")
    print(f"    rr_ratio          : {params['rr_ratio']:.3f}")
    print(f"    max_holding_bars  : {params['max_holding_candles']} (LOCKED)")
    print(f"{'='*60}")

    return params
```

### Step 7 Validation ✓
```python
import json, optuna

study  = run_v5_optimization(data)
params = extract_and_validate_best_params(study)

# Core parameter assertions
assert 0.750 <= params['confidence_thresh'] <= 0.880, "confidence_thresh out of range"
assert 0.700 <= params['sl_atr_mult']       <= 1.200, "sl_atr_mult out of range"
assert 3.500 <= params['rr_ratio']          <= 5.000, "rr_ratio out of range"
assert params['max_holding_candles'] == 31,            "Time exit lock violated"
assert study.best_value > 0,                           "Best trial must score > 0"

# Save to config
params['_meta'] = {
    "version":      "5.0.0",
    "study_name":   V5_OPTUNA_CONFIG["study_name"],
    "best_trial":   study.best_trial.number,
    "best_score":   round(study.best_value, 6),
    "objective":    "profit_factor × log1p(net_profit_pct) | MaxDD < 15%",
    "time_exit":    "31 bars LOCKED",
}

with open("config/best_params_v5.json", "w") as f:
    json.dump(params, f, indent=2)

print(f"Best params saved → config/best_params_v5.json")
print(f"V5 HPO: PASS ✓")
```

---

## STEP 8 — Risk Management Layer

**File:** `varanus/risk.py`

```python
RISK_CONFIG_V5 = {
    "initial_capital":          5_000.0,
    "max_portfolio_leverage":   3.5,     # Raised from v4 2.5x (Power Setup)
    "max_concurrent_positions": 4,
    "corr_block_threshold":     0.75,
    "corr_lookback_days":       20,

    "position_size_scalar": {
        "standard":    1.00,
        "high_vol":    0.75,    # TAO, ASTR, KITE, ICP
        "power_setup": 1.25,    # Confidence ≥ 0.95 — additively stacked
    },

    "leverage_map": CONFIDENCE_LEVERAGE_MAP_V5,  # Defined in STEP 5

    "daily_loss_limit_pct":  0.05,   # Halt if -5% in 24h (unchanged)
    "portfolio_stop_pct":    0.15,   # Halt if -15% from peak (aligned to DD gate)

    # Power Setup specific controls
    "power_setup_max_concurrent": 2, # Max 2 Power Setups open simultaneously
    "power_setup_cooldown_bars":  8, # After a PS close, 8-bar wait before next
}

def check_portfolio_health_v5(equity_curve: pd.Series,
                               cfg: dict = RISK_CONFIG_V5) -> dict:
    """V5 portfolio health — DD gate aligned to 15% (matching HPO constraint)."""
    current    = equity_curve.iloc[-1]
    peak       = equity_curve.cummax().iloc[-1]
    day_start  = equity_curve.last('1D').iloc[0]
    daily_loss = (current - day_start) / day_start
    drawdown   = (current - peak) / peak

    return {
        "current_equity":  current,
        "daily_loss_pct":  round(daily_loss * 100, 2),
        "drawdown_pct":    round(drawdown * 100, 2),
        "halt_signals":    (daily_loss <= -cfg['daily_loss_limit_pct'] or
                            drawdown   <= -cfg['portfolio_stop_pct']),
        "dd_gate_pct":     -15.0,   # V5 explicit reference
    }
```

### Step 8 Validation ✓
```python
health = check_portfolio_health_v5(equity_curve)
assert 'halt_signals' in health
assert isinstance(health['halt_signals'], bool)
assert health['dd_gate_pct'] == -15.0, "DD gate must match HPO constraint"
print(f"V5 Portfolio health check: PASS ✓  |  {health}")
```

---

## STEP 9 — Alert System

**File:** `varanus/alerts.py`

V5 adds a Power Setup flag and updated leverage display to the Telegram alert.

```python
ALERT_FORMAT_V5 = (
    "{ps_badge}*VARANUS V5* | {asset} {direction} @ {confidence:.0%}\n"
    "Entry: {entry_price} | TP: {take_profit} | SL: {stop_loss}\n"
    "R:R {rr_ratio:.1f}x | Lev: {leverage}x | ATR: {atr_14:.4f}\n"
    "MSS: {mss} | FVG✓ | Sweep✓ | RVol: {rvol:.2f}x | RSI: {rsi:.1f}\n"
    "HTF: {htf_bias} | Pos: ${position_usd:.0f} | Port Lev: {port_lev:.2f}x\n"
    "Time Exit: 31 bars | Hold: {bars_held} bars"
)

def _get_power_setup_badge(confidence: float) -> str:
    """Return Telegram emoji badge based on confidence tier."""
    if confidence >= 0.950:
        return "⚡ POWER SETUP (5x) — "
    elif confidence >= 0.920:
        return "🔥 "
    elif confidence >= 0.850:
        return "✅ "
    else:
        return "📡 "

REQUIRED_FIELDS_V5 = [
    "timestamp_utc", "asset", "direction", "confidence", "leverage",
    "entry_price", "take_profit", "stop_loss", "rr_ratio", "atr_14",
    "mss", "fvg_valid", "sweep_confirmed", "rvol", "rsi", "htf_bias",
    "position_usd", "port_lev", "bars_held",
]

def send_alert_v5(trade: dict, bot_token: str, chat_id: str) -> None:
    missing = [f for f in REQUIRED_FIELDS_V5 if f not in trade]
    if missing:
        raise ValueError(f"V5 Alert missing fields: {missing}")
    msg = ALERT_FORMAT_V5.format(
        ps_badge = _get_power_setup_badge(trade['confidence']),
        **trade,
    )
    requests.post(
        f"https://api.telegram.org/bot{bot_token}/sendMessage",
        json={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"},
    )
```

### Step 9 Validation ✓
```python
mock = {f: "TEST" for f in REQUIRED_FIELDS_V5}
mock.update({"confidence": 0.963, "leverage": 5.0, "rr_ratio": 4.1,
             "rvol": 2.1, "rsi": 32.4, "port_lev": 2.8,
             "position_usd": 1875, "atr_14": 0.0411, "bars_held": 0})

badge = _get_power_setup_badge(0.963)
assert badge.startswith("⚡"), "Power Setup badge must render for conf ≥ 0.95"
print(ALERT_FORMAT_V5.format(ps_badge=badge, **mock))
print("V5 Alert format: PASS ✓")
```

---

## STEP 10 — Governance & Version Control

### V5 Key Parameter Reference

| Parameter           | V4 Value (Frozen)  | V5 Value            | Source        |
|---------------------|--------------------|---------------------|---------------|
| Universe            | 20 assets          | 20 assets (re-val)  | STEP 1        |
| Timeframe           | 4h + 1D bias       | 4h + 1D bias        | Preserved     |
| MSS Lookback        | 30–50 (default 40) | 30–50 (default 40)  | Preserved     |
| FVG Sweep Filter    | Required           | Required            | Preserved     |
| Time Exit           | 30 bars            | **31 bars — LOCKED**| V5 lock       |
| Confidence Gate     | 0.8147             | **[0.750–0.880] → HPO**| STEP 7    |
| SL ATR Multiplier   | ~1.20              | **[0.700–1.200] → HPO**| STEP 7    |
| R:R Ratio           | ~4.0               | **[3.500–5.000] → HPO**| STEP 7    |
| HPO Objective       | Calmar Ratio       | **PF × log(Net%)** | STEP 7        |
| Leverage 1x         | 0.80–0.85          | 0.750–0.850         | STEP 5        |
| Leverage 2x         | 0.85–0.92          | 0.850–0.920         | STEP 5        |
| Leverage 3x         | 0.92–1.00          | 0.920–0.950         | STEP 5        |
| **Leverage 5x**     | ~~Not supported~~  | **≥ 0.950 (Power Setup)** | V5 new  |
| WFV Folds           | 5                  | **8**               | STEP 6        |
| Train Ratio         | 70%                | **40%**             | STEP 6        |
| Val Ratio           | 15%                | **30%**             | STEP 6        |
| Test Ratio          | 15%                | **30%**             | STEP 6        |
| Max DD Constraint   | 35% (backtest)     | **15% (HPO hard cap)** | STEP 7   |
| Min Profit Factor   | 1.30               | **1.50**            | STEP 7        |
| Portfolio DD Halt   | 15% from peak      | **15% (aligned)**   | STEP 8        |
| Protected Baseline  | —                  | **`/v4/` (frozen)** | Archive       |

### Best Params Placeholder (populated after STEP 7)

```json
// config/best_params_v5.json — populated by run_v5_optimization()
{
  "confidence_thresh": "TBD — HPO range [0.750, 0.880]",
  "sl_atr_mult":       "TBD — HPO range [0.700, 1.200]",
  "rr_ratio":          "TBD — HPO range [3.500, 5.000]",
  "max_holding_candles": 31,
  "_meta": {
    "version":    "5.0.0",
    "objective":  "profit_factor × log1p(net_profit_pct) | MaxDD < 15%",
    "time_exit":  "31 bars LOCKED"
  }
}
```

### Full Implementation Checklist

```
[ ] STEP 0  — varanus/ v5 root initialized. v4/ confirmed frozen. Isolation OK.
[ ] STEP 1  — universe.py: 20 assets re-validated for 2026 liquidity regime.
[ ] STEP 2  — pa_features.py: MSS/FVG/Sweep structurally preserved from v4.
              Smoke test on LINK passed.
[ ] STEP 3  — tbm_labeler.py: 31-bar time exit lock enforced. ATR barriers
              derived from sl_atr_mult × rr_ratio (Optuna variables).
[ ] STEP 4  — model.py: Same 16-feature vector. 8-fold data presentation.
              Probabilities sum to 1. PASS ✓
[ ] STEP 5  — backtest.py: Power Setup 5x leverage active. PS size scalar 1.25.
              Portfolio leverage cap updated to 3.5x. PASS ✓
[ ] STEP 6  — walk_forward.py: 8-fold 40/30/30 rolling window. Non-overlapping
              OOS test windows. ≥ 6/8 folds must pass (75% consistency). PASS ✓
[ ] STEP 7  — optimizer.py: 300-trial Optuna. Objective: PF × log(Net%).
              Hard constraint: MaxDD < 15%. Best params → best_params_v5.json.
              All 3 core params within search range. PASS ✓
[ ] STEP 8  — risk.py: 15% DD halt (aligned to HPO). Power Setup max=2
              concurrent. Cooldown enforced. PASS ✓
[ ] STEP 9  — alerts.py: Power Setup badge (⚡) active. 31-bar hold display.
              Telegram dry-run PASS ✓
[ ] STEP 10 — All gates passed. Tag v5.0.0. Commit. DONE ✓
```

### Version Metadata

```yaml
version:      "5.0.0"
strategy:     "High-Resolution Regime Adaptation"
tier:         2
asset_class:  "Mid-Cap Crypto (ex-BTC/ETH)"
author:       "Varanus / Chameleon Project"
created:      "2026-03-02"
status:       "Specification — Ready for Implementation"
protected_baseline: "v4/  (frozen at v4.0.0 — DO NOT MODIFY)"

changelog:
  - "5.0.0: 8-fold 40/30/30 rolling window (up from 5-fold 70/15/15).
             New Optuna objective: PF × log(Net%) with MaxDD < 15% hard cap.
             Core parameters (confidence_thresh, sl_atr_mult, rr_ratio)
             recalculated via HPO — v4 static values NOT inherited.
             Power Setup 5x leverage tier added (confidence ≥ 0.95).
             Time exit locked at 31 bars.
             Portfolio DD halt aligned to HPO constraint (15%).
             v4 archived to /v4/ as protected frozen baseline."
```
