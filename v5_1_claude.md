# Varanus v5.1 — "The Hunter"
> **Evolved from v4.0 "Structural Bridge". This version replaces passive barrier management
> with adaptive active execution and a regime-aware 8-fold rolling validation framework.**
> The v4 `/v4/` directory is the **frozen "Sniper" baseline** — it must not be modified.

---

## Architecture Lineage

```
chameleon/          <- Base system. DO NOT MODIFY.
v4/                 <- FROZEN. "Sniper" baseline. DO NOT MODIFY.
varanus/ (v5.1)     <- This module. Active overrides on top of v4 signal core.
|   +-- v5_1_claude.md          <- THIS FILE. Single source of truth for v5.1.
|   +-- universe.py             <- Unchanged from v4 (Tier 2, 20 assets)
|   +-- pa_features.py          <- Core signals UNCHANGED (MSS/FVG/Sweep)
|   +-- tbm_labeler.py          <- Barrier config updated (wider TP multiplier)
|   +-- model.py                <- XGBoost re-tuned for 40% train windows
|   +-- backtest.py             <- Hunter Mode execution layer added
|   +-- walk_forward.py         <- UPGRADED: 8-fold rolling, 40/30/30 split
|   +-- optimizer.py            <- UPGRADED: Hunter Efficiency objective
|   +-- risk.py                 <- Unchanged from v4
|   +-- alerts.py               <- Unchanged from v4
```

---

## Phase 1 — Project Continuity

### Signal Core: Preserved from v4
The following v4 logic is the **unmodified signal foundation** for v5.1:

| Component | v4 Source | v5.1 Status |
|---|---|---|
| MSS Detection | `v4/pa_features.py` | **UNCHANGED** — 40-candle default, body filter, HTF bias |
| FVG + Liquidity Sweep | `v4/pa_features.py` | **UNCHANGED** — sweep required, ATR ratio filter |
| Chameleon Confirmations | `v4/pa_features.py` | **UNCHANGED** — RVol >= 1.5x, RSI zones, EMA alignment |
| Triple-Barrier Labeling | `v4/tbm_labeler.py` | **UPDATED** — TP multiplier range expanded to [3.5x-6.0x] |
| Universe | `v4/universe.py` | **UNCHANGED** — 20 Tier 2 assets, HIGH_VOL_SUBTIER |
| Risk Layer | `v4/risk.py` | **UNCHANGED** — portfolio caps, circuit breakers |

> **Rule:** Any change to MSS/FVG/Sweep logic requires explicit versioning as v5.2+.
> The FVG `require_sweep: True` flag is sacred. Never disable it.

### What v5.1 Changes
1. Walk-Forward: 5-fold 70/15/15 -> **8-fold 40/30/30 rolling**
2. Optuna Objective: Calmar -> **Hunter Efficiency** (Net Profit / MaxDD, -12% DD penalty)
3. Leverage ceiling: 3x -> **5x for high-conviction signals** (confidence >= threshold)
4. Execution model: Static barriers -> **Active management** (decay exit, breakeven shift, MSS invalidation)

---

## Phase 2 — 8-Fold Rolling Window Validation

### Why 8 Folds / 40-30-30?
The 5-fold 70/15/15 split in v4 over-indexed on 2023-2024 data for training.
With 2026 now the live regime, a **40% train window forces the model to learn on recent
structure** rather than relying on older pattern memory. The larger 30% test slice gives
a statistically meaningful OOS evaluation per fold.

### Configuration

```python
WFV_CONFIG_V51 = {
    "n_folds":           8,
    "method":            "rolling_window",
    "shuffle":           False,          # NEVER shuffle. Temporal integrity sacred.
    "train_ratio":       0.40,           # Changed from v4's 0.70
    "val_ratio":         0.30,           # Changed from v4's 0.15
    "test_ratio":        0.30,           # Changed from v4's 0.15
    "min_train_candles": 800,            # ~133 days on 4h (reduced from 1000)
    "gap_candles":       24,             # 4-day embargo. Unchanged.
    "performance_gate": {
        "min_hunter_efficiency": 0.60,   # Net Profit / |MaxDD| >= 0.60
        "min_win_rate":          0.43,
        "max_fold_dd":          -0.12,   # Hunter hard cap: -12% per fold
        "consistency_req":       0.75,   # >= 75% of 8 folds must pass
    },
}
```

### Rolling Window Splitter

**File:** `varanus/walk_forward.py` — replace `_generate_folds` and `WFV_CONFIG`

```python
import pandas as pd
import numpy as np
from typing import List, Tuple

WFV_CONFIG_V51 = {
    "n_folds":           8,
    "method":            "rolling_window",
    "shuffle":           False,
    "train_ratio":       0.40,
    "val_ratio":         0.30,
    "test_ratio":        0.30,
    "min_train_candles": 800,
    "gap_candles":       24,
    "performance_gate": {
        "min_hunter_efficiency": 0.60,
        "min_win_rate":          0.43,
        "max_fold_dd":          -0.12,
        "consistency_req":       0.75,
    },
}


def _generate_folds_v51(
    df_dict: dict[str, pd.DataFrame],
    cfg: dict,
) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex]]:
    """
    8-Fold Rolling Window Splitter for Hunter v5.1.

    Strictly enforces 40% Train / 30% Val / 30% Test per fold.
    Each fold slides forward in time -- no anchoring, no data reuse.
    Embargo gaps between splits prevent lookahead leakage.

    Returns
    -------
    List of (train_idx, val_idx, test_idx) as DatetimeIndex tuples.
    """
    n_folds = cfg["n_folds"]
    gap     = cfg["gap_candles"]

    # Build global timeline across all assets
    all_ts = set()
    for df in df_dict.values():
        all_ts.update(df.index)
    global_idx = pd.DatetimeIndex(sorted(all_ts))
    total_len  = len(global_idx)

    t_r = cfg["train_ratio"]
    v_r = cfg["val_ratio"]
    s_r = cfg["test_ratio"]

    assert abs(t_r + v_r + s_r - 1.0) < 1e-9, \
        f"Train/Val/Test ratios must sum to 1.0, got {t_r + v_r + s_r}"

    # Compute per-fold window size.
    # We divide total_len into n_folds non-overlapping test windows,
    # then prepend the train+val window before each test window.
    fold_window = int(total_len / (n_folds * s_r + (t_r + v_r)))
    train_len   = int(fold_window * t_r)
    val_len     = int(fold_window * v_r)
    test_len    = int(fold_window * s_r)

    # Absorb gap cost into train/val lengths
    train_len -= gap
    val_len   -= gap

    if train_len < cfg["min_train_candles"]:
        raise ValueError(
            f"train_len={train_len} < min_train_candles={cfg['min_train_candles']}. "
            f"Need more data or fewer folds."
        )

    # Each fold advances forward by exactly one test_len (non-overlapping OOS windows)
    step_size   = test_len
    single_fold = train_len + gap + val_len + gap + test_len

    required = single_fold + (n_folds - 1) * step_size
    if required > total_len:
        raise ValueError(
            f"Insufficient data: need {required} bars for {n_folds} folds, "
            f"have {total_len}. Reduce n_folds or gap_candles."
        )

    folds = []
    for i in range(n_folds):
        # Fold 0 = oldest data; Fold 7 = most recent (closest to live regime)
        test_start  = (total_len - n_folds * step_size) + i * step_size
        test_end    = test_start + test_len

        val_end     = test_start - gap
        val_start   = val_end - val_len

        train_end   = val_start - gap
        train_start = train_end - train_len

        if train_start < 0:
            print(f"  [WFV] Fold {i+1}: skipped (insufficient history, "
                  f"train_start={train_start})")
            continue

        train_slice = global_idx[train_start:train_end]
        val_slice   = global_idx[val_start:val_end]
        test_slice  = global_idx[test_start:test_end]

        if len(train_slice) == 0 or len(val_slice) == 0 or len(test_slice) == 0:
            print(f"  [WFV] Fold {i+1}: skipped (empty slice -- "
                  f"train={len(train_slice)} val={len(val_slice)} "
                  f"test={len(test_slice)})")
            continue

        actual_t = len(train_slice) / total_len
        actual_v = len(val_slice)   / total_len
        actual_s = len(test_slice)  / total_len
        print(f"  [WFV] Fold {i+1}: "
              f"Train {train_slice[0].date()}-->{train_slice[-1].date()} "
              f"({actual_t:.0%}) | "
              f"Val {val_slice[0].date()}-->{val_slice[-1].date()} "
              f"({actual_v:.0%}) | "
              f"Test {test_slice[0].date()}-->{test_slice[-1].date()} "
              f"({actual_s:.0%})")

        folds.append((train_slice, val_slice, test_slice))

    if len(folds) < cfg["n_folds"]:
        print(f"  [WFV] Warning: generated {len(folds)}/{cfg['n_folds']} folds.")

    return folds
```

### Step 6 Validation (v5.1)
```python
results, hunter_eff, all_trades = run_walk_forward_v51(data_4h, data_1d, best_params)
passed = (results['hunter_efficiency'] >= WFV_CONFIG_V51['performance_gate']['min_hunter_efficiency']).sum()
consistency = passed / len(results)
assert consistency >= WFV_CONFIG_V51['performance_gate']['consistency_req'], \
    f"WFV failed: {consistency:.0%} of folds passed (need 75%)"
print("Walk-forward v5.1: PASS")
```

---

## Phase 3 — Hunter Efficiency Objective & Hyperparameter Search

### Hunter Efficiency Metric

```
Hunter Efficiency = Net Profit (USD) / |Max Drawdown (%)|

Penalty: if MaxDD > 12% --> multiply score x 0.40 (hard deterrent)
Floor:   if total_trades < 30 --> return -999 (statistically invalid)
```

The v4 Calmar ratio (CAGR / MaxDD) rewarded high annualised returns but tolerated
slow drawdown recovery. Hunter Efficiency penalises **any fold** that breaches the
-12% drawdown wall, forcing the optimiser to find tight, high-conviction setups.

### 6 Hunter Parameters

| # | Parameter | v4 Range | v5.1 Range | Reason |
|---|---|---|---|---|
| 1 | `confidence_thresh` (entry gate) | [0.78-0.92] | **[0.750-0.880]** | Wider floor for volatile 2026 regime |
| 2 | `sl_atr_mult` (stop loss) | [0.8-1.8] | **[0.700-1.200]** | Tighter ceiling enforces -12% DD constraint |
| 3 | `tp_atr_mult` (take profit) | [2.0-3.5] | **[3.5-6.0]** | Asymmetric R:R; minimum 3.5x ATR for Hunter entries |
| 4 | `leverage_5x_trigger` | N/A | **[0.93-0.98]** | New: confidence gate for 5x leverage tier |
| 5 | `xgb_lr` (learning rate) | [0.01-0.10] | **[0.005-0.08]** | Lower floor prevents overfitting on 40% windows |
| 6 | `xgb_max_depth` | [4-8] | **[3-6]** | Shallower trees for smaller train sets |

### Optuna Objective Function

**File:** `varanus/optimizer.py`

```python
import optuna
import pandas as pd
import numpy as np

from varanus.walk_forward import WFV_CONFIG_V51, _generate_folds_v51, _slice
from varanus.backtest import run_backtest, compute_metrics
from varanus.model import VaranusModel, MODEL_CONFIG
from varanus.pa_features import build_features, compute_atr
from varanus.tbm_labeler import label_trades, TBM_CONFIG

HUNTER_OPTUNA_CONFIG = {
    "n_trials":               300,
    "direction":              "maximize",
    "sampler":                "TPESampler",
    "pruner":                 "HyperbandPruner",
    "min_trades_per_fold":    8,
    "min_total_trades":       30,
    "dd_penalty_threshold":   0.12,   # 12% drawdown triggers penalty
    "dd_penalty_multiplier":  0.40,
}


def hunter_efficiency(net_profit_usd: float, max_dd_pct: float) -> float:
    """
    Hunter Efficiency = Net Profit / |Max Drawdown|

    Parameters
    ----------
    net_profit_usd : Total USD profit across the fold's trade log
    max_dd_pct     : Max drawdown as a fraction (e.g. -0.18 for -18%)
    """
    if max_dd_pct == 0.0:
        return 0.0
    return net_profit_usd / abs(max_dd_pct)


def optuna_objective_hunter(
    trial: optuna.Trial,
    df_dict_4h: dict[str, pd.DataFrame],
    df_dict_1d: dict[str, pd.DataFrame],
    cfg: dict = WFV_CONFIG_V51,
) -> float:
    """
    Hunter Efficiency Optuna Objective for Varanus v5.1.

    Optimises 6 Hunter parameters across an 8-fold rolling walk-forward.
    Penalises any parameter set that causes a fold drawdown > 12%.

    Search Space
    ------------
    1. confidence_thresh   -- entry gate          [0.750 - 0.880]
    2. sl_atr_mult         -- stop loss ATR mult  [0.700 - 1.200]
    3. tp_atr_mult         -- take profit ATR mult [3.500 - 6.000]
    4. leverage_5x_trigger -- 5x lev gate         [0.930 - 0.980]
    5. xgb_lr              -- XGBoost learn rate  [0.005 - 0.080]
    6. xgb_max_depth       -- XGBoost tree depth  [3 - 6]

    Frozen from v4 best_params (not searched):
    mss_lookback, fvg_min_atr_ratio, sweep_min_pct, fvg_max_age,
    rvol_threshold, rsi_oversold, rsi_overbought, max_holding,
    xgb_n_estimators, xgb_subsample.
    """
    # 6 Hunter Parameters
    params = {
        # 1. Entry Gate
        "confidence_thresh":   trial.suggest_float("confidence_thresh",   0.750, 0.880),
        # 2. Stop Loss
        "sl_atr_mult":         trial.suggest_float("sl_atr_mult",         0.700, 1.200),
        # 3. Take Profit (wider range vs v4's 2.0-3.5)
        "tp_atr_mult":         trial.suggest_float("tp_atr_mult",         3.500, 6.000),
        # 4. 5x Leverage Trigger (new in v5.1)
        "leverage_5x_trigger": trial.suggest_float("leverage_5x_trigger", 0.930, 0.980),
        # 5 & 6. XGBoost tuned for 40% training windows
        "xgb_lr":              trial.suggest_float("xgb_lr",              0.005, 0.080, log=True),
        "xgb_max_depth":       trial.suggest_int("xgb_max_depth",         3,     6),

        # Frozen v4 params
        "mss_lookback":        31,
        "fvg_min_atr_ratio":   0.392,
        "sweep_min_pct":       0.00641,
        "fvg_max_age":         22,
        "rvol_threshold":      1.287,
        "rsi_oversold":        36,
        "rsi_overbought":      58,
        "max_holding":         31,
        "xgb_n_estimators":    218,
        "xgb_subsample":       0.957,
    }

    print(f"\n>>> Hunter Trial {trial.number} | "
          f"conf={params['confidence_thresh']:.3f} "
          f"sl={params['sl_atr_mult']:.2f}x "
          f"tp={params['tp_atr_mult']:.2f}x "
          f"5xlev@{params['leverage_5x_trigger']:.3f} "
          f"lr={params['xgb_lr']:.4f} "
          f"depth={params['xgb_max_depth']}")

    try:
        folds = _generate_folds_v51(df_dict_4h, cfg)
        if not folds:
            return -999.0

        fold_scores     = []
        total_trades    = 0
        penalty_applied = False

        for fold_idx, (train_idx, val_idx, test_idx) in enumerate(folds):
            train_4h = _slice(df_dict_4h, train_idx)
            train_1d = _slice(df_dict_1d, train_idx)
            val_4h   = _slice(df_dict_4h, val_idx)
            val_1d   = _slice(df_dict_1d, val_idx)
            test_4h  = _slice(df_dict_4h, test_idx)
            test_1d  = _slice(df_dict_1d, test_idx)

            X_tr_list, y_tr_list = [], []
            X_vl_list, y_vl_list = [], []

            for asset in train_4h:
                if asset not in train_1d:
                    continue
                X = build_features(train_4h[asset], train_1d[asset], asset, params)
                if X.empty:
                    continue
                y = label_trades(train_4h[asset].loc[X.index],
                                 X["mss_signal"], TBM_CONFIG, asset, params)
                y = y.reindex(X.index).fillna(0).astype(int)
                X_tr_list.append(X)
                y_tr_list.append(y)

            for asset in val_4h:
                if asset not in val_1d:
                    continue
                X = build_features(val_4h[asset], val_1d[asset], asset, params)
                if X.empty:
                    continue
                y = label_trades(val_4h[asset].loc[X.index],
                                 X["mss_signal"], TBM_CONFIG, asset, params)
                y = y.reindex(X.index).fillna(0).astype(int)
                X_vl_list.append(X)
                y_vl_list.append(y)

            if not X_tr_list:
                continue

            # Re-tune XGBoost for 40% training window
            model_cfg = {**MODEL_CONFIG}
            model_cfg["xgb_params"] = {
                **MODEL_CONFIG["xgb_params"],
                "max_depth":     params["xgb_max_depth"],
                "learning_rate": params["xgb_lr"],
            }
            model = VaranusModel(model_cfg)
            model.fit(
                pd.concat(X_tr_list), pd.concat(y_tr_list),
                pd.concat(X_vl_list) if X_vl_list else None,
                pd.concat(y_vl_list) if y_vl_list else None,
            )

            # Generate signals on test window
            signals = {}
            for asset, df_4h in test_4h.items():
                if asset not in test_1d:
                    continue
                X_t = build_features(df_4h, test_1d[asset], asset, params)
                if X_t.empty:
                    continue
                probs = model.predict_proba(X_t)
                preds = model.predict(X_t)
                sig_df = pd.DataFrame(index=X_t.index)
                sig_df["confidence"]  = probs.max(axis=1)
                sig_df["direction"]   = preds
                sig_df["entry_price"] = df_4h.loc[X_t.index, "close"]
                sig_df["atr"]         = compute_atr(df_4h.loc[X_t.index], 14)
                sig_df = sig_df[sig_df["direction"] != 0]
                if not sig_df.empty:
                    signals[asset] = sig_df

            if not signals:
                continue

            equity, trades = run_backtest(test_4h, signals, model, params)
            metrics        = compute_metrics(equity, trades)

            fold_trades = metrics["total_trades"]
            fold_dd     = metrics["max_drawdown_pct"] / 100.0
            net_profit  = trades["pnl_usd"].sum() if not trades.empty else 0.0

            if fold_trades < HUNTER_OPTUNA_CONFIG["min_trades_per_fold"]:
                continue

            fold_he = hunter_efficiency(net_profit, fold_dd)

            # Hard penalty: drawdown exceeds -12%
            if abs(fold_dd) > HUNTER_OPTUNA_CONFIG["dd_penalty_threshold"]:
                fold_he *= HUNTER_OPTUNA_CONFIG["dd_penalty_multiplier"]
                penalty_applied = True

            fold_scores.append(fold_he)
            total_trades += fold_trades

            # Optuna pruning
            trial.report(float(np.mean(fold_scores)), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if total_trades < HUNTER_OPTUNA_CONFIG["min_total_trades"]:
            print(f"  -> Penalty: only {total_trades} total trades.")
            return -999.0

        if not fold_scores:
            return -999.0

        mean_he = float(np.mean(fold_scores))
        flag    = " [DD PENALTY]" if penalty_applied else ""
        print(f"  -> Trial {trial.number} | Hunter Efficiency: {mean_he:.3f}"
              f" | Folds scored: {len(fold_scores)}/8"
              f" | Trades: {total_trades}{flag}")
        return mean_he

    except optuna.TrialPruned:
        raise
    except Exception as exc:
        print(f"  -> Trial {trial.number} failed: {exc}")
        return -999.0


def run_hunter_optimization(
    df_dict_4h: dict[str, pd.DataFrame],
    df_dict_1d: dict[str, pd.DataFrame],
    n_trials: int = 300,
    study_name: str = "varanus_v51_hunter",
) -> optuna.Study:
    """
    Launch the Hunter Efficiency Optuna study.

    Usage
    -----
    study = run_hunter_optimization(data_4h, data_1d, n_trials=300)
    print(f"Best Hunter Efficiency: {study.best_value:.3f}")
    print(f"Best params: {study.best_params}")
    """
    study = optuna.create_study(
        study_name = study_name,
        direction  = "maximize",
        sampler    = optuna.samplers.TPESampler(seed=42),
        pruner     = optuna.pruners.HyperbandPruner(
            min_resource=2, max_resource=8, reduction_factor=3
        ),
    )
    study.optimize(
        lambda t: optuna_objective_hunter(t, df_dict_4h, df_dict_1d),
        n_trials          = n_trials,
        show_progress_bar = True,
    )
    return study
```

### Step 7 Validation (v5.1)
```python
study = run_hunter_optimization(data_4h, data_1d, n_trials=300)
assert study.best_value > 0.60, \
    f"Best Hunter Efficiency {study.best_value:.3f} below threshold 0.60"
print(f"Best Hunter Efficiency: {study.best_value:.3f}")
print(f"Best params: {study.best_params}")
```

---

## Phase 4 — Hunter Mode Active Execution

> **This is the core v4->v5.1 execution upgrade.**
> v4 used "set and forget" TBM barriers. v5.1 actively manages every open position
> on each 4h bar close.

### 4.1 Signal Decay Exit

**Trigger:** At each 4h bar close, re-evaluate the XGBoost confidence for the direction
the trade was entered on. If confidence has dropped by more than 0.35 from entry confidence,
exit at market open of the next bar.

```
Decay Condition:
  entry_confidence - current_confidence > 0.35
  -> EXIT at next bar open (market order)
  -> Log exit reason as "signal_decay"
```

**Rationale:** Mid-cap crypto structure changes faster than Tier 1. A signal that scored
0.87 at entry may be 0.45 two days later -- the thesis is broken before the barrier hits.

### 4.2 Dynamic Breakeven

**Trigger:** Once price moves 75% of the distance from entry to take-profit,
move the stop-loss to entry price (plus a small ATR buffer to absorb noise).

```
Breakeven Condition (Long):
  current_high >= entry_price + 0.75 x (take_profit - entry_price)
  -> stop_loss = entry_price + (0.05 x ATR14)

Breakeven Condition (Short):
  current_low <= entry_price - 0.75 x (entry_price - take_profit)
  -> stop_loss = entry_price - (0.05 x ATR14)
```

**Rationale:** Locks in breakeven on high-conviction moves without closing early,
preserving the full TP upside while eliminating loss risk once deeply in-profit.

### 4.3 MSS Invalidation

**Trigger:** If the 4h MSS detection flips to the opposite direction from the open
trade, close the position immediately at the next bar's open.

```
Invalidation Condition (Long trade):
  detect_mss(df.iloc[-lookback:]).iloc[-1] == -1
  -> EXIT immediately at next bar open
  -> Log exit reason as "mss_invalidation"

Invalidation Condition (Short trade):
  detect_mss(df.iloc[-lookback:]).iloc[-1] == 1
  -> EXIT immediately at next bar open
  -> Log exit reason as "mss_invalidation"
```

**Rationale:** The MSS is the primary structural thesis for every trade.
If the structure has shifted against you on the same timeframe, holding is speculation,
not strategy.

### Active Management Config & Integration

**File:** `varanus/backtest.py` — add to bar loop, called BEFORE standard barrier check.

```python
HUNTER_ACTIVE_CONFIG = {
    "signal_decay": {
        "enabled":         True,
        "rescan_freq":     "4h",      # Evaluate every bar close
        "decay_threshold": 0.35,      # Exit if confidence drop >= 0.35
        "exit_type":       "market",
    },
    "dynamic_breakeven": {
        "enabled":          True,
        "trigger_pct":      0.75,     # Move SL to entry when 75% of TP distance reached
        "buffer_atr_ratio": 0.05,     # SL = entry + (0.05 x ATR) beyond entry
    },
    "mss_invalidation": {
        "enabled":   True,
        "timeframe": "4h",
        "exit_type": "market",
    },
}


def _apply_hunter_active_management(
    bar: pd.Series,
    trade: dict,
    current_proba: float,
    current_mss: int,
    atr: float,
    cfg: dict = HUNTER_ACTIVE_CONFIG,
) -> dict | None:
    """
    Evaluate all three Hunter active management conditions for a single bar.
    Returns an outcome dict if an exit is triggered, else None.
    Called BEFORE standard barrier check in the backtest loop.

    Priority order: MSS Invalidation > Signal Decay > Dynamic Breakeven

    Parameters
    ----------
    bar           : Current OHLCV bar (pd.Series with open/high/low/close)
    trade         : Open trade dict (entry_price, direction, entry_confidence, ...)
    current_proba : Re-evaluated XGBoost confidence for trade direction on this bar
    current_mss   : MSS signal on this bar {-1, 0, 1}
    atr           : ATR14 value for this bar
    cfg           : HUNTER_ACTIVE_CONFIG
    """
    direction = trade["direction"]

    # 1. MSS Invalidation (highest priority)
    if cfg["mss_invalidation"]["enabled"]:
        if direction == 1 and current_mss == -1:
            return {"type": "mss_invalidation", "price": bar["open"]}
        if direction == -1 and current_mss == 1:
            return {"type": "mss_invalidation", "price": bar["open"]}

    # 2. Signal Decay Exit
    if cfg["signal_decay"]["enabled"]:
        decay = trade["entry_confidence"] - current_proba
        if decay >= cfg["signal_decay"]["decay_threshold"]:
            return {"type": "signal_decay", "price": bar["open"]}

    # 3. Dynamic Breakeven (mutates trade in place -- not an exit)
    if cfg["dynamic_breakeven"]["enabled"]:
        buffer      = cfg["dynamic_breakeven"]["buffer_atr_ratio"] * atr
        trigger_pct = cfg["dynamic_breakeven"]["trigger_pct"]

        if direction == 1:
            target_dist = trade["take_profit"] - trade["entry_price"]
            if bar["high"] >= trade["entry_price"] + trigger_pct * target_dist:
                new_sl = trade["entry_price"] + buffer
                if new_sl > trade["stop_loss"]:
                    trade["stop_loss"]             = new_sl
                    trade["breakeven_activated"]   = True

        elif direction == -1:
            target_dist = trade["entry_price"] - trade["take_profit"]
            if bar["low"] <= trade["entry_price"] - trigger_pct * target_dist:
                new_sl = trade["entry_price"] - buffer
                if new_sl < trade["stop_loss"]:
                    trade["stop_loss"]             = new_sl
                    trade["breakeven_activated"]   = True

    return None  # No exit triggered
```

### 4.4 5x Leverage Tier

v4 capped leverage at 3x (confidence >= 0.92). v5.1 adds a **5x tier** for
ultra-high-conviction signals. The `leverage_5x_trigger` threshold is an Optuna parameter.

```python
def get_leverage_v51(confidence: float, leverage_5x_trigger: float) -> float:
    """
    Hunter leverage schedule.
    Below 0.75 -> no trade (entry gate).
    0.75-0.85  -> 1x
    0.85-0.92  -> 2x
    0.92-trigger -> 3x
    trigger-1.0  -> 5x (high-conviction Hunter strike)
    """
    if confidence < 0.750:
        return 0.0   # Below entry gate
    elif confidence < 0.850:
        return 1.0
    elif confidence < 0.920:
        return 2.0
    elif confidence < leverage_5x_trigger:
        return 3.0
    else:
        return 5.0   # Hunter 5x -- Tier 2 hard cap for v5.1
```

---

## Phase 5 — Parameter Reference

### v5.1 vs v4 Summary

| Parameter | v4.0 Value | v5.1 Value | Delta |
|---|---|---|---|
| WFV Folds | 5 | **8** | +3 |
| Train Ratio | 70% | **40%** | -30% |
| Val Ratio | 15% | **30%** | +15% |
| Test Ratio | 15% | **30%** | +15% |
| Optuna Objective | Calmar | **Hunter Efficiency** | changed |
| DD Penalty Threshold | N/A | **-12%** | new |
| Max Leverage | 3x | **5x (gated)** | +2x |
| TP ATR Mult (search) | [2.0-3.5] | **[3.5-6.0]** | +75% floor |
| SL ATR Mult (search) | [0.8-1.8] | **[0.7-1.2]** | tighter ceiling |
| Entry Gate (search) | [0.78-0.92] | **[0.75-0.88]** | wider floor |
| Signal Decay Exit | None | **delta-conf > 0.35** | new |
| Dynamic Breakeven | None | **@75% TP distance** | new |
| MSS Invalidation | None | **4h flip -> exit** | new |

### v4 Parameters Frozen in v5.1

Loaded from `config/best_params.json` — not searched by Optuna:

```python
V4_FROZEN_PARAMS = {
    "mss_lookback":      31,
    "fvg_min_atr_ratio": 0.392,
    "sweep_min_pct":     0.00641,
    "fvg_max_age":       22,
    "rvol_threshold":    1.287,
    "rsi_oversold":      36,
    "rsi_overbought":    58,
    "max_holding":       31,
    "xgb_n_estimators":  218,
    "xgb_subsample":     0.957,
}
```

---

## Implementation Checklist

```
[ ] STEP 0  -- v5_1_claude.md created. v4/ directory confirmed frozen.
[ ] STEP 1  -- universe.py: no changes required (v4 baseline intact).
[ ] STEP 2  -- pa_features.py: no changes required (MSS/FVG/Sweep core preserved).
[ ] STEP 3  -- tbm_labeler.py: update tp_atr_mult search range to [3.5-6.0].
[ ] STEP 4  -- model.py: add get_leverage_v51() with 5x tier.
              Update leverage map to use leverage_5x_trigger param.
[ ] STEP 5  -- backtest.py: integrate _apply_hunter_active_management() into
              the bar loop (before barrier check). Add "signal_decay" and
              "mss_invalidation" to outcome types in compute_metrics().
[ ] STEP 6  -- walk_forward.py: replace WFV_CONFIG + _generate_folds with
              WFV_CONFIG_V51 + _generate_folds_v51.
              Update metrics to report hunter_efficiency per fold.
[ ] STEP 7  -- optimizer.py: replace optuna_objective with
              optuna_objective_hunter. 6-parameter search confirmed.
[ ] STEP 8  -- risk.py: no changes required (v4 portfolio caps retained).
[ ] STEP 9  -- alerts.py: no changes required.
[ ] STEP 10 -- All gates passed. Save best_params_v51.json. Tag v5.1.0.
```

---

## Version Metadata

```yaml
version:      "5.1.0"
codename:     "The Hunter"
parent:       "4.0.0 (Structural Bridge / Sniper)"
strategy:     "Adaptive Active Execution on Tier 2 Smart Money Signals"
tier:         2
asset_class:  "Mid-Cap Crypto (ex-BTC/ETH)"
signal_core:  "MSS + FVG + Liquidity Sweep (v4, unchanged)"
last_updated: "2026-03-05"
status:       "Specification -- Ready for Implementation"
changelog:
  - "5.1.0: 8-fold 40/30/30 rolling WFV. Hunter Efficiency objective (NetProfit/MaxDD,
             -12% DD penalty). 5x leverage tier (Optuna-gated 0.93-0.98). Signal Decay
             Exit (delta-conf > 0.35). Dynamic Breakeven (75% TP trigger). MSS
             Invalidation (4h flip to market exit). TP search [3.5-6.0x]. SL search
             [0.7-1.2x]. v4 signal core (MSS/FVG/Sweep) fully preserved and frozen."
```
