"""
varanus/optimizer.py — V5 Optuna HPO (Step 7).

V5 HPO design:
  Objective : Profit Factor × log1p(Net Return %)
  Constraint: Max Drawdown < −15%  →  returns −999.0

  Per-trial evaluation uses FOLD 1 of the 8-fold scheme:
    - Train model on fold 1 train window
    - Backtest on fold 1 val window (OOS from the model's perspective)
  This is fast (~8× faster than full WFV) and unbiased.
  The full 8-fold WFV (Step 6) evaluates the best params found here.

V5 search ranges (recalculated — do NOT inherit v4 static values):
  confidence_thresh : [0.750, 0.880]
  sl_atr_mult       : [0.700, 1.200]
  rr_ratio          : [3.500, 5.000]
  max_holding_candles: LOCKED at 31 (not searched)
"""

from __future__ import annotations

import json
import logging
import os

import numpy  as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.pruners  import HyperbandPruner

from varanus.model       import VaranusModel, MODEL_CONFIG
from varanus.pa_features import build_features
from varanus.tbm_labeler import label_trades, TBM_CONFIG
from varanus.backtest    import run_backtest, compute_metrics
from varanus.walk_forward import generate_rolling_folds, _slice_by_ts, WFV_CONFIG

logger = logging.getLogger(__name__)

# Suppress Optuna's verbose per-trial logging at INFO level
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── V5 Search Space ───────────────────────────────────────────────────────────

V5_SEARCH_SPACE: dict = {
    # ── CORE EXECUTION PARAMETERS (recalculated for 2026 regime) ─────────────
    "confidence_thresh": {"type": "float", "low": 0.750, "high": 0.880},
    "sl_atr_mult":       {"type": "float", "low": 0.700, "high": 1.200},
    "rr_ratio":          {"type": "float", "low": 3.500, "high": 5.000},

    # ── FEATURE ENGINEERING PARAMETERS ───────────────────────────────────────
    "mss_lookback":      {"type": "int",   "low": 30,    "high": 50   },
    "fvg_min_atr_ratio": {"type": "float", "low": 0.20,  "high": 0.50 },
    "sweep_min_pct":     {"type": "float", "low": 0.002, "high": 0.008},
    "fvg_max_age":       {"type": "int",   "low": 10,    "high": 25   },
    "rvol_threshold":    {"type": "float", "low": 1.20,  "high": 2.50 },
    "rsi_oversold":      {"type": "int",   "low": 28,    "high": 42   },
    "rsi_overbought":    {"type": "int",   "low": 58,    "high": 72   },

    # ── XGB PARAMETERS ────────────────────────────────────────────────────────
    "xgb_max_depth":    {"type": "int",   "low": 4,     "high": 8    },
    "xgb_n_estimators": {"type": "int",   "low": 200,   "high": 800  },
    "xgb_lr":           {"type": "float", "low": 0.01,  "high": 0.10, "log": True},
    "xgb_subsample":    {"type": "float", "low": 0.60,  "high": 1.00 },
}

# max_holding_candles is NOT in search space — locked at 31 bars (V5 LOCK)
V5_TIME_EXIT_BARS: int = 31

# ── Optuna config ─────────────────────────────────────────────────────────────

OPTUNA_CONFIG: dict = {
    "study_name":  "varanus_v5_hpo",
    "storage":     "sqlite:///config/optuna_v5.db",
    "n_trials":    300,
    "direction":   "maximize",
    "n_jobs":      1,
    "secondary_filter": {
        "min_trades":         30,
        "min_profit_factor":  1.50,
        "min_win_rate":       0.40,
        "max_drawdown_floor": -15.0,
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# Parameter sampling
# ═══════════════════════════════════════════════════════════════════════════════

def _sample_params(trial: optuna.Trial) -> dict:
    """Sample all hyperparameters from V5_SEARCH_SPACE for a given trial."""
    params = {}
    for name, spec in V5_SEARCH_SPACE.items():
        if spec["type"] == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif spec["type"] == "float":
            params[name] = trial.suggest_float(
                name, spec["low"], spec["high"],
                log=spec.get("log", False),
            )
    params["max_holding_candles"] = V5_TIME_EXIT_BARS  # Enforce 31-bar lock
    return params


# ═══════════════════════════════════════════════════════════════════════════════
# Objective function
# ═══════════════════════════════════════════════════════════════════════════════

def optuna_objective(
    trial:       optuna.Trial,
    df_dict_4h:  dict[str, pd.DataFrame],
    df_dict_1d:  dict[str, pd.DataFrame],
    cfg:         dict = WFV_CONFIG,
) -> float:
    """
    V5 HPO objective: Profit Factor × log1p(Net Return %).

    Evaluation strategy:
      Uses Fold 1 of the 8-fold rolling window scheme:
        - Model trained on fold 1 TRAIN window
        - Backtest evaluated on fold 1 VAL window (OOS from model)
      This ensures unbiased evaluation at HPO time while keeping
      per-trial cost manageable (1 model fit vs 8 for full WFV).

    Constraint (HARD):
      Max Drawdown >= -15%  →  violated trials return -999.0

    Scoring:
      base_score = profit_factor × log1p(net_return_pct)
      Penalties applied for:
        - profit_factor < 1.50  →  × 0.50
        - win_rate < 40%        →  × 0.50
        - sharpe < 0.80         →  × 0.75
        - DD within 3% of wall  →  × 0.70–1.00 (proximity penalty)

    Args:
        trial:       Optuna trial object.
        df_dict_4h:  Full 4h dataset {asset: DataFrame}.
        df_dict_1d:  Full 1d dataset {asset: DataFrame}.
        cfg:         WFV config (used for fold generation only).

    Returns:
        float: Composite score (higher is better). -999.0 on hard violations.
    """
    params = _sample_params(trial)

    # ── Build global timeline and generate fold 1 ──────────────────────────
    all_ts   = sorted(set().union(*[set(df.index) for df in df_dict_4h.values()]))
    timeline = pd.DatetimeIndex(all_ts)
    N        = len(timeline)

    try:
        folds = generate_rolling_folds(
            n_samples   = N,
            n_folds     = cfg["n_folds"],
            train_ratio = cfg["train_ratio"],
            val_ratio   = cfg["val_ratio"],
            test_ratio  = cfg["test_ratio"],
            gap_candles = cfg["gap_candles"],
            min_train   = cfg["min_train_candles"],
        )
    except ValueError as exc:
        logger.debug(f"Trial {trial.number}: fold generation failed: {exc}")
        return -999.0

    fold1    = folds[0]

    # Slice to fold 1 boundaries
    train_4h = _slice_by_ts(df_dict_4h, timeline, fold1.train)
    val_4h   = _slice_by_ts(df_dict_4h, timeline, fold1.val)
    train_1d = _slice_by_ts(df_dict_1d, timeline, fold1.train)
    val_1d   = _slice_by_ts(df_dict_1d, timeline, fold1.val)

    # ── Build feature matrices ─────────────────────────────────────────────
    X_tr_list, y_tr_list = [], []
    X_vl_list, y_vl_list = [], []

    for asset in train_4h:
        d1 = train_1d.get(asset, train_4h[asset])
        try:
            X = build_features(train_4h[asset], d1, asset, params)
        except Exception:
            continue
        if X.empty:
            continue
        mss = X["mss_signal"]
        y   = label_trades(
            train_4h[asset].loc[X.index], mss, TBM_CONFIG, asset, params
        ).reindex(X.index).fillna(0).astype(int)
        X_tr_list.append(X)
        y_tr_list.append(y)

    for asset in val_4h:
        d1 = val_1d.get(asset, val_4h[asset])
        try:
            X = build_features(val_4h[asset], d1, asset, params)
        except Exception:
            continue
        if X.empty:
            continue
        mss = X["mss_signal"]
        y   = label_trades(
            val_4h[asset].loc[X.index], mss, TBM_CONFIG, asset, params
        ).reindex(X.index).fillna(0).astype(int)
        X_vl_list.append(X)
        y_vl_list.append(y)

    if not X_tr_list:
        return -999.0

    X_train = pd.concat(X_tr_list)
    y_train = pd.concat(y_tr_list)
    X_val   = pd.concat(X_vl_list) if X_vl_list else None
    y_val   = pd.concat(y_vl_list) if y_vl_list else None

    # ── Train model ────────────────────────────────────────────────────────
    model_cfg = MODEL_CONFIG.copy()
    for opt_key in ["xgb_max_depth", "xgb_n_estimators", "xgb_lr", "xgb_subsample"]:
        if opt_key in params:
            model_cfg[opt_key] = params[opt_key]

    model = VaranusModel(model_cfg)
    try:
        model.fit(X_train, y_train, X_val, y_val)
    except Exception as exc:
        logger.debug(f"Trial {trial.number}: model.fit failed: {exc}")
        return -999.0

    # ── Generate signals on val window (OOS) ──────────────────────────────
    from varanus.walk_forward import generate_signals
    signals = generate_signals(val_4h, val_1d, model, params)

    if not signals:
        return -999.0

    # ── Backtest on val window ─────────────────────────────────────────────
    try:
        equity, trades = run_backtest(val_4h, signals, model, params)
    except Exception as exc:
        logger.debug(f"Trial {trial.number}: backtest failed: {exc}")
        return -999.0

    # ── Minimum trade floor ────────────────────────────────────────────────
    if len(trades) < OPTUNA_CONFIG["secondary_filter"]["min_trades"]:
        return -999.0

    # ── Compute metrics and score ──────────────────────────────────────────
    metrics = compute_metrics(equity, trades)

    profit_factor  = metrics["profit_factor"]
    net_return_pct = metrics["total_return_pct"]
    max_dd_pct     = metrics["max_drawdown_pct"]   # negative value
    win_rate_pct   = metrics["win_rate_pct"]
    sharpe         = metrics["sharpe_ratio"]

    # HARD CONSTRAINT: Max Drawdown < -15%
    if max_dd_pct < -15.0:
        return -999.0

    if net_return_pct <= 0:
        return -999.0

    # Base composite score
    score = profit_factor * np.log1p(net_return_pct)

    # Penalty modifiers
    if profit_factor < 1.50:
        score *= 0.50
    if win_rate_pct < 40.0:
        score *= 0.50
    if sharpe < 0.80:
        score *= 0.75

    # Drawdown proximity penalty (nudges Optuna away from the -15% wall)
    dd_headroom = -15.0 - max_dd_pct   # Positive = headroom remaining
    if dd_headroom < 3.0:
        score *= max(0.70, 0.70 + 0.10 * dd_headroom)

    logger.info(
        f"Trial {trial.number:04d} | "
        f"PF={profit_factor:.2f} Net={net_return_pct:.1f}% "
        f"DD={max_dd_pct:.1f}% WR={win_rate_pct:.1f}% "
        f"Trades={len(trades)} Score={score:.4f}"
    )

    return round(score, 6)


# ═══════════════════════════════════════════════════════════════════════════════
# Study runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_optimization(
    df_dict_4h: dict[str, pd.DataFrame],
    df_dict_1d: dict[str, pd.DataFrame],
    n_trials:   int = OPTUNA_CONFIG["n_trials"],
) -> optuna.Study:
    """
    Launch the V5 Optuna study.

    Args:
        df_dict_4h: Full 4h dataset {asset: DataFrame}.
        df_dict_1d: Full 1d dataset {asset: DataFrame}.
        n_trials:   Number of optimization trials (default 300).

    Returns:
        Completed optuna.Study object.
    """
    os.makedirs("config", exist_ok=True)

    study = optuna.create_study(
        study_name     = OPTUNA_CONFIG["study_name"],
        storage        = OPTUNA_CONFIG["storage"],
        direction      = OPTUNA_CONFIG["direction"],
        sampler        = TPESampler(seed=42),
        pruner         = HyperbandPruner(),
        load_if_exists = True,
    )

    print(f"\n{'=' * 65}")
    print(f"  V5 Optuna HPO")
    print(f"  Study    : {OPTUNA_CONFIG['study_name']}")
    print(f"  Trials   : {n_trials}")
    print(f"  Objective: PF × log1p(Net%)  |  MaxDD < 15% HARD CONSTRAINT")
    print(f"  Existing : {len(study.trials)} completed trials")
    print(f"{'=' * 65}\n")

    study.optimize(
        lambda t: optuna_objective(t, df_dict_4h, df_dict_1d),
        n_trials          = n_trials,
        n_jobs            = OPTUNA_CONFIG["n_jobs"],
        show_progress_bar = True,
        catch             = (Exception,),
    )

    return study


# ═══════════════════════════════════════════════════════════════════════════════
# Best params extraction
# ═══════════════════════════════════════════════════════════════════════════════

def extract_best_params(study: optuna.Study) -> dict:
    """
    Extract best trial params, validate against secondary filters, and save.

    Returns:
        dict of best hyperparameters with meta section.

    Raises:
        RuntimeError: No trial passes all secondary filters.
    """
    sf = OPTUNA_CONFIG["secondary_filter"]

    qualified = [
        t for t in study.trials
        if (
            t.state == optuna.trial.TrialState.COMPLETE
            and t.value is not None
            and t.value > 0
        )
    ]

    if not qualified:
        raise RuntimeError(
            "No trials passed secondary filters. "
            "Increase n_trials or check data quality."
        )

    best   = max(qualified, key=lambda t: t.value)
    params = best.params.copy()
    params["max_holding_candles"] = V5_TIME_EXIT_BARS  # Enforce lock

    print(f"\n{'=' * 65}")
    print(f"  V5 HPO COMPLETE — Best Trial #{best.number}")
    print(f"  Score (PF × log1p(Net%)) : {best.value:.4f}")
    print(f"\n  Core Parameters:")
    print(f"    confidence_thresh  : {params['confidence_thresh']:.4f}")
    print(f"    sl_atr_mult        : {params['sl_atr_mult']:.4f}")
    print(f"    rr_ratio           : {params['rr_ratio']:.3f}")
    print(f"    max_holding_candles: {params['max_holding_candles']}  (LOCKED)")
    print(f"\n  Feature Parameters:")
    print(f"    mss_lookback       : {params.get('mss_lookback')}")
    print(f"    fvg_min_atr_ratio  : {params.get('fvg_min_atr_ratio', 'N/A'):.3f}")
    print(f"    xgb_max_depth      : {params.get('xgb_max_depth')}")
    print(f"    xgb_n_estimators   : {params.get('xgb_n_estimators')}")
    print(f"{'=' * 65}")

    # Validate ranges
    assert 0.750 <= params["confidence_thresh"] <= 0.880, "confidence_thresh OOB"
    assert 0.700 <= params["sl_atr_mult"]       <= 1.200, "sl_atr_mult OOB"
    assert 3.500 <= params["rr_ratio"]          <= 5.000, "rr_ratio OOB"
    assert params["max_holding_candles"] == 31,           "Time exit lock violated"

    params["_meta"] = {
        "version":     "5.0.0",
        "study_name":  OPTUNA_CONFIG["study_name"],
        "best_trial":  best.number,
        "best_score":  round(best.value, 6),
        "objective":   "profit_factor × log1p(net_return_pct) | MaxDD < -15%",
        "time_exit":   "31 bars LOCKED",
        "n_trials_completed": len(qualified),
    }

    out_path = "config/best_params_v5.json"
    os.makedirs("config", exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(params, fh, indent=2)

    print(f"\n  Best params saved → {out_path}")
    return params
