"""
varanus/walk_forward.py — 8-Fold Sliding Window WFV (v5.0, Step 6).

V5 changes vs v4:
  - n_folds: 5 → 8
  - Ratios: 70/15/15 → 25/35/40  (less train, more val+test)
  - Method: sliding window (non-overlapping test windows, step = test_size)
  - Consistency gate: 80% → 75% (6/8 folds must pass)
  - Power Setup trade statistics added to fold output
  - generate_signals() rewritten: proper 4h + 1d separation

Window arithmetic:
  W = (N - 2×gap) / (1 + (n_folds-1) × test_ratio)
  step = W × test_ratio  ← non-overlapping OOS test windows
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy  as np
import pandas as pd

from varanus.model       import VaranusModel, MODEL_CONFIG, get_leverage
from varanus.pa_features import build_features, compute_atr
from varanus.tbm_labeler import label_trades, TBM_CONFIG
from varanus.backtest    import run_backtest, compute_metrics

logger = logging.getLogger(__name__)

# ── V5 WFV Config ─────────────────────────────────────────────────────────────

WFV_CONFIG: dict = {
    "n_folds":           8,
    "method":            "sliding_window",
    "shuffle":           False,            # NEVER shuffle — temporal integrity sacred
    "train_ratio":       0.25,
    "val_ratio":         0.35,
    "test_ratio":        0.40,
    "min_train_candles": 200,              # Reduced floor for smaller train window
    "gap_candles":       24,              # 4-day leakage embargo between splits
    "performance_gate": {
        "min_profit_factor": 1.50,         # V5 primary gate
        "min_win_rate":      43.0,
        "max_fold_dd":      -25.0,         # Per-fold DD cap (-25%)
        "min_calmar":        0.50,
        "consistency_req":   0.75,         # >= 6/8 folds must pass
    },
}


# ── Fold data structure ───────────────────────────────────────────────────────

@dataclass
class FoldIndices:
    """Integer index boundaries for a single rolling fold."""
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


# ═══════════════════════════════════════════════════════════════════════════════
# Core fold generator
# ═══════════════════════════════════════════════════════════════════════════════

def generate_rolling_folds(
    n_samples:   int,
    n_folds:     int   = WFV_CONFIG["n_folds"],
    train_ratio: float = WFV_CONFIG["train_ratio"],
    val_ratio:   float = WFV_CONFIG["val_ratio"],
    test_ratio:  float = WFV_CONFIG["test_ratio"],
    gap_candles: int   = WFV_CONFIG["gap_candles"],
    min_train:   int   = WFV_CONFIG["min_train_candles"],
) -> list[FoldIndices]:
    """
    Generate rolling-window fold index triplets for v5 Walk-Forward Validation.

    Design contract:
    ─────────────────────────────────────────────────────────────────────
    • Split ratios: 25% train / 35% val / 40% test  (must sum to 1.0)
    • Step: exactly test_size forward per fold → non-overlapping OOS tests
    • Gap: gap_candles removed at train→val and val→test boundaries
    • Shuffle: NEVER (temporal ordering is the contract)
    • All 8 test windows cover distinct, non-overlapping calendar segments
    ─────────────────────────────────────────────────────────────────────

    Window arithmetic:
        W + (n_folds-1) × step    = n_samples - 2 × gap_candles
        step                       = W × test_ratio
        W × (1 + (n_folds-1) × test_ratio) = n_samples - 2 × gap_candles
        W = (n_samples - 2 × gap_candles) / (1 + (n_folds-1) × test_ratio)

    Args:
        n_samples:   Total number of candles in the timeline.
        n_folds:     Number of rolling folds (default 8).
        train_ratio: Fraction of window for training (default 0.40).
        val_ratio:   Fraction of window for validation (default 0.30).
        test_ratio:  Fraction of window for OOS testing (default 0.30).
        gap_candles: Candles to skip at each split boundary (leakage guard).
        min_train:   Minimum candles in train window — raises ValueError if violated.

    Returns:
        List of FoldIndices (length == n_folds).

    Raises:
        ValueError: Dataset too small, ratios malformed, or gap too large.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-9:
        raise ValueError(
            f"Ratios must sum to 1.0. Got "
            f"{train_ratio + val_ratio + test_ratio:.6f}"
        )
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2. Got {n_folds}.")

    denominator = 1.0 + (n_folds - 1) * test_ratio
    W           = int((n_samples - 2 * gap_candles) / denominator)

    # Integer floor-rounding of train/val sizes makes test_size (and therefore
    # step) slightly > W × test_ratio.  Over (n_folds-1) steps the surplus
    # accumulates and can push the last fold's test_end past n_samples.
    # Reduce W by 1 until the last-fold test_end fits — never more than 2 iters.
    while True:
        train_size = int(W * train_ratio)
        val_size   = int(W * val_ratio)
        test_size  = W - train_size - val_size   # absorbs rounding residual
        step       = test_size
        last_test_end = (n_folds - 1) * step + W + 2 * gap_candles
        if last_test_end <= n_samples:
            break
        W -= 1

    if train_size < min_train:
        raise ValueError(
            f"train_size={train_size} < min_train={min_train}. "
            f"Provide more data or reduce n_folds / gap_candles."
        )
    if test_size < 30:
        raise ValueError(
            f"test_size={test_size} too small (<30). "
            f"Reduce n_folds or gap_candles."
        )

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
                f"Fold {fold_idx + 1}: test_end={test_end} > "
                f"n_samples={n_samples}. Dataset too short for "
                f"{n_folds} folds with gap_candles={gap_candles}."
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

    _validate_fold_integrity(folds, n_samples)
    return folds


def _validate_fold_integrity(folds: list[FoldIndices], n_samples: int) -> None:
    """Assert structural guarantees: no overflow, no inversion, no OOS overlap."""
    for f in folds:
        assert f.test.stop  <= n_samples, f"Fold {f.fold}: test overflows dataset"
        assert f.train.stop <= n_samples, f"Fold {f.fold}: train overflows dataset"
        assert f.train.start < f.train.stop, f"Fold {f.fold}: empty train slice"
        assert f.val.start   < f.val.stop,   f"Fold {f.fold}: empty val slice"
        assert f.test.start  < f.test.stop,  f"Fold {f.fold}: empty test slice"
        assert f.train.stop  < f.val.start,  f"Fold {f.fold}: train bleeds into val"
        assert f.val.stop    < f.test.start, f"Fold {f.fold}: val bleeds into test"

    for i in range(len(folds) - 1):
        assert folds[i].test.stop <= folds[i + 1].test.start, (
            f"OOS overlap: fold {folds[i].fold} test ends at {folds[i].test.stop}, "
            f"fold {folds[i+1].fold} test starts at {folds[i+1].test.start}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Data slicing helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _slice_by_ts(
    df_dict:  dict[str, pd.DataFrame],
    timeline: pd.DatetimeIndex,
    slc:      slice,
) -> dict[str, pd.DataFrame]:
    """
    Slice each asset DataFrame to the timestamp range defined by *slc* on *timeline*.
    """
    if slc.start >= slc.stop or slc.stop > len(timeline):
        return {}

    start_ts = timeline[slc.start]
    end_ts   = timeline[slc.stop - 1]

    sliced = {}
    for asset, df in df_dict.items():
        sub = df[(df.index >= start_ts) & (df.index <= end_ts)]
        if not sub.empty:
            sliced[asset] = sub.copy()

    return sliced


# ═══════════════════════════════════════════════════════════════════════════════
# Signal generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_signals(
    data_4h:  dict[str, pd.DataFrame],
    data_1d:  dict[str, pd.DataFrame],
    model:    VaranusModel,
    params:   dict,
) -> dict[str, pd.DataFrame]:
    """
    Generate trading signals for all assets using the fitted model.

    Args:
        data_4h:  {asset: 4h OHLCV DataFrame}
        data_1d:  {asset: 1d OHLCV DataFrame}
        model:    Fitted VaranusModel
        params:   Hyperparameter dict (confidence_thresh, etc.)

    Returns:
        {asset: signal DataFrame} with columns:
          confidence, direction, entry_price, atr
        Only rows with direction != 0 are included.
    """
    conf_thresh = params.get(
        "confidence_thresh",
        params.get("confidence_threshold", 0.750)
    )
    signals     = {}

    for asset, df_4h in data_4h.items():
        df_1d = data_1d.get(asset, df_4h)  # Fallback: 4h as 1d proxy
        try:
            X = build_features(df_4h, df_1d, asset, params)
        except Exception as exc:
            logger.warning(f"[{asset}] build_features failed: {exc}")
            continue

        if X.empty:
            continue

        probs = model.predict_proba(X)
        preds = model.predict(X, confidence_thresh=conf_thresh)

        sig_df                = pd.DataFrame(index=X.index)
        sig_df["confidence"]  = probs.max(axis=1)
        sig_df["direction"]   = preds
        sig_df["entry_price"] = df_4h.loc[X.index, "close"]
        sig_df["atr"]         = compute_atr(df_4h.loc[X.index], 14)

        sig_df = sig_df[sig_df["direction"] != 0]

        if not sig_df.empty:
            signals[asset] = sig_df

    return signals


# ═══════════════════════════════════════════════════════════════════════════════
# Walk-Forward Execution
# ═══════════════════════════════════════════════════════════════════════════════

def run_walk_forward(
    df_dict_4h: dict[str, pd.DataFrame],
    df_dict_1d: dict[str, pd.DataFrame],
    params:     dict,
    cfg:        dict = WFV_CONFIG,
) -> tuple[pd.DataFrame, float, pd.DataFrame]:
    """
    Execute 8-Fold Rolling Window Walk-Forward Validation.

    Each fold:
      1. Slice data into train / val / test by fold boundaries
      2. Retrain a fresh VaranusModel on the train slice
      3. Use val for early-stopping inside model.fit()
      4. Generate signals + run backtest on held-out test slice
      5. Collect metrics including Power Setup statistics

    Args:
        df_dict_4h: {asset: 4h OHLCV DataFrame} — full history
        df_dict_1d: {asset: 1d OHLCV DataFrame} — full history
        params:     Hyperparameter dict (from best_params_v5.json after HPO)
        cfg:        WFV_CONFIG (default: v5 8-fold 40/30/30)

    Returns:
        (fold_results_df, consistency_ratio, all_trades_df)
    """
    # Build global timeline from 4h data
    all_ts     = sorted(set().union(*[set(df.index) for df in df_dict_4h.values()]))
    timeline   = pd.DatetimeIndex(all_ts)
    N          = len(timeline)

    folds = generate_rolling_folds(
        n_samples   = N,
        n_folds     = cfg["n_folds"],
        train_ratio = cfg["train_ratio"],
        val_ratio   = cfg["val_ratio"],
        test_ratio  = cfg["test_ratio"],
        gap_candles = cfg["gap_candles"],
        min_train   = cfg["min_train_candles"],
    )

    gate         = cfg["performance_gate"]
    fold_results = []
    all_trades   = []

    for fold in folds:
        print(f"\n{'─' * 65}")
        print(f"  Fold {fold.fold}/{cfg['n_folds']}  |  {fold}")
        print(
            f"  Train: {timeline[fold.train.start].date()} → "
            f"{timeline[fold.train.stop - 1].date()}  ({fold.train_bars} bars)"
        )
        print(
            f"  Val:   {timeline[fold.val.start].date()} → "
            f"{timeline[fold.val.stop - 1].date()}    ({fold.val_bars} bars)"
        )
        print(
            f"  Test:  {timeline[fold.test.start].date()} → "
            f"{timeline[fold.test.stop - 1].date()}   ({fold.test_bars} bars)"
        )
        print(f"{'─' * 65}")

        # Slice data
        train_4h = _slice_by_ts(df_dict_4h, timeline, fold.train)
        val_4h   = _slice_by_ts(df_dict_4h, timeline, fold.val)
        test_4h  = _slice_by_ts(df_dict_4h, timeline, fold.test)
        train_1d = _slice_by_ts(df_dict_1d, timeline, fold.train)
        val_1d   = _slice_by_ts(df_dict_1d, timeline, fold.val)
        test_1d  = _slice_by_ts(df_dict_1d, timeline, fold.test)

        # Build train + val feature matrices
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
            logger.warning(f"Fold {fold.fold}: no training data. Skipping.")
            continue

        X_train = pd.concat(X_tr_list)
        y_train = pd.concat(y_tr_list)
        X_val   = pd.concat(X_vl_list) if X_vl_list else None
        y_val   = pd.concat(y_vl_list) if y_vl_list else None

        # Build Optuna XGB overrides into model config
        model_cfg = MODEL_CONFIG.copy()
        for opt_key in ["xgb_max_depth", "xgb_n_estimators", "xgb_lr", "xgb_subsample"]:
            if opt_key in params:
                model_cfg[opt_key] = params[opt_key]

        model = VaranusModel(model_cfg)
        model.fit(X_train, y_train, X_val, y_val)

        # Generate signals on test window
        signals = generate_signals(test_4h, test_1d, model, params)

        if not signals:
            logger.warning(
                f"Fold {fold.fold}: no signals in OOS test period. Skipping."
            )
            continue

        equity, trades = run_backtest(test_4h, signals, model, params)
        metrics        = compute_metrics(equity, trades)

        # Power Setup statistics
        ps_trades = (
            trades[trades["confidence"] >= 0.95]
            if "confidence" in trades.columns and len(trades)
            else pd.DataFrame()
        )
        metrics["power_setup_count"] = len(ps_trades)
        metrics["power_setup_pct"]   = (
            round(len(ps_trades) / len(trades) * 100, 1) if len(trades) else 0.0
        )

        # Fold pass/fail
        fold_pass = (
            metrics["profit_factor"]   >= gate["min_profit_factor"] and
            metrics["win_rate_pct"]    >= gate["min_win_rate"]      and
            metrics["max_drawdown_pct"] >= gate["max_fold_dd"]      and
            metrics["calmar_ratio"]    >= gate["min_calmar"]
        )
        metrics["fold_pass"] = fold_pass

        trades["fold"] = fold.fold
        all_trades.append(trades)
        fold_results.append({"fold": fold.fold, **metrics})

        status = "PASS ✓" if fold_pass else "FAIL ✗"
        print(
            f"  PF:{metrics['profit_factor']:.2f}  "
            f"WR:{metrics['win_rate_pct']:.1f}%  "
            f"DD:{metrics['max_drawdown_pct']:.1f}%  "
            f"Calmar:{metrics['calmar_ratio']:.2f}  "
            f"Trades:{metrics['total_trades']}  "
            f"PS:{metrics['power_setup_count']}  → {status}"
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    results_df     = pd.DataFrame(fold_results)
    all_trades_df  = (
        pd.concat(all_trades).reset_index(drop=True) if all_trades
        else pd.DataFrame()
    )

    if results_df.empty:
        logger.error("WFV produced no fold results.")
        return results_df, 0.0, all_trades_df

    passed_folds = int(results_df["fold_pass"].sum())
    consistency  = passed_folds / cfg["n_folds"]

    print(f"\n{'=' * 65}")
    print(f"  V5 WFV SUMMARY")
    print(f"  Folds passed : {passed_folds}/{cfg['n_folds']}")
    print(f"  Consistency  : {consistency:.0%}  "
          f"(required ≥ {gate['consistency_req']:.0%})")
    print(f"  Avg PF       : {results_df['profit_factor'].mean():.2f}")
    print(f"  Avg MaxDD    : {results_df['max_drawdown_pct'].mean():.1f}%")
    print(f"  Avg WR       : {results_df['win_rate_pct'].mean():.1f}%")
    print(f"  Avg Calmar   : {results_df['calmar_ratio'].mean():.2f}")
    print(f"  Total Trades : {int(results_df['total_trades'].sum())}")
    print(f"{'=' * 65}")

    display_cols = [
        c for c in [
            "fold", "profit_factor", "win_rate_pct", "max_drawdown_pct",
            "calmar_ratio", "total_trades", "power_setup_count", "fold_pass",
        ] if c in results_df.columns
    ]
    print(results_df[display_cols].to_string(index=False))

    return results_df, consistency, all_trades_df
