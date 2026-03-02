"""
run_optimization.py — Varanus v5 HPO Entry Point (Step 7).

Usage:
    cd /home/yagokhan
    PYTHONPATH=/home/yagokhan python varanus/run_optimization.py [--trials N] [--resume]

Steps performed:
    1. Load 4h + 1d data for all 20 TIER2_UNIVERSE assets
    2. Run Optuna HPO (300 trials, TPE + Hyperband) on Fold-1 train→val
    3. Save best params to config/best_params_v5.json
    4. Run full 8-fold Walk-Forward Validation with best params
    5. Print pass/fail summary; write WFV report to results/

Objective: profit_factor × log1p(net_return_pct)  [maximise]
Constraint: max_drawdown < -15% → score = -999.0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd

# ── Ensure project root on sys.path ───────────────────────────────────────────
ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from varanus.universe  import TIER2_UNIVERSE
from varanus.optimizer import OPTUNA_CONFIG, run_optimization, extract_best_params
from varanus.walk_forward import run_walk_forward, WFV_CONFIG

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("varanus.run_optimization")

# ── Data paths ─────────────────────────────────────────────────────────────────
CACHE = "/home/yagokhan/chameleon/claude_code_project/data/cache"


def load_data(symbol: str, timeframe: str) -> pd.DataFrame:
    """Load a parquet candle file and return a normalised OHLCV DataFrame."""
    # ASTR is stored as ASTER in the cache
    file_symbol = "ASTER" if symbol == "ASTR" else symbol

    if timeframe == "1d":
        # Try 1h file first (resample to 1D), fall back to the 4h-named parquet
        try:
            df = pd.read_parquet(f"{CACHE}/{file_symbol}_USDT_1h.parquet")
        except FileNotFoundError:
            df = pd.read_parquet(f"{CACHE}/{file_symbol}_USDT.parquet")
    elif timeframe == "4h":
        df = pd.read_parquet(f"{CACHE}/{file_symbol}_USDT.parquet")
    else:
        df = pd.read_parquet(f"{CACHE}/{file_symbol}_USDT_{timeframe}.parquet")

    df.columns = [c.lower() for c in df.columns]
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index("timestamp")
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()

    if timeframe == "1d":
        agg: dict = {}
        for col, func in [("open", "first"), ("high", "max"), ("low", "min"),
                           ("close", "last"), ("volume", "sum")]:
            if col in df.columns:
                agg[col] = func
        df = df.resample("1D").agg(agg).dropna()

    return df


def load_universe() -> tuple[dict, dict]:
    """Load 4h and 1d data for all TIER2_UNIVERSE assets. Returns (data_4h, data_1d)."""
    data_4h: dict[str, pd.DataFrame] = {}
    data_1d: dict[str, pd.DataFrame] = {}

    for asset in TIER2_UNIVERSE:
        try:
            df_4h = load_data(asset, "4h")
            df_1d = load_data(asset, "1d")
            # Align 1d to 4h start (with 100-bar lookback buffer)
            df_1d = df_1d[df_1d.index >= df_4h.index[0] - pd.Timedelta(days=100)]

            data_4h[asset] = df_4h
            data_1d[asset] = df_1d
            logger.info("Loaded %s: %d × 4h bars | %d × 1d bars",
                        asset, len(df_4h), len(df_1d))
        except Exception as exc:
            logger.warning("Skipping %s — %s", asset, exc)

    logger.info("Universe loaded: %d / %d assets", len(data_4h), len(TIER2_UNIVERSE))
    return data_4h, data_1d


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Varanus v5 HPO + WFV runner")
    p.add_argument("--trials",  type=int,  default=OPTUNA_CONFIG["n_trials"],
                   help="Number of Optuna trials (default: 300)")
    p.add_argument("--resume",  action="store_true",
                   help="Resume existing Optuna study from DB (default: create fresh)")
    p.add_argument("--skip-wfv", action="store_true",
                   help="Run HPO only; skip full 8-fold WFV after optimisation")
    p.add_argument("--dry-run", action="store_true",
                   help="Load data and check imports only; do not run Optuna")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = _parse_args()

    print("\n" + "═" * 70)
    print("  VARANUS v5.0 — Hyperparameter Optimisation + Walk-Forward Validation")
    print("═" * 70)

    # ── 1. Load data ─────────────────────────────────────────────────────────
    logger.info("Loading universe data ...")
    data_4h, data_1d = load_universe()

    if not data_4h:
        logger.error("No assets loaded — check CACHE path: %s", CACHE)
        sys.exit(1)

    if args.dry_run:
        logger.info("--dry-run: data loaded successfully. Exiting without HPO.")
        return

    # ── 2. HPO ───────────────────────────────────────────────────────────────
    logger.info("Starting Optuna HPO — %d trials ...", args.trials)

    cfg_override = {"n_trials": args.trials}
    study = run_optimization(
        data_4h     = data_4h,
        data_1d     = data_1d,
        cfg         = cfg_override,
        resume      = args.resume,
    )

    # ── 3. Extract & save best params ────────────────────────────────────────
    out_dir = Path(__file__).parent / "config"
    out_dir.mkdir(exist_ok=True)
    params_path = out_dir / "best_params_v5.json"

    best_params = extract_best_params(study, out_path=str(params_path))

    print("\n" + "─" * 50)
    print("  Best Trial Results")
    print("─" * 50)
    print(f"  Trial #:  {study.best_trial.number}")
    print(f"  Score:    {study.best_value:.4f}")
    print(f"  Params:\n{json.dumps(best_params, indent=4, default=str)}")
    print(f"\n  Saved to: {params_path}")

    if args.skip_wfv:
        logger.info("--skip-wfv: skipping 8-fold walk-forward validation.")
        return

    # ── 4. Full 8-fold Walk-Forward Validation ────────────────────────────────
    logger.info("Running full 8-fold Walk-Forward Validation ...")

    wfv_results = run_walk_forward(
        data_4h     = data_4h,
        data_1d     = data_1d,
        params      = best_params,
    )

    # ── 5. Summary ───────────────────────────────────────────────────────────
    folds_df    = wfv_results["fold_results"]
    summary     = wfv_results["summary"]

    print("\n" + "═" * 70)
    print("  Walk-Forward Validation Summary")
    print("═" * 70)
    print(f"  Folds run:    {summary.get('n_folds', 0)}")
    print(f"  Folds PASS:   {summary.get('n_pass', 0)} / {summary.get('n_folds', 0)}")
    print(f"  Consistency:  {summary.get('consistency_pct', 0):.1f}%")
    print(f"  Calmar (mean): {summary.get('calmar_mean', 0):.3f}")
    print(f"  Win Rate:     {summary.get('win_rate_mean', 0):.1f}%")
    print(f"  Max DD (worst): {summary.get('max_dd_worst', 0):.2f}%")
    print(f"  WFV PASS:     {'✅ YES' if summary.get('wfv_pass') else '❌ NO'}")
    print("═" * 70)

    # Save fold results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    folds_path = results_dir / "step6_wfv_v5.csv"
    folds_df.to_csv(folds_path, index=False)
    logger.info("WFV fold details saved to %s", folds_path)

    if not summary.get("wfv_pass"):
        logger.warning("WFV FAILED — re-run HPO with more trials or review params.")
        sys.exit(1)

    logger.info("All steps complete. v5 params ready for paper trading.")


if __name__ == "__main__":
    main()
