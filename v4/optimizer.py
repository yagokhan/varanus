import optuna
import pandas as pd
import json

from varanus.walk_forward import WFV_CONFIG, run_walk_forward
from varanus.backtest import run_backtest, compute_metrics

OPTUNA_CONFIG = {
    "n_trials":  200,
    "direction": "maximize",
    "sampler":   "TPESampler",
    "pruner":    "HyperbandPruner",
    "secondary_filter": {
        "min_trades":         30,
        "min_win_rate":       0.45,
        "max_drawdown_floor": -0.35,
    },
}

def optuna_objective(trial: optuna.Trial, df_dict_4h: dict[str, pd.DataFrame], df_dict_1d: dict[str, pd.DataFrame], cfg: dict = WFV_CONFIG) -> float:
    """
    Objective function for Optuna.
    Maximizes the average Walk-Forward Calmar Ratio across 5 test folds.
    """
    # Define the search space as specified by claude.md Step 7
    params = {
        # PA & FVG Features
        "mss_lookback":      trial.suggest_int("mss_lookback", 30, 50),
        "fvg_min_atr_ratio": trial.suggest_float("fvg_min_atr_ratio", 0.2, 0.5),
        "sweep_min_pct":     trial.suggest_float("sweep_min_pct", 0.002, 0.008),
        "fvg_max_age":       trial.suggest_int("fvg_max_age", 10, 25),
        
        # Triple Barrier Method
        "tp_atr_mult":       trial.suggest_float("tp_atr_mult", 2.0, 3.5),
        "sl_atr_mult":       trial.suggest_float("sl_atr_mult", 0.8, 1.8),
        "max_holding":       trial.suggest_int("max_holding", 15, 40),
        
        # Chameleon Confirmations
        "rvol_threshold":    trial.suggest_float("rvol_threshold", 1.2, 2.5),
        "rsi_oversold":      trial.suggest_int("rsi_oversold", 28, 42),
        "rsi_overbought":    trial.suggest_int("rsi_overbought", 58, 72),
        
        # Execution Gate
        "confidence_thresh": trial.suggest_float("confidence_thresh", 0.78, 0.92),
        
        # XGBoost Hyperparameters
        "xgb_max_depth":     trial.suggest_int("xgb_max_depth", 4, 8),
        "xgb_n_estimators":  trial.suggest_int("xgb_n_estimators", 200, 800),
        "xgb_lr":            trial.suggest_float("xgb_lr", 0.01, 0.1, log=True),
        "xgb_subsample":     trial.suggest_float("xgb_subsample", 0.6, 1.0),
    }

    print(f"\\n>>> Starting Trial {trial.number}")
    print(f"Parameters: {params}")

    try:
        # Run Walk-Forward Validation
        results_df, consistency, trades_df = run_walk_forward(df_dict_4h, df_dict_1d, params, cfg)
        
        if results_df.empty:
            print("  -> Trial Penalty: No folds generated results.")
            return -999.0
            
        total_trades = results_df['total_trades'].sum()
        if total_trades < OPTUNA_CONFIG['secondary_filter']['min_trades']:
            print(f"  -> Trial Penalty: Statistically insignificant ({total_trades} trades).")
            return -999.0

        # Maximize the average out-of-sample Calmar
        mean_calmar = results_df['calmar_ratio'].mean()
        mean_sharpe = results_df['sharpe_ratio'].mean()
        
        # Penalty: Sharpe < 1.0 (avoid erratic equity paths)
        if mean_sharpe < 1.0:
            mean_calmar *= 0.5
            
        print(f"  -> Trial {trial.number} Finished: Average OOS Calmar = {mean_calmar:.3f} | Trades = {total_trades}")
        return mean_calmar
        
    except Exception as e:
        print(f"  -> Trial {trial.number} Failed with exception: {e}")
        return -999.0
