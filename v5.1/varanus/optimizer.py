import optuna
import pandas as pd
import numpy as np
import json

from varanus.walk_forward import WFV_CONFIG_V51, _generate_folds_v51, _slice
from varanus.backtest import run_backtest, compute_metrics
from varanus.model import VaranusModel, MODEL_CONFIG
from varanus.pa_features import build_features, compute_atr
from varanus.tbm_labeler import label_trades, TBM_CONFIG

HUNTER_OPTUNA_CONFIG = {
    "n_trials":              300,
    "direction":             "maximize",
    "sampler":               "TPESampler",
    "pruner":                "HyperbandPruner",
    "min_trades_per_fold":   8,
    "min_total_trades":      30,
    "dd_penalty_threshold":  0.12,   # Folds breaching -12% DD get penalised
    "dd_penalty_multiplier": 0.40,
}

# v4 frozen params — not searched by Optuna, loaded from best_params.json
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


def _hunter_efficiency(net_profit_usd: float, max_dd_pct: float) -> float:
    """Hunter Efficiency = Net Profit / |Max Drawdown|. Returns 0 if no drawdown."""
    return net_profit_usd / abs(max_dd_pct) if max_dd_pct != 0 else 0.0


def optuna_objective_hunter(
    trial: optuna.Trial,
    df_dict_4h: dict[str, pd.DataFrame],
    df_dict_1d: dict[str, pd.DataFrame],
    cfg: dict = WFV_CONFIG_V51,
) -> float:
    """
    Hunter Efficiency Optuna Objective for Varanus v5.1.

    Searches 6 Hunter parameters across 8-fold rolling walk-forward.
    Reports per-fold intermediate scores so HyperbandPruner can prune early.
    Applies 0.40x penalty on any fold where MaxDD exceeds -12%.

    Search Space (6 Hunter Parameters)
    -----------------------------------
    1. confidence_thresh   — entry gate          [0.750 – 0.880]
    2. sl_atr_mult         — stop loss ATR mult  [0.700 – 1.200]
    3. tp_atr_mult         — take profit ATR mult [3.500 – 6.000]
    4. leverage_5x_trigger — 5x lev gate         [0.930 – 0.980]
    5. xgb_lr              — XGBoost learn rate  [0.005 – 0.080]
    6. xgb_max_depth       — XGBoost tree depth  [3 – 6]
    """
    params = {
        # 1. Entry Gate
        "confidence_thresh":   trial.suggest_float("confidence_thresh",   0.750, 0.880),
        # 2. Stop Loss
        "sl_atr_mult":         trial.suggest_float("sl_atr_mult",         0.700, 1.200),
        # 3. Take Profit
        "tp_atr_mult":         trial.suggest_float("tp_atr_mult",         3.500, 6.000),
        # 4. 5x Leverage Trigger
        "leverage_5x_trigger": trial.suggest_float("leverage_5x_trigger", 0.930, 0.980),
        # 5 & 6. XGBoost — tuned for 40% train windows
        "xgb_lr":              trial.suggest_float("xgb_lr",              0.005, 0.080, log=True),
        "xgb_max_depth":       trial.suggest_int("xgb_max_depth",         3,     6),
        # Frozen v4 params
        **V4_FROZEN_PARAMS,
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
                if asset not in train_1d: continue
                X = build_features(train_4h[asset], train_1d[asset], asset, params)
                if X.empty: continue
                y = label_trades(train_4h[asset].loc[X.index], X['mss_signal'],
                                 TBM_CONFIG, asset, params)
                y = y.reindex(X.index).fillna(0).astype(int)
                X_tr_list.append(X)
                y_tr_list.append(y)

            for asset in val_4h:
                if asset not in val_1d: continue
                X = build_features(val_4h[asset], val_1d[asset], asset, params)
                if X.empty: continue
                y = label_trades(val_4h[asset].loc[X.index], X['mss_signal'],
                                 TBM_CONFIG, asset, params)
                y = y.reindex(X.index).fillna(0).astype(int)
                X_vl_list.append(X)
                y_vl_list.append(y)

            if not X_tr_list:
                continue

            # Re-tune XGBoost params for this trial's depth/lr
            model_cfg = {**MODEL_CONFIG}
            model_cfg['xgb_params'] = {
                **MODEL_CONFIG['xgb_params'],
                'max_depth':     params['xgb_max_depth'],
                'learning_rate': params['xgb_lr'],
                'n_estimators':  params['xgb_n_estimators'],
                'subsample':     params['xgb_subsample'],
            }
            model = VaranusModel(model_cfg)
            model.fit(
                pd.concat(X_tr_list), pd.concat(y_tr_list),
                pd.concat(X_vl_list) if X_vl_list else None,
                pd.concat(y_vl_list) if y_vl_list else None,
            )

            signals = {}
            for asset, df_4h in test_4h.items():
                if asset not in test_1d: continue
                X_t = build_features(df_4h, test_1d[asset], asset, params)
                if X_t.empty: continue
                probs  = model.predict_proba(X_t)
                preds  = model.predict(X_t)
                sig_df = pd.DataFrame(index=X_t.index)
                sig_df['confidence']  = probs.max(axis=1)
                sig_df['direction']   = preds
                sig_df['entry_price'] = df_4h.loc[X_t.index, 'close']
                sig_df['atr']         = compute_atr(df_4h.loc[X_t.index], 14)
                sig_df = sig_df[sig_df['direction'] != 0]
                if not sig_df.empty:
                    signals[asset] = sig_df

            if not signals:
                continue

            equity, trades = run_backtest(test_4h, signals, model, params)
            metrics        = compute_metrics(equity, trades)

            fold_trades = metrics['total_trades']
            fold_dd     = metrics['max_drawdown_pct'] / 100.0
            net_profit  = trades['pnl_usd'].sum() if not trades.empty else 0.0

            if fold_trades < HUNTER_OPTUNA_CONFIG['min_trades_per_fold']:
                continue

            fold_he = _hunter_efficiency(net_profit, fold_dd)

            # Hard DD penalty
            if abs(fold_dd) > HUNTER_OPTUNA_CONFIG['dd_penalty_threshold']:
                fold_he *= HUNTER_OPTUNA_CONFIG['dd_penalty_multiplier']
                penalty_applied = True

            fold_scores.append(fold_he)
            total_trades += fold_trades

            # Report intermediate value for HyperbandPruner
            trial.report(float(np.mean(fold_scores)), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if total_trades < HUNTER_OPTUNA_CONFIG['min_total_trades']:
            print(f"  -> Penalty: only {total_trades} total trades.")
            return -999.0

        if not fold_scores:
            return -999.0

        mean_he = float(np.mean(fold_scores))
        flag    = " [DD PENALTY]" if penalty_applied else ""
        print(f"  -> Trial {trial.number} | Hunter Efficiency: {mean_he:.3f} | "
              f"Folds scored: {len(fold_scores)}/{len(folds)} | Trades: {total_trades}{flag}")
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
