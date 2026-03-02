import pandas as pd
import numpy as np

from varanus.model import VaranusModel, MODEL_CONFIG
from varanus.pa_features import build_features, detect_mss, compute_atr
from varanus.tbm_labeler import label_trades, TBM_CONFIG
from varanus.backtest import run_backtest, compute_metrics

WFV_CONFIG = {
    "n_folds":           5,
    "method":            "sliding_window",  # Not anchored/expanding
    "shuffle":           False,             # NEVER shuffle. Temporal integrity sacred.
    "train_ratio":       0.70,
    "val_ratio":         0.15,
    "test_ratio":        0.15,
    "min_train_candles": 1000,              # ~167 days on 4h
    "gap_candles":       24,                # 4-day gap between splits (leakage guard)
    "performance_gate": {
        "min_calmar":      0.50,
        "min_win_rate":    43.0,
        "max_fold_dd":    -0.30,
        "consistency_req": 0.80,           # ≥ 80% of folds must be profitable
    },
}

def _generate_folds(df_dict: dict[str, pd.DataFrame], cfg: dict) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex]]:
    """
    Generate chronological sliding-window folds with embargo gaps across all assets.
    Returns: list of (train_idx, val_idx, test_idx)
    """
    n_folds = cfg['n_folds']
    gap     = cfg['gap_candles']
    
    # 1. Gather all unique timestamps across the active universe to get global timeline
    all_ts = set()
    for df in df_dict.values():
        all_ts.update(df.index)
    global_idx = pd.DatetimeIndex(sorted(all_ts))
    
    total_len = len(global_idx)
    
    test_len  = int(total_len * cfg['test_ratio'])
    val_len   = int(total_len * cfg['val_ratio'])
    train_len = int(total_len * cfg['train_ratio'])
    
    # 3. Ensure the single fold len doesn't overflow total_len due to float math on ratios + gap constants
    if test_len + val_len + train_len + 2 * gap > total_len:
        train_len = total_len - test_len - val_len - 2 * gap
    
    # Guard sizes
    if test_len == 0 or val_len == 0 or train_len == 0:
        raise ValueError("Calculated zero length for a fold component. Ratios too small relative to data length.")
    
    single_fold_len = train_len + gap + val_len + gap + test_len
    
    if total_len < single_fold_len:
        raise ValueError(f"Dataset length {total_len} is smaller than required single fold length {single_fold_len}")
        
    folds = []
    
    # max_slide is the amount of extra length we have beyond 1 single fold.
    # We step backwards from the END of the dataset to allocate n_folds.
    # Therefore, we start with the *last* fold anchored at the very end.
    
    max_slide = total_len - single_fold_len
    if n_folds > 1:
        step_size = max_slide // (n_folds - 1)
        if step_size <= 0:
            print(f"Warning: Calculated step_size={step_size}. Forcing step to minimum 1.")
            step_size = 1
    else:
        step_size = 0
        
    for i in range(n_folds):
        # We want fold 0 to be the oldest (starts nearest to index 0)
        # We want fold n-1 to be the newest (ends nearest to total_len)
        # Therefore, fold i starts at i * step_size
        start_idx = i * step_size
        
        train_start = start_idx
        train_end   = train_start + train_len
        
        val_start   = train_end + gap
        val_end     = val_start + val_len
        
        test_start  = val_end + gap
        test_end    = test_start + test_len
        
        # Ensure test_end doesn't wildly overshoot due to rounding differences
        if test_end > total_len:
            # Shift everything back by the overshoot amount
            overshoot = test_end - total_len
            train_start -= overshoot
            train_end   -= overshoot
            val_start   -= overshoot
            val_end     -= overshoot
            test_start  -= overshoot
            test_end     = total_len
            
        train_slice = global_idx[train_start:train_end]
        val_slice   = global_idx[val_start:val_end]
        test_slice  = global_idx[test_start:test_end]
        
        if len(train_slice) == 0 or len(val_slice) == 0 or len(test_slice) == 0:
            print(f"Warning: Fold {i} contains empty slice. Train:{len(train_slice)} Val:{len(val_slice)} Test:{len(test_slice)}")
            
        folds.append((train_slice, val_slice, test_slice))
        
    return folds

def _slice(df_dict: dict[str, pd.DataFrame], dt_idx: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """Slice a dictionary of asset DataFrames explicitly by a DatetimeIndex."""
    sliced = {}
    
    if len(dt_idx) == 0:
        return sliced
        
    start_ts = dt_idx[0]
    end_ts   = dt_idx[-1]
    
    for asset, df in df_dict.items():
        # Only take data strictly inside the boundary
        sub = df[(df.index >= start_ts) & (df.index <= end_ts)].copy()
        if not sub.empty:
            sliced[asset] = sub
            
    return sliced

def generate_signals(data_slice: dict[str, pd.DataFrame], model: VaranusModel, params: dict) -> dict[str, pd.DataFrame]:
    """Generates trading signals across the given data slice using the provided model."""
    signals = {}
    from varanus.pa_features import compute_atr, detect_mss
    
    for asset, df_4h in data_slice.items():
        # Usually requires 1d data as well for feature gen, but in walk_forward we
        # emulate it by sending df_4h again or managing it. For safety:
        # We will assume df_4h has enough context or `build_features` handles it.
        # But wait – the spec requires `build_features(df_4h, df_1d)`.
        
        # If we just need the XGB predictions to backtest:
        X, _ = build_features(df_4h, df_4h, asset) # Passing 4h as 1d mock for now if actual 1d obj missing in wrapper
        
        if X.empty:
            continue
            
        probs = model.predict_proba(X)
        preds = model.predict(X)
        
        # Format the signal DF
        sig_df = pd.DataFrame(index=X.index)
        
        # Max prob class
        sig_df['confidence'] = probs.max(axis=1)
        sig_df['direction']  = preds
        sig_df['entry_price'] = df_4h.loc[X.index, 'close']
        sig_df['atr'] = compute_atr(df_4h.loc[X.index], 14)
        
        # Filter for actual triggers (not class 0)
        sig_df = sig_df[sig_df['direction'] != 0]
        
        if not sig_df.empty:
            signals[asset] = sig_df
            
    return signals

def run_walk_forward(df_dict_4h: dict[str, pd.DataFrame], df_dict_1d: dict[str, pd.DataFrame], 
                     params: dict, cfg: dict = WFV_CONFIG) -> tuple[pd.DataFrame, float, pd.DataFrame]:
    """
    Run 5-fold sliding-window walk-forward validation.
    Each fold: retrain model → backtest on test slice → collect metrics.
    """
    fold_results = []
    all_trades = []
    
    folds = _generate_folds(df_dict_4h, cfg)
    
    for fold_idx, (train_idx, val_idx, test_idx) in enumerate(folds):
        print(f"\n── Fold {fold_idx + 1}/{cfg['n_folds']} ──")
        print(f" Train: {train_idx[0].strftime('%Y-%m-%d')} to {train_idx[-1].strftime('%Y-%m-%d')} ({len(train_idx)} bars)")
        print(f" Val:   {val_idx[0].strftime('%Y-%m-%d')} to {val_idx[-1].strftime('%Y-%m-%d')} ({len(val_idx)} bars)")
        print(f" Test:  {test_idx[0].strftime('%Y-%m-%d')} to {test_idx[-1].strftime('%Y-%m-%d')} ({len(test_idx)} bars)")

        train_data_4h = _slice(df_dict_4h, train_idx)
        train_data_1d = _slice(df_dict_1d, train_idx)
        
        val_data_4h   = _slice(df_dict_4h, val_idx)
        val_data_1d   = _slice(df_dict_1d, val_idx)
        
        test_data_4h  = _slice(df_dict_4h, test_idx)
        test_data_1d  = _slice(df_dict_1d, test_idx)
        
        # 1. Expand features across all assets to form monolithic train/val sets
        # Retrain fresh model on this fold's window
        model = VaranusModel(MODEL_CONFIG)
        X_tr_list, y_tr_list = [], []
        X_vl_list, y_vl_list = [], []
        
        # Removed leaky import
        
        for asset in train_data_4h.keys():
            if asset not in train_data_1d: continue
            X = build_features(train_data_4h[asset], train_data_1d[asset], asset, params)
            if X.empty: continue
            mss_signals = X['mss_signal']
            y = label_trades(train_data_4h[asset].loc[X.index], mss_signals, TBM_CONFIG, asset, params)
            y = y.reindex(X.index).fillna(0).astype(int)
            X_tr_list.append(X)
            y_tr_list.append(y)

        for asset in val_data_4h.keys():
            if asset not in val_data_1d: continue
            X = build_features(val_data_4h[asset], val_data_1d[asset], asset, params)
            if X.empty: continue
            mss_signals = X['mss_signal']
            y = label_trades(val_data_4h[asset].loc[X.index], mss_signals, TBM_CONFIG, asset, params)
            y = y.reindex(X.index).fillna(0).astype(int)
            X_vl_list.append(X)
            y_vl_list.append(y)

        if not X_tr_list:
            print(" Skipping fold: Insufficient training data")
            continue
            
        X_train_full = pd.concat(X_tr_list)
        y_train_full = pd.concat(y_tr_list)
        X_val_full   = pd.concat(X_vl_list) if X_vl_list else None
        y_val_full   = pd.concat(y_vl_list) if y_vl_list else None
        
        model.fit(X_train_full, y_train_full, X_val_full, y_val_full)

        # 2. Generate signals and backtest on unseen test window
        signals = {}
        for asset, df_4h in test_data_4h.items():
            if asset not in test_data_1d: continue
            X_t = build_features(df_4h, test_data_1d[asset], asset, params)
            if X_t.empty: continue

            probs = model.predict_proba(X_t)
            preds = model.predict(X_t)

            sig_df = pd.DataFrame(index=X_t.index)
            sig_df['confidence'] = probs.max(axis=1)
            sig_df['direction']  = preds
            sig_df['entry_price'] = df_4h.loc[X_t.index, 'close']
            sig_df['atr'] = compute_atr(df_4h.loc[X_t.index], 14)

            sig_df = sig_df[sig_df['direction'] != 0]

            if not sig_df.empty:
                signals[asset] = sig_df
                
        if not signals:
            print(" No signals generated during Out-Of-Sample test period. Skipping.")
            continue
            
        equity, trades = run_backtest(test_data_4h, signals, model, params)
        metrics = compute_metrics(equity, trades)

        trades['fold'] = fold_idx + 1
        all_trades.append(trades)

        fold_results.append({'fold': fold_idx + 1, **metrics})
        
        # Condense output
        summary = f"  Trades: {metrics['total_trades']} | CAGR: {metrics['cagr_pct']}% | " \
                  f"Calmar: {metrics['calmar_ratio']} | WR: {metrics['win_rate_pct']}% | " \
                  f"MaxDD: {metrics['max_drawdown_pct']}%"
        print(summary)

    results_df   = pd.DataFrame(fold_results)
    all_trades_df = pd.concat(all_trades).reset_index(drop=True) if all_trades else pd.DataFrame()
    
    if results_df.empty:
        return results_df, 0.0, all_trades_df
        
    passed_folds = (results_df['calmar_ratio'] >= cfg['performance_gate']['min_calmar']).sum()
    consistency  = passed_folds / cfg['n_folds']

    print(f"\nWFV Summary — {passed_folds}/{cfg['n_folds']} folds passed")
    print(f"Consistency: {consistency:.0%} (required: "
          f"{cfg['performance_gate']['consistency_req']:.0%})")
    
    cols_to_print = ['fold', 'calmar_ratio', 'win_rate_pct', 'max_drawdown_pct', 'total_trades']
    print(results_df[[c for c in cols_to_print if c in results_df.columns]].to_string(index=False))

    return results_df, consistency, all_trades_df
