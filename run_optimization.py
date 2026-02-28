import optuna
import json
import os
import sys
import pandas as pd

from varanus.universe import TIER2_UNIVERSE
from varanus.optimizer import OPTUNA_CONFIG, optuna_objective
CACHE = "/home/yagokhan/chameleon/claude_code_project/data/cache"

def load_data(symbol: str, timeframe: str) -> pd.DataFrame:
    file_symbol = "ASTER" if symbol == "ASTR" else symbol
    if timeframe == "1d":
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
        agg_dict = {}
        if 'open' in df.columns: agg_dict['open'] = 'first'
        if 'high' in df.columns: agg_dict['high'] = 'max'
        if 'low' in df.columns: agg_dict['low'] = 'min'
        if 'close' in df.columns: agg_dict['close'] = 'last'
        if 'volume' in df.columns: agg_dict['volume'] = 'sum'
        df = df.resample('1D').agg(agg_dict).dropna()
        
    return df

def run():
    print("=== Varanus Tier 2 Strategy Optimization ===")
    
    # 1. Load Parquet Data
    print("\n[+] Loading Universe Data...")
    valid_assets = []
    data_4h = {}
    data_1d = {}

    for asset in TIER2_UNIVERSE:
        try:
            df_4h = load_data(asset, "4h")
            df_1d = load_data(asset, "1d")
            df_1d = df_1d[df_1d.index >= df_4h.index[0] - pd.Timedelta(days=100)]
            
            data_4h[asset] = df_4h
            data_1d[asset] = df_1d
            valid_assets.append(asset)
            print(f"Loaded {asset}: {len(df_4h)} candles")
        except Exception as e:
            print(f"Skipping {asset} due to error: {e}")

    print(f"\n[+] Loaded {len(valid_assets)} assets completely.")

    # 2. Configure Optuna
    target_obj = lambda trial: optuna_objective(trial, data_4h, data_1d)
    
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner  = optuna.pruners.HyperbandPruner()
    
    study = optuna.create_study(
        study_name="Varanus_Tier2_Calmar_Search",
        direction=OPTUNA_CONFIG["direction"],
        sampler=sampler,
        pruner=pruner
    )

    # 3. Execute Grid Search
    print(f"\n[+] Commencing {OPTUNA_CONFIG['n_trials']}-trial Optimization...")
    print("    Objective: Maximize Out-of-Sample Calmar Ratio across 5-Folds\n")
    
    try:
        study.optimize(target_obj, n_trials=OPTUNA_CONFIG["n_trials"], n_jobs=1)
    except KeyboardInterrupt:
        print("\n[!] Optimization interrupted by user. Saving best known params...")

    # 4. Report Best Parameters
    print("\n=== Optimization Complete ===")
    print(f"Best Trial: {study.best_trial.number}")
    print(f"Best Calmar: {study.best_value:.3f}")
    
    best_params = study.best_params
    print(f"\nOptimal Paramer Configuration:\n{json.dumps(best_params, indent=4)}")

    # 5. Output JSON Artifact
    out_dir = "/home/yagokhan/varanus/config"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "best_params.json")
    
    with open(out_file, "w") as f:
        json.dump(best_params, f, indent=4)
        
    print(f"\n[+] Wrote best configuration to {out_file}")

if __name__ == "__main__":
    run()
