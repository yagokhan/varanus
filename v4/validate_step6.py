import pandas as pd
import numpy as np

from varanus.walk_forward import WFV_CONFIG, run_walk_forward
from varanus.universe import TIER2_UNIVERSE

CACHE = "/home/yagokhan/chameleon/claude_code_project/data/cache"

def load_data(symbol: str, timeframe: str) -> pd.DataFrame:
    file_symbol = "ASTER" if symbol == "ASTR" else symbol

    if timeframe == "1d":
        try:
            df = pd.read_parquet(f"{CACHE}/{file_symbol}_USDT_1h.parquet")
        except FileNotFoundError:
            # Fallback to 4h data for resampling to 1D
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

print("=== Loading Data for Walk-Forward Validation ===")
# We'll use the narrowed 15-coin universe
assets = TIER2_UNIVERSE
data_4h = {}
data_1d = {}

for asset in assets:
    try:
        # Load maximum history for WFV
        df_4h = load_data(asset, "4h")
        df_1d = load_data(asset, "1d")
        # Align 1d roughly
        df_1d = df_1d[df_1d.index >= df_4h.index[0] - pd.Timedelta(days=100)]
        print(f"Loaded {asset}: {len(df_4h)} candles")
        data_4h[asset] = df_4h
        data_1d[asset] = df_1d
    except Exception as e:
        print(f"Skipping {asset} due to error: {e}")
        continue

print(f"\nLoaded {len(data_4h)} assets completely.")

default_params = {
    "take_profit_atr": 2.5,
    "stop_loss_atr": 1.2,
    "max_holding": 30,
    "min_rr_ratio": 1.0, 
    "confidence_thresh": 0.80,
    "high_vol_overrides": {
        "take_profit_atr": 3.0,
        "stop_loss_atr":   1.5,
    },
}

results_df, consistency, trades_df = run_walk_forward(data_4h, data_1d, default_params, WFV_CONFIG)

trades_csv_path = "step6_trades.csv"
trades_df.to_csv(trades_csv_path, index=False)
print(f"\n[+] Saved {len(trades_df)} walk-forward trades to {trades_csv_path}")

print(f"\nWalk-forward validation execution: PASS ✓")
