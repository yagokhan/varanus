import pandas as pd
import numpy as np

from varanus.model import VaranusModel, MODEL_CONFIG, build_features
from varanus.pa_features import detect_mss
from varanus.tbm_labeler import TBM_CONFIG
from varanus.backtest import run_backtest, compute_metrics, passes_backtest_gate, BACKTEST_CONFIG

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


print("=== Loading Data & Training Mock Model ===")
from varanus.universe import TIER2_UNIVERSE

# Run on the full Tier 2 Universe
assets = TIER2_UNIVERSE
data = {}
signals = {}

model = VaranusModel(MODEL_CONFIG)

for asset in assets:
    try:
        df_4h = load_data(asset, "4h").tail(1000)
        df_1d = load_data(asset, "1d")
        df_1d = df_1d[df_1d.index >= df_4h.index[0] - pd.Timedelta(days=100)]
        print(f"Loaded {asset} successfully.")
    except Exception as e:
        print(f"Skipping {asset} due to missing data: {e}")
        continue
    
    data[asset] = df_4h
    
    # We need a trained model, so we train on the first half and predict on second half 
    # to generate signals for the backtester
    features = build_features(df_4h, df_1d, asset)
    
    # Dummy target for quick fit
    np.random.seed(42)
    y_dummy = pd.Series(np.random.choice([-1, 0, 1], size=len(features)), index=features.index)
    model.fit(features.iloc[:500], y_dummy.iloc[:500])
    
    # Generate mock signals based on mss_signal mainly for testing the backtester loop
    # We structure the signal df as the backtester expects
    sig_df = pd.DataFrame(index=df_4h.index)
    sig_df['confidence'] = 0.85 # Artificial confidence
    sig_df['direction'] = df_4h['close'].diff().apply(lambda x: 1 if x > 0 else -1) # Randomish direction
    sig_df['entry_price'] = df_4h['close']
    
    # Needed for calculate_barriers inside run_backtest
    from varanus.pa_features import compute_atr
    sig_df['atr'] = compute_atr(df_4h, 14)
    
    # Only keep 10% of signals to represent actual trade entries (prevent maxing out instantly)
    mask = np.random.rand(len(sig_df)) < 0.10
    signals[asset] = sig_df[mask]

print(f"Loaded signals for {len(signals)} assets.")

print("\n=== Running Backtest ===")
default_params = {
    "take_profit_atr": 2.5,
    "stop_loss_atr": 1.2,
    "max_holding": 30,
    "min_rr_ratio": 1.0, # low for test
    "confidence_thresh": 0.80,
    "high_vol_overrides": {
        "take_profit_atr": 3.0,
        "stop_loss_atr":   1.5,
    },
}

equity, trades = run_backtest(data, signals, model, default_params, BACKTEST_CONFIG)

print(f"Total Trades: {len(trades)}")
if len(trades) > 0:
    print("\nFirst 3 trades:")
    print(trades.head(3)[['asset', 'direction', 'entry_price', 'exit_price', 'outcome', 'pnl_usd']])
    trades.to_csv("step5_trades.csv", index=False)
    print("\n[+] Saved trades to step5_trades.csv")

print("\n=== Performance Metrics ===")
metrics = compute_metrics(equity, trades)
for k, v in metrics.items():
    print(f"{k:20s}: {v}")

passed = passes_backtest_gate(metrics)
print(f"\nBacktest gate (informational only for mock): {'PASS ✓' if passed else 'FAIL ✗'}")

assert len(trades) > 5, "Too few trades — check signal emission pipeline"
assert isinstance(equity, pd.Series), "Equity must be a Series"
assert "pnl_usd" in trades.columns, "Trades missing pnl_usd column"
print("\nStep 5 Validation: PASS")
