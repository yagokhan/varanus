import pandas as pd
import numpy as np

from varanus.model import VaranusModel, MODEL_CONFIG, build_features, get_leverage
from varanus.pa_features import detect_mss
from varanus.universe import HIGH_VOL_SUBTIER

CACHE = "/home/yagokhan/chameleon/claude_code_project/data/cache"

def load_data(symbol: str, timeframe: str) -> pd.DataFrame:
    if timeframe == "1d":
        df = pd.read_parquet(f"{CACHE}/{symbol}_USDT_1h.parquet")
    elif timeframe == "4h":
        df = pd.read_parquet(f"{CACHE}/{symbol}_USDT.parquet")
    else:
        df = pd.read_parquet(f"{CACHE}/{symbol}_USDT_{timeframe}.parquet")
        
    df.columns = [c.lower() for c in df.columns]
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index("timestamp")
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    
    if timeframe == "1d":
        # Check if columns exist before resampling
        agg_dict = {}
        if 'open' in df.columns: agg_dict['open'] = 'first'
        if 'high' in df.columns: agg_dict['high'] = 'max'
        if 'low' in df.columns: agg_dict['low'] = 'min'
        if 'close' in df.columns: agg_dict['close'] = 'last'
        if 'volume' in df.columns: agg_dict['volume'] = 'sum'
        df = df.resample('1D').agg(agg_dict).dropna()
        
    return df


print("=== build_features ===")
df_4h = load_data("LINK", "4h")
df_1d = load_data("LINK", "1d")

# Limit to 500 rows for faster feature building in validation
df_4h_small = df_4h.tail(500).copy()
df_1d_small = df_1d[df_1d.index >= df_4h_small.index[0] - pd.Timedelta(days=100)].copy()

features = build_features(df_4h_small, df_1d_small, "LINK")
print(f"Features shape: {features.shape}")
print("Missing values per column:")
print(features.isna().sum()[features.isna().sum() > 0])

assert features.shape[1] == 16, f"Expected 16 features, got {features.shape[1]}"
assert "mss_signal" in features.columns
print("Feature extraction syntax OK.")

print("\n=== VaranusModel sanity check ===")
# Create dummy labels for testing fit()
np.random.seed(42)
y_dummy = pd.Series(np.random.choice([-1, 0, 1], size=len(features)), index=features.index)

# Train/test split
train_size = int(len(features) * 0.8)
X_train, X_test = features.iloc[:train_size], features.iloc[train_size:]
y_train, y_test = y_dummy.iloc[:train_size], y_dummy.iloc[train_size:]

model = VaranusModel(MODEL_CONFIG)
model.fit(X_train, y_train, X_val=X_test, y_val=y_test)
print("Model fit OK.")

probs = model.predict_proba(X_test)
print(f"Probabilities shape: {probs.shape}")
assert probs.shape[1] == 3, "Expected 3-class output"
assert np.allclose(probs.sum(axis=1), 1.0), "Probabilities don't sum to 1"

preds = model.predict(X_test)
print(f"Predictions shape: {preds.shape}")
assert set(np.unique(preds)).issubset({-1, 0, 1}), "Predictions must be in {-1, 0, 1}"

# Test specific Claude instruction: confidence threshold
confident_signals = (probs.max(axis=1) >= MODEL_CONFIG['confidence_threshold']).sum()
print(f"Signals above {MODEL_CONFIG['confidence_threshold']} threshold: {confident_signals}")

print("\n=== get_leverage ===")
assert get_leverage(0.79) == 1.0
assert get_leverage(0.82) == 1.0
assert get_leverage(0.88) == 2.0
assert get_leverage(0.95) == 3.0
print("get_leverage mappings OK.")

print("\nStep 4 Validation: PASS")
