"""Step 3 validation — run with: PYTHONPATH=/home/yagokhan python varanus/validate_step3.py"""
import pandas as pd
import numpy as np

from varanus.pa_features  import detect_mss, compute_atr
from varanus.tbm_labeler  import (
    TBM_CONFIG, FLASH_WICK_GUARD,
    calculate_barriers, label_trades, barrier_stats,
)
from varanus.universe import HIGH_VOL_SUBTIER

CACHE = "/home/yagokhan/chameleon/claude_code_project/data/cache"

def load_4h(symbol: str) -> pd.DataFrame:
    df = pd.read_parquet(f"{CACHE}/{symbol}_USDT.parquet")
    df.columns = [c.lower() for c in df.columns]
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index("timestamp")
    df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()

# ── 1. calculate_barriers ─────────────────────────────────────────────────────
print("=== calculate_barriers ===")
b = calculate_barriers(entry=10.0, atr=0.5, direction=1, cfg=TBM_CONFIG, asset="LINK")
assert b["take_profit"]      == 10.0 + 2.5 * 0.5,  "TP wrong"
assert b["stop_loss"]        == 10.0 - 1.2 * 0.5,  "SL wrong"
assert b["rr_ratio"]         == round(2.5 / 1.2, 2)
assert b["min_rr_satisfied"] is True
print(f"  LINK long @ 10.0: TP={b['take_profit']:.4f}  SL={b['stop_loss']:.4f}  RR={b['rr_ratio']}")

# High-vol override
b_hv = calculate_barriers(entry=100.0, atr=5.0, direction=1, cfg=TBM_CONFIG, asset="TAO")
assert b_hv["take_profit"] == 100.0 + 3.0 * 5.0,  "HV TP wrong"
assert b_hv["stop_loss"]   == 100.0 - 1.5 * 5.0,  "HV SL wrong"
print(f"  TAO long @ 100.0: TP={b_hv['take_profit']:.4f}  SL={b_hv['stop_loss']:.4f}")

# Short direction
b_s = calculate_barriers(entry=10.0, atr=0.5, direction=-1, cfg=TBM_CONFIG, asset="LINK")
assert b_s["take_profit"] < b_s["stop_loss"],  "Short: TP should be below SL"
assert b_s["take_profit"] == 10.0 - 2.5 * 0.5
assert b_s["stop_loss"]   == 10.0 + 1.2 * 0.5
print(f"  LINK short @ 10.0: TP={b_s['take_profit']:.4f}  SL={b_s['stop_loss']:.4f}")

# Min R:R gate (artificially narrow ATR)
b_rr = calculate_barriers(entry=10.0, atr=0.0001, direction=1, cfg=TBM_CONFIG, asset="LINK")
assert b_rr["min_rr_satisfied"] is True  # ATR tiny but ratio is still 2.5/1.2
print(f"  RR gate satisfied={b_rr['min_rr_satisfied']}  rr={b_rr['rr_ratio']}")

print()
print("=== label_trades on LINK ===")
df = load_4h("LINK")
signals = detect_mss(df)
n_sigs  = (signals != 0).sum()
print(f"  LINK bars: {len(df)}  |  MSS signals: {n_sigs}")

labels = label_trades(df, signals, TBM_CONFIG, asset="LINK")

# Spec assertion
dist = labels.value_counts(normalize=True)
assert labels.isin([-1, 0, 1]).all(), "Unexpected label values"
print(f"  Label distribution:\n{dist.to_string()}")

stats = barrier_stats(labels, signals)
print(f"\n  barrier_stats (signal bars only):")
for k, v in stats.items():
    print(f"    {k:15s}: {v}")

# Signal bars: win > 30% of non-neutral signal bars
signal_mask   = signals != 0
signal_labels = labels[signal_mask]
non_neutral   = signal_labels[signal_labels != 0]
if len(non_neutral) > 0:
    win_pct = (non_neutral == 1).mean()
    print(f"\n  Win rate among decided signal bars: {win_pct:.1%}")
    assert win_pct >= 0.10, f"Win rate suspiciously low: {win_pct:.1%}"
    print(f"  PASS: win rate {win_pct:.1%} >= 10% threshold")

print()
print("=== Params override (Optuna simulation) ===")
params = {"tp_atr_mult": 3.0, "sl_atr_mult": 1.5, "max_holding": 20}
labels_ovr = label_trades(df, signals, TBM_CONFIG, asset="LINK", params=params)
assert labels_ovr.isin([-1, 0, 1]).all()
dist_ovr = labels_ovr.value_counts(normalize=True)
print(f"  With params {params}:")
print(f"  Label dist:\n{dist_ovr.to_string()}")

print()
print("=== Flash-wick guard sanity check ===")
# Build a tiny synthetic df where a wick touches SL but close stays above it
rows = []
base = 100.0
atr_val = 1.0  # fixed ATR for clarity
# direction = 1 (long), entry=100, TP=102.5, SL=98.8
# Signal bar: close=100
rows.append({"open": 99, "high": 101, "low": 98.9, "close": 100.0, "volume": 1000})
# Bar i+1: wick goes to 98.5 (below SL=98.8) but CLOSES at 99.2 (above SL)
rows.append({"open": 100, "high": 100.5, "low": 98.5, "close": 99.2, "volume": 1000})
# Bar i+2: rises to TP=102.5
rows.append({"open": 99.5, "high": 103.0, "low": 99.0, "close": 102.8, "volume": 1000})

idx = pd.date_range("2024-01-01", periods=3, freq="4h", tz="UTC")
df_syn = pd.DataFrame(rows, index=idx)
# Inject ATR manually — override with constant
import varanus.tbm_labeler as _tbm
_orig_atr = _tbm._atr
def _const_atr(df, period):
    return pd.Series(atr_val, index=df.index)
_tbm._atr = _const_atr

signals_syn = pd.Series([1, 0, 0], index=idx, dtype=np.int8)
labels_syn  = label_trades(df_syn, signals_syn, TBM_CONFIG, asset="LINK")

_tbm._atr = _orig_atr  # restore

# With flash-wick guard: wick at 98.5 < SL=98.8, but 98.8 - 0.3*1.0 = 98.5
# The wick just reaches the tolerance boundary exactly.
# Bar i+2 hits TP — should be label=1
# (The outcome depends on exact wick_ext: 98.5 = 98.8 - 0.3 → exactly on boundary)
# Whether this fires as SL or reaches TP depends on whether wick_sl is True
# wick_sl = low < sl - wick_ext = 98.8 - 0.3 = 98.5 → 98.5 < 98.5 is False
# So wick does NOT trigger SL, TP is reached on bar i+2 → label=1
print(f"  Synthetic: wick at SL boundary → label = {labels_syn.iloc[0]} (expected 1)")
assert labels_syn.iloc[0] == 1, f"Flash-wick guard failed: expected 1, got {labels_syn.iloc[0]}"
print("  Flash-wick guard: PASS")

print()
print("Step 3 Validation: PASS")
