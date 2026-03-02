"""
varanus/dashboard.py — Interactive Investigation Dashboard (v5.0).

Tabs:
  1. Overview       — Project status, config, asset universe
  2. Data Explorer  — Per-asset OHLCV, features, TBM labels
  3. Backtest Lab   — Run demo backtest + full results investigation
  4. Walk-Forward   — 8-fold WFV summary, per-fold equity, consistency
  5. Model Insights — Feature importance, confidence tiers, leverage
  6. HPO Explorer   — Optuna trial history, best params, search space

Run:
    cd /home/yagokhan
    PYTHONPATH=/home/yagokhan streamlit run varanus/dashboard.py
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.chdir(Path(__file__).resolve().parent)   # keep relative paths working

import numpy  as np
import pandas as pd
import plotly.express        as px
import plotly.graph_objects  as go
from plotly.subplots import make_subplots
import streamlit as st
import yaml

# ── Varanus imports ───────────────────────────────────────────────────────────
from varanus.universe    import TIER2_UNIVERSE, HIGH_VOL_SUBTIER
from varanus.model       import FEATURE_LIST, CONFIDENCE_LEVERAGE_MAP_V5, get_leverage, is_power_setup
from varanus.backtest    import BACKTEST_CONFIG, BACKTEST_PASS_CRITERIA, compute_metrics, run_backtest
from varanus.optimizer   import V5_SEARCH_SPACE, V5_TIME_EXIT_BARS
from varanus.walk_forward import WFV_CONFIG
from varanus.risk        import RISK_CONFIG
from varanus.tbm_labeler import TBM_CONFIG

# ── Constants ─────────────────────────────────────────────────────────────────
CACHE    = "/home/yagokhan/chameleon/claude_code_project/data/cache"
CFG_DIR  = Path(__file__).parent / "config"
RES_DIR  = Path(__file__).parent / "results"
PARAMS_PATH = CFG_DIR / "best_params_v5.json"
WFV_CSV     = RES_DIR / "step6_wfv_v5.csv"
TRADE_CSV   = RES_DIR / "step5_trades_v5.csv"
OPTUNA_DB   = CFG_DIR / "optuna_v5.db"
VERSION_YML = CFG_DIR / "version.yaml"

# Demo params (v4 best → reasonable v5 defaults for instant preview)
DEMO_PARAMS = {
    "confidence_thresh":  0.8147,
    "sl_atr_mult":        0.835,
    "rr_ratio":           4.1,
    "mss_lookback":       31,
    "fvg_min_atr_ratio":  0.392,
    "fvg_max_age":        22,
    "sweep_min_pct":      0.00641,
    "rvol_threshold":     1.287,
    "rsi_oversold":       36,
    "rsi_overbought":     58,
    "max_holding_candles": 31,
    "xgb_max_depth":      7,
    "xgb_n_estimators":   218,
    "xgb_lr":             0.0656,
    "xgb_subsample":      0.957,
}

DEMO_ASSETS = ["BNB", "SOL", "ADA", "LINK", "AVAX"]
UNIVERSE_DF = pd.DataFrame([
    {"Asset": a, "Tier": "High-Vol ⚡" if a in HIGH_VOL_SUBTIER else "Standard",
     "Size Scalar": 0.75 if a in HIGH_VOL_SUBTIER else 1.0}
    for a in TIER2_UNIVERSE
])

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Varanus v5 Dashboard",
    page_icon="🦎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🦎 Varanus v5.0")
    st.caption("High-Resolution Regime Adaptation")
    st.divider()

    # Quick status badges
    params_ready = PARAMS_PATH.exists() and json.loads(PARAMS_PATH.read_text()).get("confidence_thresh") is not None
    wfv_ready    = WFV_CSV.exists()
    trades_ready = TRADE_CSV.exists()
    optuna_ready = OPTUNA_DB.exists()

    st.markdown("**Pipeline Status**")
    st.markdown(f"{'✅' if params_ready else '⏳'} HPO / Best Params")
    st.markdown(f"{'✅' if optuna_ready else '⏳'} Optuna DB")
    st.markdown(f"{'✅' if wfv_ready   else '⏳'} WFV Results")
    st.markdown(f"{'✅' if trades_ready else '⏳'} Trade Log")
    st.divider()

    st.markdown("**Run**")
    st.code("PYTHONPATH=/home/yagokhan\npython3 varanus/run_optimization.py", language="bash")


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading OHLCV…")
def load_ohlcv(symbol: str, timeframe: str = "4h") -> pd.DataFrame | None:
    """Load a single asset from the Chameleon parquet cache."""
    file_symbol = "ASTER" if symbol == "ASTR" else symbol
    try:
        if timeframe == "4h":
            path = f"{CACHE}/{file_symbol}_USDT.parquet"
        else:
            path = f"{CACHE}/{file_symbol}_USDT_1h.parquet"
        df = pd.read_parquet(path)
        df.columns = [c.lower() for c in df.columns]
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.set_index("timestamp")
        df.index = pd.to_datetime(df.index, utc=True)
        return df.sort_index()
    except Exception:
        return None


@st.cache_data(show_spinner="Loading universe…")
def load_universe_data(assets: list[str]) -> tuple[dict, dict]:
    data_4h, data_1d = {}, {}
    for a in assets:
        d4 = load_ohlcv(a, "4h")
        if d4 is not None:
            data_4h[a] = d4
            # Resample 1h→1d for HTF bias
            d1 = load_ohlcv(a, "1h")
            if d1 is not None:
                d1 = d1.resample("1D").agg(
                    {"open": "first", "high": "max",
                     "low": "min", "close": "last", "volume": "sum"}
                ).dropna()
                data_1d[a] = d1
            else:
                data_1d[a] = d4
    return data_4h, data_1d


def load_best_params() -> dict:
    if PARAMS_PATH.exists():
        raw = json.loads(PARAMS_PATH.read_text())
        return {k: v for k, v in raw.items() if not k.startswith("_")}
    return DEMO_PARAMS.copy()


def load_version_yaml() -> dict:
    if VERSION_YML.exists():
        return yaml.safe_load(VERSION_YML.read_text()) or {}
    return {}


@st.cache_data
def load_wfv_results() -> pd.DataFrame | None:
    if WFV_CSV.exists():
        return pd.read_csv(WFV_CSV)
    return None


@st.cache_data
def load_trade_log() -> pd.DataFrame | None:
    if TRADE_CSV.exists():
        df = pd.read_csv(TRADE_CSV, parse_dates=["entry_ts", "exit_ts"])
        return df
    return None


def _metric_card(label: str, value: str, delta: str = "", color: str = "normal"):
    st.metric(label, value, delta=delta if delta else None)


def _confidence_tier(c: float) -> str:
    if c >= 0.95:  return "⚡ Power Setup (5×)"
    if c >= 0.92:  return "🔴 High Conviction (3×)"
    if c >= 0.85:  return "🟡 Standard (2×)"
    return "⚪ Base (1×)"


def _equity_fig(equity: pd.Series, title: str = "Equity Curve") -> go.Figure:
    roll_max = equity.cummax()
    dd       = (equity - roll_max) / roll_max * 100

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3],
                        subplot_titles=[title, "Drawdown %"])
    fig.add_trace(go.Scatter(x=equity.index, y=equity.values,
                             name="Equity", line=dict(color="#00b4d8", width=2)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values,
                             name="Drawdown %", fill="tozeroy",
                             line=dict(color="#ef476f", width=1)),
                  row=2, col=1)
    fig.update_layout(template="plotly_dark", height=500,
                      legend=dict(orientation="h"), margin=dict(l=0, r=0, t=40, b=0))
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview",
    "🔍 Data Explorer",
    "💰 Backtest Lab",
    "📅 Walk-Forward",
    "🤖 Model Insights",
    "⚙️ HPO Explorer",
])


# ────────────────────────────────────────────────────────────────────────────
# TAB 1 — Overview
# ────────────────────────────────────────────────────────────────────────────
with tab1:
    ver = load_version_yaml()
    st.header("Varanus v5.0 — High-Resolution Regime Adaptation")
    st.caption(f"Version {ver.get('version','5.0.0')} | Tier {ver.get('tier',2)} | "
               f"{ver.get('asset_class','Mid-Cap Crypto')} | Status: {ver.get('status','—')}")
    st.divider()

    # ── Key architecture cards ───────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("WFV Folds",        "8")
    c2.metric("Split Ratio",      "40/30/30")
    c3.metric("Time Exit (bars)", "31 🔒")
    c4.metric("Universe",         "20 assets")
    c5.metric("HPO Objective",    "PF × log1p(Net%)")

    st.divider()

    # ── Universe table ────────────────────────────────────────────────────────
    col_l, col_r = st.columns([1.5, 1])
    with col_l:
        st.subheader("Asset Universe (20 Tier 2 Assets)")
        st.dataframe(
            UNIVERSE_DF.set_index("Asset"),
            use_container_width=True,
            height=380,
        )

    with col_r:
        st.subheader("V5 Leverage Map")
        lev_df = pd.DataFrame([
            {"Tier": "Base",           "Confidence": "[0.750, 0.850)", "Leverage": "1×", "Size Scalar": "1.00×"},
            {"Tier": "Standard",       "Confidence": "[0.850, 0.920)", "Leverage": "2×", "Size Scalar": "1.00×"},
            {"Tier": "High Conviction","Confidence": "[0.920, 0.950)", "Leverage": "3×", "Size Scalar": "1.00×"},
            {"Tier": "⚡ Power Setup", "Confidence": "[0.950, 1.000]", "Leverage": "5×", "Size Scalar": "1.25×"},
        ])
        st.dataframe(lev_df.set_index("Tier"), use_container_width=True)

        st.subheader("Acceptance Gate (per fold)")
        gate_df = pd.DataFrame([
            {"Metric": "Min Trades",        "Threshold": "≥ 50"},
            {"Metric": "Min Win Rate",       "Threshold": "≥ 43%"},
            {"Metric": "Min Profit Factor",  "Threshold": "≥ 1.50"},
            {"Metric": "Max Drawdown",       "Threshold": "≥ −15%"},
            {"Metric": "Min Sharpe",         "Threshold": "≥ 0.90"},
            {"Metric": "Min Calmar",         "Threshold": "≥ 0.60"},
        ])
        st.dataframe(gate_df.set_index("Metric"), use_container_width=True)

    st.divider()

    # ── Best params panel ─────────────────────────────────────────────────────
    st.subheader("Best Parameters (HPO Output)")
    params = load_best_params()
    if not params_ready:
        st.warning("⏳ HPO has not been run yet. Showing demo defaults (v4 best params).")

    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        st.markdown("**Core Execution**")
        for k in ["confidence_thresh", "sl_atr_mult", "rr_ratio"]:
            v = params.get(k)
            st.metric(k, f"{v:.4f}" if v is not None else "—")
    with pc2:
        st.markdown("**Feature Engineering**")
        for k in ["mss_lookback", "fvg_min_atr_ratio", "fvg_max_age", "rvol_threshold"]:
            v = params.get(k)
            st.metric(k, f"{v}" if v is not None else "—")
    with pc3:
        st.markdown("**XGBoost**")
        for k in ["xgb_max_depth", "xgb_n_estimators", "xgb_lr", "xgb_subsample"]:
            v = params.get(k)
            st.metric(k, f"{v}" if v is not None else "—")

    with st.expander("Show raw JSON"):
        st.json(params)


# ────────────────────────────────────────────────────────────────────────────
# TAB 2 — Data Explorer
# ────────────────────────────────────────────────────────────────────────────
with tab2:
    st.header("Data Explorer")

    de_asset = st.selectbox("Select asset", TIER2_UNIVERSE, key="de_asset")
    df_4h    = load_ohlcv(de_asset, "4h")

    if df_4h is None:
        st.error(f"No cache data found for {de_asset}. Check CACHE path.")
    else:
        # ── Date range + stats ─────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Candles (4h)", f"{len(df_4h):,}")
        c2.metric("From",         str(df_4h.index[0].date()))
        c3.metric("To",           str(df_4h.index[-1].date()))
        c4.metric("Days",         str((df_4h.index[-1] - df_4h.index[0]).days))

        st.divider()

        # ── OHLCV Chart ────────────────────────────────────────────────────
        st.subheader("Price Chart (4h)")
        n_bars = st.slider("Show last N bars", 100, len(df_4h), min(500, len(df_4h)), key="de_bars")
        df_plot = df_4h.tail(n_bars)

        fig_ohlcv = go.Figure()
        fig_ohlcv.add_trace(go.Candlestick(
            x=df_plot.index, open=df_plot["open"], high=df_plot["high"],
            low=df_plot["low"], close=df_plot["close"], name="OHLCV",
        ))
        fig_ohlcv.update_layout(
            template="plotly_dark", xaxis_rangeslider_visible=False,
            height=400, margin=dict(l=0, r=0, t=20, b=0),
        )
        st.plotly_chart(fig_ohlcv, use_container_width=True)

        # ── Volume ─────────────────────────────────────────────────────────
        fig_vol = px.bar(df_plot.reset_index(), x=df_plot.index, y="volume",
                         title="Volume", template="plotly_dark", height=200,
                         color_discrete_sequence=["#0077b6"])
        fig_vol.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_vol, use_container_width=True)

        st.divider()

        # ── ATR & derived stats ────────────────────────────────────────────
        st.subheader("Volatility (ATR 14)")
        tr  = pd.concat([
            df_4h["high"] - df_4h["low"],
            (df_4h["high"] - df_4h["close"].shift()).abs(),
            (df_4h["low"]  - df_4h["close"].shift()).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        atr_pct = (atr / df_4h["close"] * 100)

        vc1, vc2, vc3 = st.columns(3)
        vc1.metric("Current ATR",         f"{atr.iloc[-1]:.4f}")
        vc2.metric("ATR% of Price",        f"{atr_pct.iloc[-1]:.2f}%")
        vc3.metric("High-Vol Sub-Tier",    "Yes ⚡" if de_asset in HIGH_VOL_SUBTIER else "No")

        fig_atr = go.Figure()
        fig_atr.add_trace(go.Scatter(x=atr.tail(n_bars).index,
                                     y=atr.tail(n_bars).values,
                                     name="ATR(14)", line=dict(color="#f77f00")))
        fig_atr.update_layout(template="plotly_dark", height=250,
                               margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_atr, use_container_width=True)

        # ── RSI ────────────────────────────────────────────────────────────
        st.subheader("RSI (14)")
        delta = df_4h["close"].diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        rsi   = 100 - 100 / (1 + rs)

        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=rsi.tail(n_bars).index,
                                     y=rsi.tail(n_bars).values,
                                     name="RSI(14)", line=dict(color="#9d4edd")))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red",   annotation_text="Overbought 70")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold 30")
        fig_rsi.update_layout(template="plotly_dark", height=250,
                               margin=dict(l=0, r=0, t=10, b=0), yaxis_range=[0, 100])
        st.plotly_chart(fig_rsi, use_container_width=True)

        # ── Returns distribution ───────────────────────────────────────────
        st.subheader("4h Returns Distribution")
        rets = df_4h["close"].pct_change().dropna() * 100
        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("Mean 4h Return",  f"{rets.mean():.4f}%")
        rc2.metric("Std Dev",         f"{rets.std():.4f}%")
        rc3.metric("Skewness",        f"{rets.skew():.3f}")
        rc4.metric("Kurtosis",        f"{rets.kurt():.3f}")

        fig_ret = px.histogram(rets, nbins=80, template="plotly_dark",
                               title="4h Returns (%)", height=300,
                               color_discrete_sequence=["#48cae4"])
        fig_ret.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_ret, use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────
# TAB 3 — Backtest Lab
# ────────────────────────────────────────────────────────────────────────────
with tab3:
    st.header("Backtest Lab")

    # ── Load saved results or offer demo run ─────────────────────────────────
    saved_trades = load_trade_log()

    if saved_trades is not None:
        st.success(f"Loaded saved trade log: {len(saved_trades)} trades from {TRADE_CSV.name}")
        trades_df = saved_trades
    else:
        st.info("No saved trade log found. Use the controls below to run a demo backtest.")
        trades_df = None

    with st.expander("⚙️ Demo Backtest Controls", expanded=(trades_df is None)):
        bl_assets  = st.multiselect("Assets to include",
                                    TIER2_UNIVERSE, default=DEMO_ASSETS, key="bl_assets")
        bl_conf    = st.slider("Confidence threshold", 0.70, 0.95, DEMO_PARAMS["confidence_thresh"],
                               step=0.005, key="bl_conf")
        bl_sl      = st.slider("SL ATR multiplier",   0.50, 1.50, DEMO_PARAMS["sl_atr_mult"],
                               step=0.05, key="bl_sl")
        bl_rr      = st.slider("R:R ratio",            2.0,  6.0, DEMO_PARAMS["rr_ratio"],
                               step=0.1,  key="bl_rr")
        bl_run = st.button("▶ Run Demo Backtest", type="primary", key="bl_run")

        if bl_run:
            with st.spinner("Loading data and running backtest…"):
                from varanus.pa_features import build_features, compute_atr
                from varanus.tbm_labeler import label_trades
                from varanus.model       import VaranusModel, MODEL_CONFIG

                bl_params = {**DEMO_PARAMS,
                             "confidence_thresh": bl_conf,
                             "sl_atr_mult": bl_sl,
                             "rr_ratio": bl_rr}
                data_4h, data_1d = load_universe_data(bl_assets)

                if not data_4h:
                    st.error("No data loaded — check cache path.")
                else:
                    # Single-split: train on first 60%, test on last 40%
                    all_ts    = sorted(set().union(*[set(d.index) for d in data_4h.values()]))
                    split_idx = int(len(all_ts) * 0.60)
                    split_ts  = all_ts[split_idx]

                    X_list, y_list = [], []
                    for asset, df_4h in data_4h.items():
                        df_tr = df_4h[df_4h.index < split_ts]
                        d1_tr = data_1d.get(asset, df_4h)[
                            data_1d.get(asset, df_4h).index < split_ts]
                        try:
                            X = build_features(df_tr, d1_tr, asset, bl_params)
                        except Exception:
                            continue
                        if X.empty: continue
                        mss = X["mss_signal"]
                        from varanus.tbm_labeler import TBM_CONFIG
                        y = label_trades(df_tr.loc[X.index], mss,
                                         TBM_CONFIG, asset, bl_params).reindex(X.index).fillna(0).astype(int)
                        X_list.append(X); y_list.append(y)

                    if not X_list:
                        st.error("Feature extraction failed for all assets.")
                    else:
                        X_train = pd.concat(X_list)
                        y_train = pd.concat(y_list)
                        model_cfg = MODEL_CONFIG.copy()
                        model_cfg.update({k: bl_params[k] for k in
                                          ["xgb_max_depth","xgb_n_estimators","xgb_lr","xgb_subsample"]
                                          if k in bl_params})
                        model = VaranusModel(model_cfg)
                        model.fit(X_train, y_train)

                        from varanus.walk_forward import generate_signals
                        test_4h = {a: d[d.index >= split_ts] for a, d in data_4h.items()}
                        test_1d = {a: d[d.index >= split_ts] for a, d in data_1d.items()}
                        signals = generate_signals(test_4h, test_1d, model, bl_params)

                        if not signals:
                            st.warning("No signals generated in test period.")
                        else:
                            equity, trades_df = run_backtest(test_4h, signals, model, bl_params)
                            st.session_state["bt_equity"]  = equity
                            st.session_state["bt_trades"]  = trades_df
                            st.session_state["bt_metrics"] = compute_metrics(equity, trades_df)
                            st.success(f"Done! {len(trades_df)} trades generated.")
                            st.rerun()

    # ── Show results if available ────────────────────────────────────────────
    equity  = st.session_state.get("bt_equity")
    trades  = st.session_state.get("bt_trades", trades_df)
    metrics = st.session_state.get("bt_metrics")

    if trades is not None and len(trades) > 0:
        if metrics is None:
            metrics = compute_metrics(
                equity if equity is not None else pd.Series(dtype=float), trades
            )

        st.divider()
        st.subheader("Performance Metrics")

        mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
        mc1.metric("Total Return",   f"{metrics['total_return_pct']:.2f}%")
        mc2.metric("CAGR",           f"{metrics['cagr_pct']:.2f}%")
        mc3.metric("Max Drawdown",   f"{metrics['max_drawdown_pct']:.2f}%")
        mc4.metric("Sharpe",         f"{metrics['sharpe_ratio']:.3f}")
        mc5.metric("Profit Factor",  f"{metrics['profit_factor']:.2f}")
        mc6.metric("Win Rate",       f"{metrics['win_rate_pct']:.1f}%")

        tc1, tc2, tc3, tc4 = st.columns(4)
        tc1.metric("Total Trades",  str(metrics["total_trades"]))
        tc2.metric("TP Hits",       str(metrics["tp_hits"]))
        tc3.metric("SL Hits",       str(metrics["sl_hits"]))
        tc4.metric("Time Exits",    str(metrics["time_exits"]))

        # Acceptance gate pass/fail
        gate = BACKTEST_PASS_CRITERIA
        gate_checks = {
            "Min Trades ≥ 50":    metrics["total_trades"]     >= gate["min_trades"],
            "Win Rate ≥ 43%":     metrics["win_rate_pct"]     >= gate["min_win_rate"] * 100,
            "PF ≥ 1.50":          metrics["profit_factor"]    >= gate["min_profit_factor"],
            "MaxDD ≥ −15%":       metrics["max_drawdown_pct"] >= gate["max_drawdown"] * 100,
            "Sharpe ≥ 0.90":      metrics["sharpe_ratio"]     >= gate["min_sharpe"],
            "Calmar ≥ 0.60":      metrics["calmar_ratio"]     >= gate["min_calmar"],
        }
        gate_label = "✅ PASS" if all(gate_checks.values()) else "❌ FAIL"
        st.subheader(f"Acceptance Gate: {gate_label}")
        gcols = st.columns(6)
        for i, (lbl, ok) in enumerate(gate_checks.items()):
            gcols[i].markdown(f"{'✅' if ok else '❌'} {lbl}")

        st.divider()

        # ── Equity curve ─────────────────────────────────────────────────────
        if equity is not None and len(equity) > 0:
            st.subheader("Equity Curve & Drawdown")
            st.plotly_chart(_equity_fig(equity), use_container_width=True)

        # ── Trade log ─────────────────────────────────────────────────────────
        st.subheader("Trade Log")
        tl = trades.copy()

        # Filters
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            f_assets = st.multiselect("Filter assets", sorted(tl["asset"].unique()),
                                      key="tl_assets")
        with fc2:
            f_outcome = st.multiselect("Filter outcome", ["tp", "sl", "time"],
                                       key="tl_outcome")
        with fc3:
            f_dir = st.multiselect("Direction", [1, -1],
                                   format_func=lambda x: "LONG" if x == 1 else "SHORT",
                                   key="tl_dir")

        if f_assets:  tl = tl[tl["asset"].isin(f_assets)]
        if f_outcome: tl = tl[tl["outcome"].isin(f_outcome)]
        if f_dir:     tl = tl[tl["direction"].isin(f_dir)]

        disp_cols = [c for c in ["asset", "direction", "entry_ts", "exit_ts",
                                  "entry_price", "exit_price", "outcome",
                                  "pnl_usd", "confidence", "leverage",
                                  "rr_ratio", "power_setup"] if c in tl.columns]
        st.dataframe(tl[disp_cols].sort_values("entry_ts", ascending=False),
                     use_container_width=True, height=300)

        # CSV download
        st.download_button("⬇ Download Trade Log CSV",
                           data=tl.to_csv(index=False),
                           file_name="varanus_v5_trades.csv",
                           mime="text/csv")

        st.divider()

        # ── Charts ───────────────────────────────────────────────────────────
        ch1, ch2 = st.columns(2)

        with ch1:
            # PnL distribution
            st.subheader("PnL Distribution")
            fig_pnl = px.histogram(trades, x="pnl_usd", nbins=50,
                                   color_discrete_sequence=["#06d6a0"],
                                   template="plotly_dark", height=300)
            fig_pnl.add_vline(x=0, line_dash="dash", line_color="white")
            fig_pnl.update_layout(margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_pnl, use_container_width=True)

        with ch2:
            # Exit type pie
            st.subheader("Exit Type Breakdown")
            out_counts = trades["outcome"].value_counts().reset_index()
            out_counts.columns = ["outcome", "count"]
            fig_pie = px.pie(out_counts, names="outcome", values="count",
                             color_discrete_map={"tp": "#06d6a0", "sl": "#ef476f", "time": "#ffd166"},
                             template="plotly_dark", height=300)
            fig_pie.update_layout(margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_pie, use_container_width=True)

        # Per-asset PnL
        st.subheader("Cumulative PnL by Asset")
        if "asset" in trades.columns:
            asset_pnl = (
                trades.groupby("asset")["pnl_usd"]
                .agg(total="sum", count="count", win_rate=lambda x: (x > 0).mean() * 100)
                .reset_index()
                .sort_values("total", ascending=False)
            )
            fig_bar = px.bar(asset_pnl, x="asset", y="total",
                             color="win_rate", color_continuous_scale="RdYlGn",
                             template="plotly_dark", height=350,
                             labels={"total": "Total PnL ($)", "win_rate": "Win Rate %"})
            fig_bar.update_layout(margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_bar, use_container_width=True)

        # Confidence vs PnL scatter
        if "confidence" in trades.columns:
            st.subheader("Confidence vs PnL")
            fig_sc = px.scatter(trades, x="confidence", y="pnl_usd",
                                color="outcome",
                                color_discrete_map={"tp": "#06d6a0", "sl": "#ef476f", "time": "#ffd166"},
                                template="plotly_dark", height=350,
                                labels={"confidence": "Model Confidence", "pnl_usd": "PnL ($)"})
            fig_sc.add_vline(x=0.950, line_dash="dash", line_color="yellow",
                              annotation_text="Power Setup threshold")
            fig_sc.update_layout(margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_sc, use_container_width=True)

        # Rolling win rate
        if equity is not None and len(equity) > 10:
            st.subheader("Rolling Performance")
            eq_ret    = equity.pct_change().dropna()
            roll_win  = eq_ret.rolling(20).apply(lambda x: (x > 0).mean() * 100)
            roll_vol  = eq_ret.rolling(20).std() * np.sqrt(365 * 6) * 100

            fig_roll = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                     subplot_titles=["20-bar Rolling Win Rate (%)",
                                                     "20-bar Annualised Volatility (%)"])
            fig_roll.add_trace(go.Scatter(x=roll_win.index, y=roll_win.values,
                                          name="Win Rate", line=dict(color="#06d6a0")),
                               row=1, col=1)
            fig_roll.add_trace(go.Scatter(x=roll_vol.index, y=roll_vol.values,
                                          name="Vol %", line=dict(color="#f77f00")),
                               row=2, col=1)
            fig_roll.update_layout(template="plotly_dark", height=400,
                                   margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_roll, use_container_width=True)

    elif not st.session_state.get("bt_trades") and trades_df is None:
        st.info("Run the demo backtest above to see results.")


# ────────────────────────────────────────────────────────────────────────────
# TAB 4 — Walk-Forward
# ────────────────────────────────────────────────────────────────────────────
with tab4:
    st.header("Walk-Forward Validation (8-Fold)")

    wfv_df = load_wfv_results()
    saved_wfv = wfv_df is not None

    if not saved_wfv:
        st.info("No WFV results found. You can run WFV from the command line:")
        st.code("PYTHONPATH=/home/yagokhan python3 varanus/run_optimization.py", language="bash")

        # ── Live WFV Run Button ───────────────────────────────────────────────
        with st.expander("⚙️ Run Walk-Forward Validation in Dashboard", expanded=False):
            wfv_assets = st.multiselect("Assets (fewer = faster)", TIER2_UNIVERSE,
                                         default=DEMO_ASSETS[:3], key="wfv_assets")
            wfv_run = st.button("▶ Run WFV (demo, ~2-5 min)", type="primary", key="wfv_run")

            if wfv_run and wfv_assets:
                with st.spinner("Running 8-fold WFV… this may take several minutes"):
                    from varanus.walk_forward import run_walk_forward as _rwfv
                    d4h, d1d = load_universe_data(wfv_assets)
                    if d4h:
                        wfv_result = _rwfv(d4h, d1d, DEMO_PARAMS)
                        wfv_df, consistency, wfv_trades = wfv_result
                        st.session_state["wfv_df"]     = wfv_df
                        st.session_state["wfv_trades"] = wfv_trades
                        st.session_state["wfv_con"]    = consistency
                        st.success("WFV complete!")
                        st.rerun()

        wfv_df    = st.session_state.get("wfv_df")
        wfv_trades = st.session_state.get("wfv_trades")
        consistency = st.session_state.get("wfv_con", 0.0)
    else:
        consistency = (wfv_df["fold_pass"].sum() / len(wfv_df)
                       if "fold_pass" in wfv_df.columns else 0.0)

    if wfv_df is not None and len(wfv_df) > 0:
        st.divider()

        # ── Summary badges ────────────────────────────────────────────────────
        n_folds   = len(wfv_df)
        n_pass    = int(wfv_df["fold_pass"].sum()) if "fold_pass" in wfv_df.columns else 0
        wfv_pass  = consistency >= WFV_CONFIG.get("consistency_req", 0.75)

        wc1, wc2, wc3, wc4, wc5 = st.columns(5)
        wc1.metric("Folds Run",    str(n_folds))
        wc2.metric("Folds PASS",   f"{n_pass} / {n_folds}")
        wc3.metric("Consistency",  f"{consistency:.0%}")
        wc4.metric("WFV Result",   "✅ PASS" if wfv_pass else "❌ FAIL")
        wc5.metric("Avg Calmar",   f"{wfv_df['calmar_ratio'].mean():.3f}" if "calmar_ratio" in wfv_df.columns else "—")

        st.subheader("Fold Summary Table")
        style_cols = ["fold_pass"] if "fold_pass" in wfv_df.columns else []
        st.dataframe(wfv_df.set_index("fold") if "fold" in wfv_df.columns else wfv_df,
                     use_container_width=True)

        st.divider()

        # ── Per-fold metric charts ────────────────────────────────────────────
        metric_cols = [c for c in ["profit_factor", "win_rate_pct", "max_drawdown_pct",
                                    "calmar_ratio", "sharpe_ratio", "total_trades"]
                       if c in wfv_df.columns]

        if metric_cols:
            sel_metric = st.selectbox("Metric to chart", metric_cols, key="wfv_metric")
            fold_x = wfv_df["fold"].astype(str) if "fold" in wfv_df.columns else wfv_df.index.astype(str)
            colors = ["#06d6a0" if v else "#ef476f"
                      for v in wfv_df.get("fold_pass", [True]*len(wfv_df))]

            fig_wfv = go.Figure(go.Bar(
                x=["Fold " + str(f) for f in fold_x],
                y=wfv_df[sel_metric],
                marker_color=colors,
                name=sel_metric,
            ))
            fig_wfv.update_layout(template="plotly_dark", height=350,
                                  title=f"Per-Fold {sel_metric}",
                                  margin=dict(l=0, r=0, t=40, b=0))
            if sel_metric == "max_drawdown_pct":
                fig_wfv.add_hline(y=-15, line_dash="dash", line_color="red",
                                   annotation_text="Max DD Gate −15%")
            if sel_metric == "profit_factor":
                fig_wfv.add_hline(y=1.5, line_dash="dash", line_color="green",
                                   annotation_text="PF Gate 1.50")
            st.plotly_chart(fig_wfv, use_container_width=True)

        # ── Radar chart (multi-metric per fold) ───────────────────────────────
        radar_metrics = [c for c in ["profit_factor", "win_rate_pct", "calmar_ratio",
                                      "sharpe_ratio"]
                         if c in wfv_df.columns]
        if len(radar_metrics) >= 3:
            st.subheader("Multi-Metric Radar (normalised)")
            norm = wfv_df[radar_metrics].copy()
            for col in norm.columns:
                cmin, cmax = norm[col].min(), norm[col].max()
                norm[col] = (norm[col] - cmin) / (cmax - cmin + 1e-9)

            fig_radar = go.Figure()
            for _, row in norm.iterrows():
                fold_id = int(wfv_df.loc[row.name, "fold"]) if "fold" in wfv_df.columns else row.name
                fig_radar.add_trace(go.Scatterpolar(
                    r=row[radar_metrics].tolist() + [row[radar_metrics[0]]],
                    theta=radar_metrics + [radar_metrics[0]],
                    fill="toself", name=f"Fold {fold_id}",
                ))
            fig_radar.update_layout(template="plotly_dark", height=400,
                                    polar=dict(radialaxis=dict(range=[0, 1])),
                                    margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig_radar, use_container_width=True)

        # ── All-trades analysis (if available) ────────────────────────────────
        wfv_trades_df = st.session_state.get("wfv_trades")
        if wfv_trades_df is not None and len(wfv_trades_df) > 0:
            st.subheader("All WFV Trades — Cumulative PnL per Fold")
            cum_pnl = wfv_trades_df.sort_values("entry_ts").copy()
            cum_pnl["cum_pnl"] = cum_pnl.groupby("fold")["pnl_usd"].cumsum()
            fig_cum = px.line(cum_pnl, x=cum_pnl.index,
                              y="cum_pnl", color="fold",
                              template="plotly_dark", height=350,
                              labels={"cum_pnl": "Cumulative PnL ($)"})
            fig_cum.update_layout(margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_cum, use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────
# TAB 5 — Model Insights
# ────────────────────────────────────────────────────────────────────────────
with tab5:
    st.header("Model Insights")

    # ── Feature list ─────────────────────────────────────────────────────────
    st.subheader("Feature Engineering (16 Features)")
    feat_meta = {
        "mss_signal":          ("Price Action", "Market Structure Shift: −1/0/+1"),
        "fvg_type":            ("Price Action", "Fair Value Gap: −1/0/+1"),
        "fvg_distance_atr":    ("Price Action", "FVG size relative to ATR(14)"),
        "fvg_age_candles":     ("Price Action", "Bars since FVG formed"),
        "sweep_occurred":      ("Price Action", "Liquidity sweep flag 0/1"),
        "htf_bias":            ("Price Action", "1D trend bias: −1/0/+1"),
        "relative_volume":     ("Confirmation", "20-bar relative volume ratio"),
        "rsi_14":              ("Confirmation", "RSI(14) value"),
        "rsi_slope_3":         ("Confirmation", "3-bar RSI slope"),
        "ema21_55_alignment":  ("Confirmation", "EMA21 vs EMA55 alignment"),
        "atr_percentile_100":  ("Confirmation", "ATR 100-bar percentile rank"),
        "volatility_rank":     ("Market Char.", "Volatility percentile vs universe"),
        "volume_rank":         ("Market Char.", "Volume percentile vs universe"),
        "asset_tier_flag":     ("Market Char.", "High-vol sub-tier flag 0/1"),
        "hour_of_day":         ("Market Char.", "UTC hour (0–23)"),
        "day_of_week":         ("Market Char.", "Day of week (0–6)"),
    }
    feat_df = pd.DataFrame([
        {"Feature": k, "Category": v[0], "Description": v[1]}
        for k, v in feat_meta.items()
    ])
    st.dataframe(feat_df.set_index("Feature"), use_container_width=True, height=400)

    # ── Feature importance from session model ─────────────────────────────────
    st.subheader("Feature Importance")
    bt_trades = st.session_state.get("bt_trades")
    if bt_trades is not None and len(bt_trades) > 0:
        # Rebuild model to extract feature importance
        st.info("Feature importance requires re-training the model. Run demo backtest first.")
    else:
        st.info("Run a demo backtest (Backtest Lab tab) to see feature importance.")

    # ── Confidence tier analysis ───────────────────────────────────────────────
    st.divider()
    st.subheader("Confidence Tier Breakdown")

    trades_any = st.session_state.get("bt_trades", load_trade_log())
    if trades_any is not None and "confidence" in trades_any.columns:
        def _tier(c):
            if c >= 0.95: return "⚡ Power Setup (5×)"
            if c >= 0.92: return "High Conviction (3×)"
            if c >= 0.85: return "Standard (2×)"
            return "Base (1×)"

        trades_any = trades_any.copy()
        trades_any["tier"]   = trades_any["confidence"].apply(_tier)
        trades_any["is_win"] = trades_any["pnl_usd"] > 0

        tier_stats = (
            trades_any.groupby("tier")
            .agg(count=("pnl_usd","count"),
                 total_pnl=("pnl_usd","sum"),
                 win_rate=("is_win","mean"),
                 avg_pnl=("pnl_usd","mean"))
            .reset_index()
        )
        tier_stats["win_rate"] = (tier_stats["win_rate"] * 100).round(1)
        tier_stats["total_pnl"] = tier_stats["total_pnl"].round(2)
        tier_stats["avg_pnl"]   = tier_stats["avg_pnl"].round(2)

        st.dataframe(tier_stats.set_index("tier"), use_container_width=True)

        fig_tier = px.bar(tier_stats, x="tier", y="total_pnl", color="win_rate",
                          color_continuous_scale="RdYlGn",
                          template="plotly_dark", height=300,
                          labels={"total_pnl": "Total PnL ($)", "win_rate": "Win Rate %"})
        fig_tier.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_tier, use_container_width=True)

        # Confidence histogram
        fig_conf = px.histogram(trades_any, x="confidence", nbins=40,
                                color="outcome",
                                color_discrete_map={"tp":"#06d6a0","sl":"#ef476f","time":"#ffd166"},
                                template="plotly_dark", height=300,
                                labels={"confidence": "Model Confidence"})
        for thresh, label, col in [(0.95, "Power Setup", "yellow"),
                                   (0.92, "High Conv.", "orange"),
                                   (0.85, "Standard",   "lightblue")]:
            fig_conf.add_vline(x=thresh, line_dash="dash", line_color=col,
                                annotation_text=label)
        fig_conf.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.subheader("Confidence Distribution by Outcome")
        st.plotly_chart(fig_conf, use_container_width=True)
    else:
        st.info("Run a demo backtest to see confidence tier analysis.")

    # ── Leverage tier simulator ────────────────────────────────────────────────
    st.divider()
    st.subheader("Position Sizing Simulator")
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        sim_cap  = st.number_input("Capital ($)", value=5000.0, step=500.0, key="sim_cap")
    with sc2:
        sim_conf = st.slider("Confidence", 0.75, 1.00, 0.90, step=0.01, key="sim_conf")
    with sc3:
        sim_asset = st.selectbox("Asset", TIER2_UNIVERSE, key="sim_asset")

    lev      = get_leverage(sim_conf)
    ps       = is_power_setup(sim_conf)
    hv       = sim_asset in HIGH_VOL_SUBTIER
    base     = sim_cap / 4
    ps_mult  = 1.25 if ps else 1.0
    hv_mult  = 0.75 if hv else 1.0
    pos_usd  = base * ps_mult * hv_mult * lev
    port_lev = pos_usd / sim_cap

    rc1, rc2, rc3, rc4, rc5 = st.columns(5)
    rc1.metric("Tier",          _confidence_tier(sim_conf))
    rc2.metric("Leverage",      f"{lev:.0f}×")
    rc3.metric("PS Scalar",     f"{ps_mult:.2f}×")
    rc4.metric("HV Scalar",     f"{hv_mult:.2f}×")
    rc5.metric("Position USD",  f"${pos_usd:,.2f}")
    st.metric("Contribution to Portfolio Leverage", f"{port_lev:.3f}×  (max 3.5×)")


# ────────────────────────────────────────────────────────────────────────────
# TAB 6 — HPO Explorer
# ────────────────────────────────────────────────────────────────────────────
with tab6:
    st.header("HPO Explorer — Optuna")

    # ── Search space table ────────────────────────────────────────────────────
    st.subheader("V5 Search Space")
    ss_rows = []
    for name, spec in V5_SEARCH_SPACE.items():
        ss_rows.append({
            "Parameter": name,
            "Type":      spec["type"],
            "Low":       spec["low"],
            "High":      spec["high"],
            "Log Scale": spec.get("log", False),
        })
    ss_rows.append({
        "Parameter": "max_holding_candles",
        "Type":      "LOCKED",
        "Low":       31,
        "High":      31,
        "Log Scale": False,
    })
    ss_df = pd.DataFrame(ss_rows)
    st.dataframe(ss_df.set_index("Parameter"), use_container_width=True)

    st.divider()

    # ── Optuna DB exploration ─────────────────────────────────────────────────
    if OPTUNA_DB.exists():
        st.subheader("Optuna Study — Trial History")
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            study = optuna.load_study(
                study_name=None,
                storage=f"sqlite:///{OPTUNA_DB}",
            )
            trials_df = study.trials_dataframe()

            # Filter to completed trials
            done = trials_df[trials_df["state"] == "COMPLETE"].copy()
            st.caption(f"Loaded {len(trials_df)} trials ({len(done)} complete) from {OPTUNA_DB.name}")

            hc1, hc2, hc3, hc4 = st.columns(4)
            hc1.metric("Total Trials",    len(trials_df))
            hc2.metric("Completed",       len(done))
            hc3.metric("Best Score",      f"{study.best_value:.4f}" if study.best_value is not None else "—")
            hc4.metric("Best Trial #",    str(study.best_trial.number) if study.best_trial else "—")

            # Optimization history
            if not done.empty and "value" in done.columns:
                done = done.sort_values("number")
                done["best_so_far"] = done["value"].cummax()
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Scatter(x=done["number"], y=done["value"],
                                              mode="markers", name="Trial Score",
                                              marker=dict(size=4, color="#48cae4", opacity=0.5)))
                fig_hist.add_trace(go.Scatter(x=done["number"], y=done["best_so_far"],
                                              mode="lines", name="Best So Far",
                                              line=dict(color="#f77f00", width=2)))
                fig_hist.update_layout(template="plotly_dark", height=350,
                                       title="Optimization Progress",
                                       xaxis_title="Trial #", yaxis_title="Score",
                                       margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig_hist, use_container_width=True)

            # Parameter importance / distribution
            param_cols = [c for c in done.columns if c.startswith("params_")]
            if param_cols and "value" in done.columns:
                st.subheader("Parameter vs Score Scatter")
                sel_param = st.selectbox(
                    "Parameter",
                    [c.replace("params_", "") for c in param_cols],
                    key="hpo_param",
                )
                full_col = f"params_{sel_param}"
                if full_col in done.columns:
                    fig_scatter = px.scatter(
                        done[done["value"] > -900],   # exclude pruned/failed
                        x=full_col, y="value",
                        color="value", color_continuous_scale="Plasma",
                        template="plotly_dark", height=350,
                        labels={full_col: sel_param, "value": "HPO Score"},
                    )
                    # Mark best trial
                    best_row = done.loc[done["value"].idxmax()]
                    fig_scatter.add_trace(go.Scatter(
                        x=[best_row[full_col]], y=[best_row["value"]],
                        mode="markers", marker=dict(size=14, symbol="star",
                                                    color="gold", line=dict(width=1, color="black")),
                        name=f"Best trial #{int(best_row['number'])}",
                    ))
                    fig_scatter.update_layout(margin=dict(l=0, r=0, t=10, b=0))
                    st.plotly_chart(fig_scatter, use_container_width=True)

                # Parallel coordinates for top trials
                st.subheader("Parallel Coordinates — Top 50 Trials")
                top50 = done[done["value"] > -900].nlargest(50, "value")
                if len(top50) >= 5 and param_cols:
                    plot_params = param_cols[:8]   # max 8 axes
                    dims = [dict(label=c.replace("params_", ""),
                                 values=top50[c]) for c in plot_params if c in top50.columns]
                    dims.append(dict(label="Score", values=top50["value"]))
                    fig_par = go.Figure(go.Parcoords(
                        line=dict(color=top50["value"], colorscale="Plasma",
                                  showscale=True, colorbar=dict(title="Score")),
                        dimensions=dims,
                    ))
                    fig_par.update_layout(template="plotly_dark", height=400,
                                          margin=dict(l=100, r=40, t=40, b=40))
                    st.plotly_chart(fig_par, use_container_width=True)

            # Best trial params
            if study.best_trial:
                st.subheader("Best Trial Parameters")
                bp = study.best_params
                bp["max_holding_candles"] = V5_TIME_EXIT_BARS   # Always locked
                bp_df = pd.DataFrame([{"Parameter": k, "Value": v} for k, v in bp.items()])
                st.dataframe(bp_df.set_index("Parameter"), use_container_width=True)

        except Exception as e:
            st.error(f"Could not load Optuna DB: {e}")
    else:
        st.info(f"Optuna DB not found at `{OPTUNA_DB}`. Run HPO first.")
        st.markdown("""
Run optimization (creates the DB):
```bash
cd /home/yagokhan
PYTHONPATH=/home/yagokhan python3 varanus/run_optimization.py --trials 50
```
Or for quick demo (10 trials, skip WFV):
```bash
PYTHONPATH=/home/yagokhan python3 varanus/run_optimization.py --trials 10 --skip-wfv
```
""")

    # ── Best params vs search space comparison ────────────────────────────────
    st.divider()
    st.subheader("Best Params vs Search Space (position within range)")
    params = load_best_params()
    comp_rows = []
    for name, spec in V5_SEARCH_SPACE.items():
        val = params.get(name)
        if val is not None:
            lo, hi = spec["low"], spec["high"]
            pct = (val - lo) / (hi - lo) * 100 if hi != lo else 50
            comp_rows.append({"Parameter": name, "Low": lo, "High": hi,
                               "Best": val, "Position%": round(pct, 1)})
    if comp_rows:
        cmp_df = pd.DataFrame(comp_rows)
        fig_cmp = px.bar(cmp_df, x="Parameter", y="Position%",
                         template="plotly_dark", height=350,
                         title="Best param as % of search range (0% = low end, 100% = high end)",
                         color="Position%", color_continuous_scale="RdYlGn_r")
        fig_cmp.add_hline(y=50, line_dash="dash", line_color="white",
                           annotation_text="Midpoint")
        fig_cmp.update_layout(margin=dict(l=0, r=0, t=50, b=0))
        st.plotly_chart(fig_cmp, use_container_width=True)
        st.dataframe(cmp_df.set_index("Parameter"), use_container_width=True)
    else:
        st.info("Run HPO to populate best params.")
