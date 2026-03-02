"""
save_results_v5.py — Run quick HPO → 8-Fold WFV → Generate premium plots + Excel.

Usage:
    cd /home/yagokhan
    PYTHONPATH=/home/yagokhan python varanus/save_results_v5.py [--trials 20] [--skip-hpo]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import logging
from pathlib import Path

import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import seaborn as sns

# ── Ensure project root on sys.path ──────────────────────────────────────────
ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from varanus.universe     import TIER2_UNIVERSE
from varanus.optimizer    import run_optimization, extract_best_params
from varanus.walk_forward import run_walk_forward, WFV_CONFIG
from varanus.backtest     import compute_metrics

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("varanus.save_results_v5")

# ── Paths ────────────────────────────────────────────────────────────────────
CACHE       = "/home/yagokhan/chameleon/claude_code_project/data/cache"
PARAMS_FILE = str(Path(__file__).parent / "config" / "best_params_v5.json")
PLOTS_DIR   = str(Path(__file__).parent / "plots")
RESULTS_DIR = str(Path(__file__).parent / "results")

os.makedirs(PLOTS_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Colour palette ───────────────────────────────────────────────────────────
plt.style.use("dark_background")
TEAL    = "#00ffcc"
RED     = "#ff4444"
ORANGE  = "#ffaa00"
GREEN   = "#44ff88"
GREY    = "#888888"
PURPLE  = "#b388ff"
CYAN    = "#00e5ff"
GOLD    = "#ffd740"


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading (mirrors run_optimization.py)
# ═══════════════════════════════════════════════════════════════════════════════

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
        agg = {}
        for col, fn in [("open","first"),("high","max"),("low","min"),
                         ("close","last"),("volume","sum")]:
            if col in df.columns:
                agg[col] = fn
        df = df.resample("1D").agg(agg).dropna()

    return df


def load_universe() -> tuple[dict, dict]:
    data_4h, data_1d = {}, {}
    for asset in TIER2_UNIVERSE:
        try:
            df_4h = load_data(asset, "4h")
            df_1d = load_data(asset, "1d")
            df_1d = df_1d[df_1d.index >= df_4h.index[0] - pd.Timedelta(days=100)]
            data_4h[asset] = df_4h
            data_1d[asset] = df_1d
            logger.info("Loaded %s: %d × 4h | %d × 1d", asset, len(df_4h), len(df_1d))
        except Exception as exc:
            logger.warning("Skipping %s — %s", asset, exc)
    logger.info("Universe: %d / %d assets loaded", len(data_4h), len(TIER2_UNIVERSE))
    return data_4h, data_1d


# ═══════════════════════════════════════════════════════════════════════════════
# Formatters
# ═══════════════════════════════════════════════════════════════════════════════

def _usd(x, _):
    if abs(x) >= 1_000:
        return f"${x/1_000:.1f}k"
    return f"${x:.0f}"

def _pct(x, _):
    return f"{x:.0f}%"


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 1: Performance Dashboard (4-panel)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_performance_dashboard(trades: pd.DataFrame, metrics: dict):
    fig = plt.figure(figsize=(22, 16))
    fig.patch.set_facecolor("#0d0d0d")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)

    calmar = metrics.get("calmar_ratio", 0)
    sharpe = metrics.get("sharpe_ratio", 0)
    wr     = metrics.get("win_rate_pct", 0)
    mdd    = metrics.get("max_drawdown_pct", 0)
    cagr   = metrics.get("cagr_pct", 0)
    pf     = metrics.get("profit_factor", 0)
    n      = metrics.get("total_trades", 0)

    fig.suptitle(
        f"Varanus v5.0 Tier 2 — 8-Fold Walk-Forward Backtest\n"
        f"CAGR {cagr:.1f}%  |  PF {pf:.2f}  |  Calmar {calmar:.2f}  |  Sharpe {sharpe:.2f}  |  "
        f"WR {wr:.1f}%  |  MaxDD {mdd:.1f}%  |  {n} trades",
        fontsize=15, color="white", y=0.98
    )

    # ── 1. Equity curve ──────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    initial_cap = 5_000.0
    tr = trades.sort_values("exit_ts").copy()
    tr["cumulative_pnl"] = tr["pnl_usd"].cumsum() + initial_cap
    ax1.plot(tr["exit_ts"], tr["cumulative_pnl"], color=TEAL, lw=2)
    ax1.fill_between(tr["exit_ts"], initial_cap, tr["cumulative_pnl"],
                     where=tr["cumulative_pnl"] >= initial_cap,
                     alpha=0.15, color=GREEN)
    ax1.fill_between(tr["exit_ts"], initial_cap, tr["cumulative_pnl"],
                     where=tr["cumulative_pnl"] < initial_cap,
                     alpha=0.20, color=RED)
    ax1.axhline(initial_cap, color=GREY, ls="--", lw=0.8, alpha=0.6)
    ax1.yaxis.set_major_formatter(FuncFormatter(_usd))
    ax1.set_title("Equity Curve", color="white", fontsize=13)
    ax1.set_xlabel("Date", color=GREY, fontsize=9)
    ax1.set_ylabel("Portfolio Value ($)", color=GREY, fontsize=9)
    ax1.tick_params(colors=GREY, labelsize=8)
    for sp in ax1.spines.values(): sp.set_color("#333")

    # ── 2. Drawdown ──────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    equity_s = tr.set_index("exit_ts")["cumulative_pnl"]
    roll_max = equity_s.cummax()
    drawdown = (equity_s - roll_max) / roll_max * 100
    ax2.fill_between(drawdown.index, drawdown.values, 0, color=RED, alpha=0.5)
    ax2.plot(drawdown.index, drawdown.values, color=RED, lw=1)
    ax2.axhline(mdd, color=ORANGE, ls=":", lw=1, label=f"MaxDD {mdd:.1f}%")
    ax2.axhline(-15, color=PURPLE, ls="--", lw=1, alpha=0.7, label="v5 Hard Cap -15%")
    ax2.yaxis.set_major_formatter(FuncFormatter(_pct))
    ax2.set_title("Drawdown", color="white", fontsize=13)
    ax2.set_xlabel("Date", color=GREY, fontsize=9)
    ax2.set_ylabel("Drawdown (%)", color=GREY, fontsize=9)
    ax2.legend(fontsize=8, facecolor="#1a1a1a", edgecolor=GREY)
    ax2.tick_params(colors=GREY, labelsize=8)
    for sp in ax2.spines.values(): sp.set_color("#333")

    # ── 3. PnL by asset ─────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    asset_pnl = trades.groupby("asset")["pnl_usd"].sum().sort_values()
    colors = [GREEN if v >= 0 else RED for v in asset_pnl.values]
    bars = ax3.barh(asset_pnl.index, asset_pnl.values, color=colors,
                    edgecolor="#222", height=0.6)
    ax3.axvline(0, color=GREY, lw=0.8)
    for bar in bars:
        w = bar.get_width()
        ax3.text(w + (5 if w >= 0 else -5), bar.get_y() + bar.get_height()/2,
                 f"${w:,.0f}", va="center", ha="left" if w >= 0 else "right",
                 color="white", fontsize=7)
    ax3.xaxis.set_major_formatter(FuncFormatter(_usd))
    ax3.set_title("Total PnL by Asset", color="white", fontsize=13)
    ax3.tick_params(colors=GREY, labelsize=8)
    for sp in ax3.spines.values(): sp.set_color("#333")

    # ── 4. Outcome distribution ──────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    outcome_counts = trades["outcome"].value_counts()
    pie_colors = {"tp": GREEN, "sl": RED, "time": ORANGE}
    wedge_colors = [pie_colors.get(k, GREY) for k in outcome_counts.index]
    wedges, texts, autotexts = ax4.pie(
        outcome_counts, labels=[o.upper() for o in outcome_counts.index],
        colors=wedge_colors, autopct="%1.1f%%", startangle=90,
        wedgeprops=dict(width=0.45, edgecolor="#0d0d0d")
    )
    plt.setp(texts, color="white", size=11)
    plt.setp(autotexts, color="white", size=10, weight="bold")
    ax4.set_title("Exit Outcome Distribution", color="white", fontsize=13)

    path = os.path.join(PLOTS_DIR, "v5_01_performance_dashboard.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 2: Fold Comparison
# ═══════════════════════════════════════════════════════════════════════════════

def plot_fold_comparison(results_df: pd.DataFrame):
    if results_df.empty:
        return
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    fig.patch.set_facecolor("#0d0d0d")
    fig.suptitle("V5 8-Fold Walk-Forward Comparison", color="white", fontsize=14)

    metrics_to_plot = [
        ("profit_factor",    "Profit Factor",   TEAL),
        ("calmar_ratio",     "Calmar Ratio",    CYAN),
        ("win_rate_pct",     "Win Rate (%)",    GREEN),
        ("max_drawdown_pct", "Max Drawdown (%)", RED),
    ]
    folds = results_df["fold"].astype(str)

    for ax, (col, title, color) in zip(axes, metrics_to_plot):
        if col not in results_df.columns:
            continue
        vals = results_df[col]
        ax.bar(folds, vals, color=color, edgecolor="#222", width=0.5, alpha=0.85)
        ax.set_title(title, color="white", fontsize=12)
        ax.set_xlabel("Fold", color=GREY, fontsize=9)
        ax.tick_params(colors=GREY, labelsize=9)
        ax.set_facecolor("#111")
        for sp in ax.spines.values(): sp.set_color("#333")

        # Pass/fail dot above each bar
        if "fold_pass" in results_df.columns:
            for i, (v, passed) in enumerate(zip(vals, results_df["fold_pass"])):
                marker = "✓" if passed else "✗"
                mc     = GREEN if passed else RED
                offset = abs(vals.max() - vals.min()) * 0.06 if vals.max() != vals.min() else 1.0
                ax.text(i, v + offset, marker, ha="center", color=mc, fontsize=10, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "v5_02_fold_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 3: Confidence Analysis (with Power Setup highlight)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_confidence_analysis(trades: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.patch.set_facecolor("#0d0d0d")
    fig.suptitle("V5 Confidence Analysis — Power Setup (≥0.95) Highlighted",
                 color="white", fontsize=14)

    outcome_colors = {"tp": GREEN, "sl": RED, "time": ORANGE}

    # ── Scatter: Confidence vs PnL ───────────────────────────────────────────
    for outcome in trades["outcome"].unique():
        sub = trades[trades["outcome"] == outcome]
        ax1.scatter(sub["confidence"], sub["pnl_usd"],
                    color=outcome_colors.get(outcome, GREY),
                    alpha=0.5, s=40, label=outcome.upper(), edgecolors="none")

    # Highlight Power Setup tier
    if "confidence" in trades.columns:
        ax1.axvspan(0.95, 1.0, alpha=0.12, color=GOLD, label="Power Setup (3x)")

    ax1.axhline(0, color=GREY, ls="--", lw=0.8)
    ax1.set_title("Confidence vs PnL per Trade", color="white")
    ax1.set_xlabel("Model Confidence", color=GREY)
    ax1.set_ylabel("PnL ($)", color=GREY)
    ax1.legend(facecolor="#1a1a1a", edgecolor=GREY, labelcolor="white", fontsize=9)
    ax1.yaxis.set_major_formatter(FuncFormatter(_usd))
    ax1.tick_params(colors=GREY)
    ax1.set_facecolor("#111")
    for sp in ax1.spines.values(): sp.set_color("#333")

    # ── Histogram: Confidence distribution ───────────────────────────────────
    for outcome in trades["outcome"].unique():
        sub = trades[trades["outcome"] == outcome]
        ax2.hist(sub["confidence"], bins=25, alpha=0.55,
                 color=outcome_colors.get(outcome, GREY),
                 label=outcome.upper(), edgecolor="none")

    if "confidence" in trades.columns:
        ax2.axvspan(0.95, 1.0, alpha=0.12, color=GOLD)

    # Draw leverage tier boundaries
    for thresh, lbl in [(0.75, "1x"), (0.85, "2x"), (0.92, "3x"), (0.95, "3x★")]:
        ax2.axvline(thresh, color=GREY, ls=":", lw=0.7, alpha=0.5)
        ax2.text(thresh + 0.002, ax2.get_ylim()[1] * 0.95 if ax2.get_ylim()[1] > 0 else 5,
                 lbl, color=GREY, fontsize=8, va="top")

    ax2.set_title("Confidence Score Distribution", color="white")
    ax2.set_xlabel("Model Confidence", color=GREY)
    ax2.set_ylabel("Count", color=GREY)
    ax2.legend(facecolor="#1a1a1a", edgecolor=GREY, labelcolor="white", fontsize=9)
    ax2.tick_params(colors=GREY)
    ax2.set_facecolor("#111")
    for sp in ax2.spines.values(): sp.set_color("#333")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "v5_03_confidence_analysis.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 4: Monthly PnL Heatmap
# ═══════════════════════════════════════════════════════════════════════════════

def plot_monthly_heatmap(trades: pd.DataFrame):
    tr = trades.copy()
    tr["exit_ts"] = pd.to_datetime(tr["exit_ts"], utc=True)
    tr["year"]  = tr["exit_ts"].dt.year
    tr["month"] = tr["exit_ts"].dt.month

    monthly = tr.groupby(["year", "month"])["pnl_usd"].sum().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(14, max(3, len(monthly) * 1.5)))
    fig.patch.set_facecolor("#0d0d0d")

    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]
    col_labels = [month_labels[m-1] for m in monthly.columns]

    sns.heatmap(monthly, annot=True, fmt=".0f", cmap="RdYlGn", center=0,
                linewidths=0.5, linecolor="#222",
                xticklabels=col_labels,
                cbar_kws={"label": "PnL ($)", "shrink": 0.6},
                ax=ax)

    ax.set_title("Monthly PnL Heatmap ($)", color="white", fontsize=13)
    ax.set_xlabel("Month", color=GREY)
    ax.set_ylabel("Year", color=GREY)
    ax.tick_params(colors=GREY)

    path = os.path.join(PLOTS_DIR, "v5_04_monthly_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 5: Asset × Confidence Tier Heatmap (v5 — 4 tiers including Power Setup)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_asset_confidence_heatmap(trades: pd.DataFrame):
    tr = trades.copy()
    tr["conf_bin"] = pd.cut(
        tr["confidence"],
        bins=[0.74, 0.85, 0.92, 0.95, 1.01],
        labels=["Base (1x)", "Standard (2x)", "High (3x)", "Power (3x★)"]
    )

    pivot = tr.pivot_table(
        values="pnl_usd", index="asset", columns="conf_bin",
        aggfunc="sum", fill_value=0
    )

    fig, ax = plt.subplots(figsize=(13, max(5, len(pivot) * 0.55)))
    fig.patch.set_facecolor("#0d0d0d")

    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="RdYlGn", center=0,
                linewidths=0.5, linecolor="#222",
                cbar_kws={"label": "PnL ($)", "shrink": 0.7},
                ax=ax)

    ax.set_title("PnL Heatmap: Asset × Leverage Tier (v5 Power Setup)",
                 color="white", fontsize=13)
    ax.set_xlabel("Confidence Tier (Leverage)", color=GREY)
    ax.set_ylabel("Asset", color=GREY)
    ax.tick_params(colors=GREY)

    path = os.path.join(PLOTS_DIR, "v5_05_asset_confidence_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 6: Duration vs PnL scatter
# ═══════════════════════════════════════════════════════════════════════════════

def plot_duration_scatter(trades: pd.DataFrame):
    tr = trades.copy()
    tr["entry_ts"]   = pd.to_datetime(tr["entry_ts"], utc=True)
    tr["exit_ts"]    = pd.to_datetime(tr["exit_ts"],  utc=True)
    tr["duration_h"] = (tr["exit_ts"] - tr["entry_ts"]).dt.total_seconds() / 3600

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("#0d0d0d")

    outcome_colors = {"tp": GREEN, "sl": RED, "time": ORANGE}
    for outcome in tr["outcome"].unique():
        sub = tr[tr["outcome"] == outcome]
        ax.scatter(sub["duration_h"], sub["pnl_usd"],
                   color=outcome_colors.get(outcome, GREY),
                   alpha=0.55, s=50, label=outcome.upper(), edgecolors="none")

    ax.axhline(0, color=GREY, ls="--", lw=0.8)
    # 31-bar time exit = 124 hours
    ax.axvline(124, color=ORANGE, ls=":", lw=1, alpha=0.6, label="31-bar time exit (124h)")
    ax.yaxis.set_major_formatter(FuncFormatter(_usd))
    ax.set_title("Trade Duration vs PnL (v5 — 31-bar time exit lock)",
                 color="white", fontsize=13)
    ax.set_xlabel("Duration (hours)", color=GREY)
    ax.set_ylabel("PnL ($)", color=GREY)
    ax.legend(facecolor="#1a1a1a", edgecolor=GREY, labelcolor="white")
    ax.tick_params(colors=GREY)
    ax.set_facecolor("#111")
    for sp in ax.spines.values(): sp.set_color("#333")

    path = os.path.join(PLOTS_DIR, "v5_06_duration_vs_pnl.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Excel export
# ═══════════════════════════════════════════════════════════════════════════════

def save_excel(trades: pd.DataFrame, results_df: pd.DataFrame,
               metrics: dict, params: dict):
    path = os.path.join(RESULTS_DIR, "v5_backtest_results.xlsx")

    with pd.ExcelWriter(path, engine="openpyxl") as writer:

        # Sheet 1: Summary
        summary_rows = [
            ("VARANUS v5.0 PERFORMANCE", ""),
            ("", ""),
            ("Total Return (%)",     metrics.get("total_return_pct", 0)),
            ("CAGR (%)",             metrics.get("cagr_pct", 0)),
            ("Max Drawdown (%)",     metrics.get("max_drawdown_pct", 0)),
            ("Calmar Ratio",         metrics.get("calmar_ratio", 0)),
            ("Sharpe Ratio",         metrics.get("sharpe_ratio", 0)),
            ("Win Rate (%)",         metrics.get("win_rate_pct", 0)),
            ("Profit Factor",        metrics.get("profit_factor", 0)),
            ("Total Trades",         metrics.get("total_trades", 0)),
            ("TP Hits",              metrics.get("tp_hits", 0)),
            ("SL Hits",              metrics.get("sl_hits", 0)),
            ("Time Exits",           metrics.get("time_exits", 0)),
            ("Avg Win ($)",          metrics.get("avg_win_usd", 0)),
            ("Avg Loss ($)",         metrics.get("avg_loss_usd", 0)),
            ("", ""),
            ("OPTIMISED PARAMETERS", ""),
        ]
        for k, v in params.items():
            if k.startswith("_"):
                continue
            summary_rows.append((k, round(v, 6) if isinstance(v, float) else v))

        summary_df = pd.DataFrame(summary_rows, columns=["Metric", "Value"])
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

        # Sheet 2: Trade Log
        trades_out = trades.copy()
        for col in ["entry_ts", "exit_ts", "max_hold_bar"]:
            if col in trades_out.columns:
                trades_out[col] = pd.to_datetime(trades_out[col]).dt.tz_localize(None)
        trades_out.to_excel(writer, sheet_name="Trade Log", index=False)

        # Sheet 3: Asset Stats
        asset_stats = trades.groupby("asset").agg(
            trades    = ("pnl_usd", "count"),
            total_pnl = ("pnl_usd", "sum"),
            avg_pnl   = ("pnl_usd", "mean"),
            win_rate  = ("pnl_usd", lambda x: (x > 0).mean() * 100),
            avg_conf  = ("confidence", "mean"),
            tp_hits   = ("outcome", lambda x: (x == "tp").sum()),
            sl_hits   = ("outcome", lambda x: (x == "sl").sum()),
            time_exits= ("outcome", lambda x: (x == "time").sum()),
        ).round(3).reset_index()
        asset_stats.to_excel(writer, sheet_name="Asset Stats", index=False)

        # Sheet 4: Fold Results
        if not results_df.empty:
            results_df.to_excel(writer, sheet_name="Fold Results", index=False)

        # Sheet 5: Monthly PnL
        trades_m = trades.copy()
        trades_m["exit_ts"] = pd.to_datetime(trades_m["exit_ts"], utc=True)
        trades_m["year"]  = trades_m["exit_ts"].dt.year
        trades_m["month"] = trades_m["exit_ts"].dt.strftime("%b")
        trades_m["month_num"] = trades_m["exit_ts"].dt.month
        monthly = (trades_m.groupby(["year", "month_num", "month"])["pnl_usd"]
                   .sum().reset_index()
                   .sort_values(["year", "month_num"])
                   .drop("month_num", axis=1))
        monthly.columns = ["Year", "Month", "PnL ($)"]
        monthly.to_excel(writer, sheet_name="Monthly PnL", index=False)

    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(description="Varanus v5 — Run HPO + WFV + Generate Plots")
    p.add_argument("--trials", type=int, default=20,
                   help="Number of Optuna HPO trials (default: 20)")
    p.add_argument("--skip-hpo", action="store_true",
                   help="Skip HPO — use existing best_params_v5.json")
    return p.parse_args()


def main():
    args = _parse_args()

    print("\n" + "═" * 70)
    print("  VARANUS v5.0 — Backtest Results Generator")
    print("  Quick HPO → 8-Fold WFV → Premium Plots + Excel")
    print("═" * 70)

    # ── 1. Load data ─────────────────────────────────────────────────────────
    logger.info("Loading universe data ...")
    data_4h, data_1d = load_universe()

    if not data_4h:
        logger.error("No assets loaded. Check CACHE path: %s", CACHE)
        sys.exit(1)

    # ── 2. HPO or load params ────────────────────────────────────────────────
    if args.skip_hpo:
        logger.info("Loading existing params from %s", PARAMS_FILE)
        with open(PARAMS_FILE) as f:
            best_params = json.load(f)
        # Check if params are populated
        if best_params.get("confidence_thresh") is None:
            logger.error("best_params_v5.json has null params. Run without --skip-hpo first.")
            sys.exit(1)
    else:
        logger.info("Starting quick Optuna HPO — %d trials ...", args.trials)
        study = run_optimization(
            df_dict_4h = data_4h,
            df_dict_1d = data_1d,
            n_trials   = args.trials,
        )
        best_params = extract_best_params(study)
        logger.info("HPO complete — best trial #%d, score %.4f",
                    study.best_trial.number, study.best_value)

    # Remove meta keys for display
    display_params = {k: v for k, v in best_params.items() if not k.startswith("_")}
    print(f"\n  Best Params:\n{json.dumps(display_params, indent=4, default=str)}")

    # ── 3. Full 8-fold Walk-Forward Validation ───────────────────────────────
    logger.info("Running full 8-fold Walk-Forward Validation ...")
    results_df, consistency, all_trades = run_walk_forward(
        df_dict_4h = data_4h,
        df_dict_1d = data_1d,
        params     = best_params,
    )

    if all_trades.empty:
        logger.error("No trades generated. Aborting.")
        sys.exit(1)

    print(f"\n  Total trades across all folds: {len(all_trades)}")
    print(f"  Consistency: {consistency:.0%}")

    # ── 4. Aggregate metrics ─────────────────────────────────────────────────
    initial_cap = 5_000.0
    all_trades_sorted = all_trades.sort_values("exit_ts")
    equity = all_trades_sorted["pnl_usd"].cumsum() + initial_cap
    equity.index = pd.to_datetime(all_trades_sorted["exit_ts"].values)
    metrics = compute_metrics(equity, all_trades_sorted)

    print(f"\n  Aggregate Metrics:")
    for k, v in metrics.items():
        print(f"    {k:25s}: {v}")

    # Power Setup stats
    ps_trades = all_trades[all_trades.get("confidence", pd.Series()) >= 0.95] \
                if "confidence" in all_trades.columns else pd.DataFrame()
    print(f"\n  Power Setup (≥0.95) trades: {len(ps_trades)}")

    # ── 5. Save trade log CSV ────────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, "v5_trade_log.csv")
    all_trades.to_csv(csv_path, index=False)
    print(f"\n  Saved trade log: {csv_path}")

    # Save fold results CSV
    folds_csv = os.path.join(RESULTS_DIR, "v5_fold_results.csv")
    results_df.to_csv(folds_csv, index=False)
    print(f"  Saved fold results: {folds_csv}")

    # ── 6. Generate plots ────────────────────────────────────────────────────
    print("\n[+] Generating plots ...")
    plot_performance_dashboard(all_trades, metrics)
    plot_fold_comparison(results_df)
    plot_confidence_analysis(all_trades)
    plot_monthly_heatmap(all_trades)
    plot_asset_confidence_heatmap(all_trades)
    plot_duration_scatter(all_trades)

    # ── 7. Save Excel ────────────────────────────────────────────────────────
    print("\n[+] Saving Excel workbook ...")
    save_excel(all_trades, results_df, metrics, best_params)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  VARANUS v5.0 — RESULTS COMPLETE")
    print(f"  Plots   → {PLOTS_DIR}/")
    print(f"  Excel   → {RESULTS_DIR}/v5_backtest_results.xlsx")
    print(f"  Trades  → {RESULTS_DIR}/v5_trade_log.csv")
    print(f"  Folds   → {RESULTS_DIR}/v5_fold_results.csv")
    wfv_pass = consistency >= 0.75
    print(f"  WFV     → {'✅ PASS' if wfv_pass else '❌ FAIL'} ({consistency:.0%})")
    print("═" * 70)


if __name__ == "__main__":
    main()
