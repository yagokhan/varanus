import requests

# ── Entry alert ───────────────────────────────────────────────────────────────
ALERT_FORMAT = (
    "🦎 *VARANUS T2* | {asset} {direction} @ {confidence:.0%}\n"
    "Entry: {entry_price} | TP: {take_profit} | SL: {stop_loss}\n"
    "R:R {rr_ratio:.1f}x | Lev: {leverage}x | ATR: {atr_14:.4f}\n"
    "MSS: {mss} | FVG✓ | Sweep✓ | RVol: {rvol:.2f}x | RSI: {rsi:.1f}\n"
    "HTF: {htf_bias} | Pos: ${position_usd:.0f} | Port Lev: {port_lev:.2f}x"
)

REQUIRED_FIELDS = [
    "timestamp_utc", "asset", "direction", "confidence", "leverage",
    "entry_price", "take_profit", "stop_loss", "rr_ratio", "atr_14",
    "mss", "fvg_valid", "sweep_confirmed", "rvol", "rsi", "htf_bias",
    "position_usd", "port_lev",
]

# ── Exit alert ────────────────────────────────────────────────────────────────
EXIT_FORMAT = (
    "🔒 *VARANUS T2 EXIT* | {asset} {outcome}\n"
    "Entry: {entry_price:.4f} → Exit: {exit_price:.4f}\n"
    "PnL: {pnl_sign}${pnl_abs:.2f} | Duration: {duration_h:.0f}h\n"
    "Outcome: {outcome_label}"
)

# ── Circuit breaker alert ─────────────────────────────────────────────────────
HALT_FORMAT = (
    "🚨 *VARANUS T2 — SIGNALS HALTED*\n"
    "Daily Loss: {daily_loss_pct:.1f}% | Drawdown: {drawdown_pct:.1f}%\n"
    "Current Equity: ${current_equity:.2f}\n"
    "Reason: {reason}"
)


def _post(msg: str, bot_token: str, chat_id: str) -> None:
    """Send a Markdown message via the Telegram Bot API. Fails silently."""
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"},
            timeout=5.0,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"[alerts] Warning: Telegram send failed: {e}")


def send_alert(trade: dict, bot_token: str, chat_id: str,
               dry_run: bool = False) -> None:
    """
    Validate and send an entry signal alert to Telegram.
    Raises ValueError if any REQUIRED_FIELDS are missing.
    In dry_run mode, prints the message instead of sending.
    """
    missing = [f for f in REQUIRED_FIELDS if f not in trade]
    if missing:
        raise ValueError(f"Alert missing fields: {missing}")

    msg = ALERT_FORMAT.format(**trade)

    if dry_run:
        print(f"[dry-run] Entry alert:\n{msg}\n")
        return

    _post(msg, bot_token, chat_id)


def send_exit_alert(trade: dict, bot_token: str, chat_id: str,
                    dry_run: bool = False) -> None:
    """
    Send a trade exit alert (TP / SL / time).
    Expects keys: asset, entry_price, exit_price, pnl_usd, outcome,
                  entry_ts, exit_ts.
    """
    import pandas as pd

    outcome = trade.get("outcome", "unknown")
    outcome_labels = {
        "tp":   "✅ Take-Profit Hit",
        "sl":   "❌ Stop-Loss Hit",
        "time": "⏱ Time Barrier (Force Close)",
    }

    try:
        duration_h = (
            pd.to_datetime(trade["exit_ts"]) - pd.to_datetime(trade["entry_ts"])
        ).total_seconds() / 3600
    except Exception:
        duration_h = 0.0

    pnl = trade.get("pnl_usd", 0.0)
    msg = EXIT_FORMAT.format(
        asset         = trade.get("asset", "?"),
        outcome       = outcome.upper(),
        entry_price   = float(trade.get("entry_price", 0)),
        exit_price    = float(trade.get("exit_price", 0)),
        pnl_sign      = "+" if pnl >= 0 else "-",
        pnl_abs       = abs(pnl),
        duration_h    = duration_h,
        outcome_label = outcome_labels.get(outcome, outcome),
    )

    if dry_run:
        print(f"[dry-run] Exit alert:\n{msg}\n")
        return

    _post(msg, bot_token, chat_id)


def send_no_signal_alert(cycle_time: str, equity: float, daily_pct: float,
                         bot_token: str, chat_id: str,
                         dry_run: bool = False) -> None:
    """Send a no-signal notification at the end of each cycle."""
    msg = (
        f"🔍 *VARANUS T2 — No Signal*\n"
        f"Cycle: {cycle_time}\n"
        f"Scanned 15 assets — no setup above confidence threshold\n"
        f"Equity: ${equity:,.2f} | Daily: {daily_pct:+.1f}%"
    )
    if dry_run:
        print(f"[dry-run] No-signal alert:\n{msg}\n")
        return
    _post(msg, bot_token, chat_id)


def send_heartbeat_alert(state: dict, health: dict,
                         bot_token: str, chat_id: str) -> None:
    """Send current portfolio status in response to a heartbeat command."""
    open_trades  = state.get("open_trades", {})
    closed       = state.get("closed_trades", [])
    initial      = state.get("initial_capital", 0.0)
    equity       = health["current_equity"]
    total_pnl    = equity - initial
    pnl_pct      = total_pnl / initial * 100 if initial else 0.0
    pnl_sign     = "+" if total_pnl >= 0 else ""
    halted       = "🚨 YES" if state.get("halted") else "✅ No"

    lines = [
        "💓 *VARANUS T2 — Heartbeat*",
        f"Equity: ${equity:,.2f} ({pnl_sign}{pnl_pct:.1f}%)",
        f"Daily: {health['daily_loss_pct']:+.1f}% | Drawdown: {health['drawdown_pct']:+.1f}%",
        f"Halted: {halted}",
        f"Open: {len(open_trades)} | Closed: {len(closed)}",
    ]

    if open_trades:
        lines.append("── Open Positions ──")
        for asset, t in open_trades.items():
            d = "LONG ↑" if t["direction"] == 1 else "SHORT ↓"
            lines.append(
                f"  {asset} {d} @ {t['entry_price']} | "
                f"TP {t['take_profit']} | SL {t['stop_loss']} | ${t['position_usd']:.0f}"
            )

    if closed:
        wins    = sum(1 for t in closed if t.get("pnl_usd", 0) > 0)
        tot_pnl = sum(t.get("pnl_usd", 0) for t in closed)
        lines.append(f"── Closed: {wins}/{len(closed)} wins | PnL ${tot_pnl:+.2f} ──")

    _post("\n".join(lines), bot_token, chat_id)


def send_halt_alert(health: dict, bot_token: str, chat_id: str,
                    dry_run: bool = False) -> None:
    """
    Send a circuit-breaker halt notification.
    Expects the dict returned by risk.check_portfolio_health().
    """
    daily  = health.get("daily_loss_pct", 0.0)
    dd     = health.get("drawdown_pct", 0.0)
    equity = health.get("current_equity", 0.0)

    if daily <= -5.0 and dd <= -15.0:
        reason = "Daily loss AND peak drawdown limits both breached"
    elif daily <= -5.0:
        reason = f"Daily loss limit breached ({daily:.1f}%)"
    else:
        reason = f"Peak drawdown limit breached ({dd:.1f}%)"

    msg = HALT_FORMAT.format(
        daily_loss_pct = daily,
        drawdown_pct   = dd,
        current_equity = equity,
        reason         = reason,
    )

    if dry_run:
        print(f"[dry-run] Halt alert:\n{msg}\n")
        return

    _post(msg, bot_token, chat_id)
