"""
varanus/universe.py — Tier 2 Universe Definition (v5.0).

Full 20-asset static list re-validated for 2026 liquidity regime.
Any asset failing the $50M 24h vol gate for 5+ consecutive days
is suspended from the active universe for that HPO/scan cycle.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# ── Static universe (all 20 Tier 2 assets) ────────────────────────────────────

TIER2_UNIVERSE: list[str] = [
    "ADA",   # Cardano         — High OI, clean structure
    "AVAX",  # Avalanche       — Subnet narrative, high ATR
    "LINK",  # Chainlink       — Oracle premium, strong FVG reactions
    "DOT",   # Polkadot        — Parachain liquidity depth
    "TRX",   # Tron            — High volume, DeFi TVL anchor
    "NEAR",  # Near            — Sharding narrative, clean PA
    "UNI",   # Uniswap         — DeFi bellwether, fee-switch catalyst
    "SUI",   # Sui             — New L1, momentum-driven FVGs
    "ARB",   # Arbitrum        — L2 leader, airdrop-driven liquidity
    "OP",    # Optimism        — L2 counterpart, correlated to ARB
    "POL",   # Polygon/POL     — Mature L2, reliable MSS
    "APT",   # Aptos           — L1 narrative, sharp MSS events
    "ATOM",  # Cosmos          — IBC narrative, range-bound tendencies
    "FIL",   # Filecoin        — Storage narrative, high ATR spikes
    "ICP",   # Internet Computer — High vol, wide FVGs
    "TAO",   # Bittensor       — AI narrative, extreme volatility tier
    "ASTR",  # Astar           — Lower cap, higher momentum factor
    "KITE",  # Kite.ai         — Emerging, wider spreads (reduce size)
    "HBAR",  # Hedera          — Institutional backing, slow burn
    "LTC",   # Litecoin        — Halving cycles, legacy structure
]

# High-Volatility Sub-Tier: wider barriers + reduced position size (0.75×)
HIGH_VOL_SUBTIER: list[str] = ["TAO", "ASTR", "KITE", "ICP"]

# Timeframe config
TIER2_QUOTE       = "USDT"
TIER2_TF          = "4h"    # Primary timeframe
TIER2_TF_HTF      = "1d"    # Higher-timeframe bias filter
TIER2_MIN_VOL_USD = 50_000_000   # 24h minimum volume gate (USD)

V5_UNIVERSE_NOTE = (
    "Re-validate for 2026 liquidity regime before running HPO. "
    "Delist any asset that fails the vol gate for 5+ consecutive days."
)

# ── Exclusion rules ───────────────────────────────────────────────────────────

EXCLUSION_RULES: dict = {
    "min_volume_usd":         50_000_000,  # Suspend if 24h vol < $50M for 3+ days
    "options_expiry_pause":   True,        # Suppress signals ±2h around BTC/ETH
                                           # options expiry (last Friday of month)
    "high_vol_size_scalar":   0.75,        # Position size multiplier for HIGH_VOL_SUBTIER
    "power_setup_size_scalar": 1.25,       # v5: Power Setup (conf ≥ 0.95) size boost
}

# ── Public API ─────────────────────────────────────────────────────────────────

def get_symbols() -> list[str]:
    """Return the full Tier 2 universe as exchange symbol strings (BASE/USDT)."""
    return [f"{asset}/{TIER2_QUOTE}" for asset in TIER2_UNIVERSE]


def is_high_vol(asset: str) -> bool:
    """True if *asset* belongs to the high-volatility sub-tier."""
    return asset.upper() in HIGH_VOL_SUBTIER


def get_size_scalar(asset: str, confidence: float = 0.0) -> float:
    """
    Position size scalar for *asset*.

    v5 stacking logic (applied in this order):
      1. high_vol scalar (0.75) if asset in HIGH_VOL_SUBTIER
      2. power_setup scalar (1.25) if confidence >= 0.95
    Both may apply simultaneously (e.g. TAO at 0.95 conf = 0.75 × 1.25 = 0.9375×).
    """
    scalar = EXCLUSION_RULES["high_vol_size_scalar"] if is_high_vol(asset) else 1.0
    if confidence >= 0.95:
        scalar *= EXCLUSION_RULES["power_setup_size_scalar"]
    return scalar


def get_active_universe(volume_data: dict[str, float]) -> list[str]:
    """
    Filter TIER2_UNIVERSE by current 24h USD volume.

    Args:
        volume_data: mapping {asset: 24h_volume_usd}.
                     Assets missing from the dict are treated as zero volume.

    Returns:
        Subset of TIER2_UNIVERSE whose volume meets TIER2_MIN_VOL_USD.
    """
    active    = [
        asset for asset in TIER2_UNIVERSE
        if volume_data.get(asset, 0) >= EXCLUSION_RULES["min_volume_usd"]
    ]
    suspended = set(TIER2_UNIVERSE) - set(active)
    if suspended:
        logger.info(
            f"Volume-suspended assets ({len(suspended)}): {sorted(suspended)}"
        )
    return active


def is_options_expiry_window(dt: Optional[datetime] = None) -> bool:
    """
    Return True if *dt* falls within ±2 hours of BTC/ETH options expiry.
    Options expire at 08:00 UTC on the last Friday of each month.
    """
    if not EXCLUSION_RULES["options_expiry_pause"]:
        return False
    if dt is None:
        dt = datetime.now(timezone.utc)

    expiry     = _last_friday_of_month(dt.year, dt.month)
    expiry_utc = expiry.replace(hour=8, minute=0, second=0, microsecond=0)
    delta      = abs((dt - expiry_utc).total_seconds())
    return delta <= 2 * 3600


def fetch_volumes(exchange=None) -> dict[str, float]:
    """
    Fetch current 24h USD quote volume for all Tier 2 assets from Binance.

    Args:
        exchange: optional pre-initialized ccxt.binance instance.

    Returns:
        dict mapping asset base currency → 24h volume in USD.
    """
    try:
        import ccxt
    except ImportError:
        raise RuntimeError(
            "ccxt is required for fetch_volumes(). Install it in algo_env."
        )

    if exchange is None:
        exchange = ccxt.binance({"enableRateLimit": True})

    symbols = get_symbols()
    tickers = exchange.fetch_tickers(symbols)

    volumes: dict[str, float] = {}
    for asset in TIER2_UNIVERSE:
        sym    = f"{asset}/{TIER2_QUOTE}"
        ticker = tickers.get(sym, {})
        volumes[asset] = float(ticker.get("quoteVolume") or 0.0)

    return volumes


# ── Internal helpers ───────────────────────────────────────────────────────────

def _last_friday_of_month(year: int, month: int) -> datetime:
    """Return the last Friday of *year*/*month* as a date-only datetime (UTC)."""
    if month == 12:
        next_month_first = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        next_month_first = datetime(year, month + 1, 1, tzinfo=timezone.utc)
    last_day          = next_month_first - timedelta(days=1)
    days_since_friday = (last_day.weekday() - 4) % 7
    return last_day - timedelta(days=days_since_friday)
