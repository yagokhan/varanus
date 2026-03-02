"""
Varanus v5.0 — High-Resolution Regime Adaptation Strategy.

8-Fold 40/30/30 Rolling Window Walk-Forward Validation.
Power Setup 5x Leverage tier (confidence >= 0.95).
Regime-adaptive HPO: Profit Factor × Net Profit | MaxDD < 15%.

Protected baseline: v4/ (frozen at v4.0.0 — DO NOT IMPORT).
"""

__version__  = "5.0.0"
__strategy__ = "High-Resolution Regime Adaptation"
__tier__     = 2
__baseline__ = "v4/  (frozen — DO NOT import from this path)"
