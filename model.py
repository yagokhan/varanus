"""
varanus/model.py — XGBoost Model + V5 Leverage Logic (Step 4).

V5 changes vs v4:
  - Leverage map: fourth tier added (confidence >= 0.95 → 3x Power Setup, capped)
  - confidence_threshold: resolved from params at predict time (not hardcoded)
  - is_power_setup(): new helper for backtest/risk layer
  - build_features() removed — imported from pa_features (no duplication)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import xgboost as xgb

logger = logging.getLogger(__name__)

# ── Feature list (16 features — identical to v4) ──────────────────────────────

FEATURE_LIST: list[str] = [
    # PA features
    "mss_signal", "fvg_type", "fvg_distance_atr", "fvg_age_candles",
    "sweep_occurred", "htf_bias",
    # Chameleon confirmation
    "relative_volume", "rsi_14", "rsi_slope_3", "ema21_55_alignment",
    "atr_percentile_100",
    # Market character
    "volatility_rank", "volume_rank", "asset_tier_flag",
    "hour_of_day", "day_of_week",
]

# ── V5 Leverage Map ───────────────────────────────────────────────────────────

# (lower_inclusive, upper_exclusive): leverage_multiplier
CONFIDENCE_LEVERAGE_MAP_V5: dict = {
    (0.750, 0.850): 1.0,   # Base signal
    (0.850, 0.920): 2.0,   # Standard conviction
    (0.920, 0.950): 3.0,   # High conviction
    (0.950, 1.001): 3.0,   # Power Setup — capped at 3x (was 5x)
}

# ── Model config ──────────────────────────────────────────────────────────────

MODEL_CONFIG: dict = {
    "type":                 "XGBoostClassifier",
    "target_classes":       3,       # {-1, 0, 1}
    "confidence_threshold": None,    # Resolved from best_params_v5.json at runtime

    "xgb_params": {
        "n_estimators":          500,
        "max_depth":             6,
        "learning_rate":         0.05,
        "subsample":             0.8,
        "colsample_bytree":      0.8,
        "eval_metric":           "mlogloss",
        "early_stopping_rounds": 30,
        "use_label_encoder":     False,
        "objective":             "multi:softprob",
        "num_class":             3,
        "random_state":          42,
    },
}


# ── Leverage functions ────────────────────────────────────────────────────────

def get_leverage(confidence: float) -> float:
    """
    V5 leverage resolution with Power Setup tier.

    Tiers:
      [0.750, 0.850) → 1x   Base
      [0.850, 0.920) → 2x   Standard conviction
      [0.920, 0.950) → 3x   High conviction
      [0.950, 1.001) → 3x   Power Setup (capped from 5x)

    Returns 1.0 as safe default for any confidence below 0.750.
    """
    for (lo, hi), lev in CONFIDENCE_LEVERAGE_MAP_V5.items():
        if lo <= confidence < hi:
            return lev
    return 1.0


def is_power_setup(confidence: float) -> bool:
    """True if this trade qualifies for the 5x Power Setup tier."""
    return confidence >= 0.950


# ═══════════════════════════════════════════════════════════════════════════════
# VaranusModel
# ═══════════════════════════════════════════════════════════════════════════════

class VaranusModel:
    """
    XGBoost 3-class classifier wrapper for Varanus v5.

    Internal label mapping: {-1, 0, 1} → {0, 1, 2} for XGBoost.
    predict_proba() maps back so output columns correspond to classes [-1, 0, 1].
    """

    def __init__(self, config: dict = MODEL_CONFIG):
        self.config  = config
        self.model   = None
        self.classes_ = np.array([-1, 0, 1])

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val:   pd.DataFrame = None,
        y_val:   pd.Series   = None,
    ) -> None:
        """
        Train the XGBoost model.

        Applies Optuna-supplied XGB hyperparameters if present in self.config.
        """
        params = self.config["xgb_params"].copy()

        # Apply Optuna hyperparameter overrides if stored in config
        optuna_xgb_keys = {
            "xgb_max_depth":    "max_depth",
            "xgb_n_estimators": "n_estimators",
            "xgb_lr":           "learning_rate",
            "xgb_subsample":    "subsample",
        }
        for opt_key, xgb_key in optuna_xgb_keys.items():
            if opt_key in self.config:
                params[xgb_key] = self.config[opt_key]

        early_stopping = params.pop("early_stopping_rounds", None)

        # XGBoost requires 0-indexed labels: map {-1,0,1} → {0,1,2}
        y_train_mapped = y_train + 1
        self.model     = xgb.XGBClassifier(**params)

        if X_val is not None and y_val is not None:
            y_val_mapped = y_val + 1
            self.model.fit(
                X_train, y_train_mapped,
                eval_set    = [(X_train, y_train_mapped), (X_val, y_val_mapped)],
                verbose     = False,
            )
        else:
            self.model.fit(X_train, y_train_mapped, verbose=False)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns class probabilities shaped (n_samples, 3).
        Columns correspond to classes [-1, 0, 1] respectively.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")

        probs = self.model.predict_proba(X)

        if probs.shape[1] < 3:
            full_probs = np.zeros((probs.shape[0], 3))
            for i, c in enumerate(self.model.classes_):
                full_probs[:, int(c)] = probs[:, i]
            return full_probs

        return probs

    def predict(
        self,
        X:                  pd.DataFrame,
        confidence_thresh:  float = None,
    ) -> np.ndarray:
        """
        Returns predicted class {-1, 0, 1}.

        Signals below confidence_thresh are suppressed to 0.

        Args:
            X:                   Feature DataFrame.
            confidence_thresh:   Override threshold. Falls back to
                                 self.config['confidence_threshold'] then 0.750.
        """
        probs      = self.predict_proba(X)
        max_probs  = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1) - 1   # {0,1,2} → {-1,0,1}

        thresh = (
            confidence_thresh
            or self.config.get("confidence_threshold")
            or 0.750
        )
        predictions[max_probs < thresh] = 0

        return predictions

    def get_confidence(self, X: pd.DataFrame) -> np.ndarray:
        """Return the max class probability for each row."""
        return np.max(self.predict_proba(X), axis=1)
