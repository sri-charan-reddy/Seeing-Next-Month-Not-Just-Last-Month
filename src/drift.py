"""
Member 2: Population Stability Index (PSI) Drift Monitoring
"Seeing Next Month, Not Just Last Month" - Predictive Dashboard

Monitors feature population drift between training baseline and holdout/production cohorts.
Thresholds:
- PSI < 0.10: Stable (No Shift) -> drift_flag = False
- 0.10 <= PSI <= 0.20: Moderate Shift (Monitor) -> drift_flag = False
- PSI > 0.20: Critical Drift (Action Required) -> drift_flag = True
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.config import CATEGORICAL_FEATURES, MODEL_FEATURE_COLUMNS, NUMERICAL_FEATURES


def calculate_single_feature_psi(
    baseline: pd.Series,
    target: pd.Series,
    num_bins: int = 10,
    is_numeric: bool = True,
    epsilon: float = 1e-4,
) -> float:
    """
    Computes Population Stability Index for a single feature between baseline and target series.
    Safely handles zero-frequency bins using Laplace epsilon smoothing.
    """
    clean_base = baseline.dropna()
    clean_target = target.dropna()

    if len(clean_base) == 0 or len(clean_target) == 0:
        return 0.0

    if is_numeric:
        # Quantile bin edges based on baseline distribution
        try:
            percentiles = np.linspace(0, 100, num_bins + 1)
            raw_bins = np.percentile(clean_base, percentiles)
            bins = np.unique(raw_bins)

            if len(bins) < 2:
                # Fallback to uniform binning if quantiles are constant
                bins = np.linspace(clean_base.min() - 0.001, clean_base.max() + 0.001, num_bins + 1)

            bins[0] = -np.inf
            bins[-1] = np.inf

            base_counts, _ = np.histogram(clean_base, bins=bins)
            target_counts, _ = np.histogram(clean_target, bins=bins)
        except Exception:
            return 0.0
    else:
        # Categorical feature: group by categories
        categories = list(set(clean_base.unique()).union(set(clean_target.unique())))
        base_counts = np.array([np.sum(clean_base == c) for c in categories])
        target_counts = np.array([np.sum(clean_target == c) for c in categories])

    # Convert counts to smoothed proportions
    k = len(base_counts)
    if k == 0:
        return 0.0

    p_base = (base_counts + epsilon) / (np.sum(base_counts) + k * epsilon)
    p_target = (target_counts + epsilon) / (np.sum(target_counts) + k * epsilon)

    # PSI calculation: sum((p_target - p_base) * ln(p_target / p_base))
    psi_val = np.sum((p_target - p_base) * np.log(p_target / p_base))
    return float(max(0.0, psi_val))


class DriftMonitor:
    """
    Enterprise population drift detector for ML feature governance.
    """

    STABILITY_THRESHOLD = 0.10
    CRITICAL_DRIFT_THRESHOLD = 0.20

    def __init__(
        self,
        features: Optional[List[str]] = None,
        numerical_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
    ):
        self.features = features or MODEL_FEATURE_COLUMNS
        self.numerical_features = set(numerical_features or NUMERICAL_FEATURES)
        self.categorical_features = set(categorical_features or CATEGORICAL_FEATURES)

    def compute_psi(
        self,
        train_df: pd.DataFrame,
        holdout_df: pd.DataFrame,
    ) -> Tuple[float, Dict[str, float], bool]:
        """
        Computes feature-level and aggregate PSI between training baseline and holdout dataset.

        Returns:
            Tuple of:
            - aggregate_psi: Mean PSI across monitored features
            - psi_by_feature: Dictionary mapping feature name to its individual PSI
            - drift_flag: True if aggregate PSI exceeds 0.20
        """
        psi_by_feature: Dict[str, float] = {}

        for feat in self.features:
            if feat not in train_df.columns or feat not in holdout_df.columns:
                continue

            is_numeric = feat in self.numerical_features or pd.api.types.is_numeric_dtype(train_df[feat])
            feat_psi = calculate_single_feature_psi(
                baseline=train_df[feat],
                target=holdout_df[feat],
                num_bins=10,
                is_numeric=is_numeric,
            )
            psi_by_feature[feat] = round(feat_psi, 4)

        if not psi_by_feature:
            return 0.0, {}, False

        aggregate_psi = round(float(np.mean(list(psi_by_feature.values()))), 4)
        drift_flag = aggregate_psi > self.CRITICAL_DRIFT_THRESHOLD

        return aggregate_psi, psi_by_feature, drift_flag

    def get_drift_status_label(self, psi_score: float) -> str:
        """
        Returns governance human-readable label matching Member 3 Power BI view specifications.
        """
        if psi_score < self.STABILITY_THRESHOLD:
            return "Stable (No Shift)"
        elif psi_score <= self.CRITICAL_DRIFT_THRESHOLD:
            return "Moderate Shift (Monitor)"
        else:
            return "Critical Drift (Action Required)"
