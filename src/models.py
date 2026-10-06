"""
Member 2: Dual ML Models & Uncertainty Estimation
"Seeing Next Month, Not Just Last Month" - Predictive Dashboard

Provides:
1. Calibrated Churn Classifier (HistGradientBoostingClassifier + CalibratedClassifierCV)
2. Sales/MRR Regressor (Ridge Regression with non-negative projection)
3. 95% Residual Prediction Interval Estimation
4. Temporal Train/Test Splitting (No Future Leakage)
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    precision_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.pipeline import Pipeline

from src.config import (
    CHURN_FEATURE_COLUMNS,
    IDENTIFIER_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    SALES_FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from src.pipeline import get_preprocessor, get_sales_preprocessor


def temporal_train_test_split(
    df: pd.DataFrame,
    split_date: Optional[str] = None,
    test_ratio: float = 0.20,
    cutoff_col: str = "cutoff_date",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs a strict chronological/temporal split across snapshot cutoffs
    to prevent future-to-past temporal leakage.

    Earlier snapshots form the training set; later snapshots form the holdout test set.
    """
    if cutoff_col not in df.columns:
        # Fallback to index-based chronological split if cutoff_col not present
        split_idx = int(len(df) * (1.0 - test_ratio))
        return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

    unique_cutoffs = sorted(df[cutoff_col].unique())
    if len(unique_cutoffs) < 2:
        # Single cutoff provided (e.g., small test fixture): split rows chronologically
        split_idx = max(1, int(len(df) * (1.0 - test_ratio)))
        return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

    if split_date is not None:
        train_df = df[df[cutoff_col] < split_date].copy()
        test_df = df[df[cutoff_col] >= split_date].copy()
    else:
        split_idx = max(1, int(len(unique_cutoffs) * (1.0 - test_ratio)))
        train_cutoffs = set(unique_cutoffs[:split_idx])
        test_cutoffs = set(unique_cutoffs[split_idx:])
        train_df = df[df[cutoff_col].isin(train_cutoffs)].copy()
        test_df = df[df[cutoff_col].isin(test_cutoffs)].copy()

    # Safety: ensure both sets are non-empty
    if train_df.empty or test_df.empty:
        split_idx = max(1, int(len(df) * (1.0 - test_ratio)))
        return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

    return train_df, test_df


class ChurnModel:
    """
    Calibrated Churn Classifier combining HistGradientBoostingClassifier
    and CalibratedClassifierCV (isotonic / sigmoid calibration).
    """

    def __init__(
        self,
        preprocessor: Optional[ColumnTransformer] = None,
        base_params: Optional[Dict[str, Any]] = None,
        calibration_method: str = "sigmoid",
        cv: int = 3,
        random_state: int = 42,
    ):
        self.preprocessor = preprocessor or get_preprocessor()
        self.calibration_method = calibration_method
        self.cv = cv
        self.random_state = random_state

        default_base_params = {
            "max_iter": 100,
            "learning_rate": 0.08,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
            "random_state": self.random_state,
        }
        if base_params:
            default_base_params.update(base_params)

        self.base_estimator = HistGradientBoostingClassifier(**default_base_params)
        self.calibrated_classifier = CalibratedClassifierCV(
            estimator=self.base_estimator,
            method=self.calibration_method,
            cv=self.cv,
        )
        self.pipeline = Pipeline(
            steps=[
                ("preprocessor", self.preprocessor),
                ("classifier", self.calibrated_classifier),
            ]
        )
        self.is_fitted = False
        self.risk_threshold: Optional[float] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ChurnModel":
        """
        Fits the preprocessor and calibrated classifier on training data.
        """
        # Ensure only feature columns are passed
        feat_cols = [c for c in MODEL_FEATURE_COLUMNS if c in X.columns]
        self.pipeline.fit(X[feat_cols], y)
        self.is_fitted = True
        return self

    def compute_training_risk_threshold(
        self, X_train: pd.DataFrame, percentile: float = 0.90
    ) -> float:
        """
        Determines the operational risk queue threshold from the training risk score distribution.
        Strictly leakage-safe: uses training data only, never untouched holdout data.
        """
        train_probs = self.predict_proba(X_train)
        threshold = float(np.quantile(train_probs, percentile))
        self.risk_threshold = threshold
        return threshold

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts calibrated churn probabilities for positive class (churn = 1).
        Guaranteed to be within [0.0, 1.0].
        """
        if not self.is_fitted:
            raise RuntimeError("ChurnModel must be fitted before calling predict_proba.")
        feat_cols = [c for c in MODEL_FEATURE_COLUMNS if c in X.columns]
        probs = self.pipeline.predict_proba(X[feat_cols])[:, 1]
        return np.clip(probs, 0.0, 1.0)

    def predict(self, X: pd.DataFrame, threshold: float = 0.50) -> np.ndarray:
        """
        Generates discrete binary predictions based on specified probability threshold.
        """
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Calculates classification evaluation metrics on holdout test data.
        """
        probs = self.predict_proba(X)
        preds = (probs >= 0.50).astype(int)

        # ROC-AUC calculation (handles single-class safely)
        try:
            auc = float(roc_auc_score(y, probs))
        except ValueError:
            auc = 0.50

        prec = float(precision_score(y, preds, zero_division=0))
        rec = float(recall_score(y, preds, zero_division=0))
        f1 = float(f1_score(y, preds, zero_division=0))
        brier = float(brier_score_loss(y, probs))
        cm = confusion_matrix(y, preds).tolist()

        return {
            "churn_auc": round(auc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "brier_score": round(brier, 4),
            "confusion_matrix": cm,
            "sample_count": len(y),
        }


class SalesModel:
    """
    Revenue / MRR Forecaster using Ridge Regression with feature preprocessing
    and non-negative prediction projection.
    Strictly excludes current contract revenue features (mrr_amount, arr_amount) to prevent
    identity-function shortcuts and force genuine forward prediction from usage/tenure/tier.
    """

    def __init__(
        self,
        preprocessor: Optional[ColumnTransformer] = None,
        feature_columns: Optional[List[str]] = None,
        alpha: float = 1.0,
        random_state: int = 42,
    ):
        self.feature_columns = feature_columns or SALES_FEATURE_COLUMNS
        self.preprocessor = preprocessor or get_sales_preprocessor()
        self.alpha = alpha
        self.random_state = random_state

        self.regressor = Ridge(alpha=self.alpha, random_state=self.random_state)
        self.pipeline = Pipeline(
            steps=[
                ("preprocessor", self.preprocessor),
                ("regressor", self.regressor),
            ]
        )
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SalesModel":
        """
        Fits the preprocessor and Ridge regressor on training data.
        """
        feat_cols = [c for c in self.feature_columns if c in X.columns]
        self.pipeline.fit(X[feat_cols], y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generates point sales forecasts, bounded below at 0.0 (non-negative revenue).
        """
        if not self.is_fitted:
            raise RuntimeError("SalesModel must be fitted before calling predict.")
        feat_cols = [c for c in self.feature_columns if c in X.columns]
        raw_preds = self.pipeline.predict(X[feat_cols])
        return np.maximum(0.0, raw_preds)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Calculates regression metrics on holdout test data:
        - RMSE: root mean squared error
        - MAE: mean absolute error
        - MAPE: calculated defensibly on non-zero actual future revenue (y > 0)
        - WMAPE: weighted MAPE sum(|y - pred|) / sum(|y|)
        """
        preds = self.predict(X)
        y_true = y.values.astype(float)

        rmse = float(root_mean_squared_error(y_true, preds))
        mae = float(mean_absolute_error(y_true, preds))

        # Defensible MAPE on non-zero actuals
        non_zero_mask = y_true > 0
        if np.any(non_zero_mask):
            mape = float(
                np.mean(
                    np.abs(y_true[non_zero_mask] - preds[non_zero_mask])
                    / y_true[non_zero_mask]
                )
            )
        else:
            mape = 0.0

        # WMAPE
        sum_actual = float(np.sum(np.abs(y_true)))
        if sum_actual > 0:
            wmape = float(np.sum(np.abs(y_true - preds)) / sum_actual)
        else:
            wmape = 0.0

        return {
            "sales_mape": round(mape, 4),
            "sales_rmse": round(rmse, 2),
            "sales_mae": round(mae, 2),
            "sales_wmape": round(wmape, 4),
            "sample_count": len(y),
        }


def compute_prediction_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    z_multiplier: float = 1.96,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Estimates 95% uncertainty prediction intervals using residual standard deviation:
    interval = pred +/- 1.96 * sigma_residual.
    Enforces non-negativity: lower_bound = max(0.0, pred - half_width).
    """
    residuals = y_true - y_pred
    if len(residuals) > 1:
        residual_std = float(np.std(residuals, ddof=1))
    else:
        residual_std = float(np.std(residuals, ddof=0)) if len(residuals) > 0 else 1.0

    half_width = z_multiplier * residual_std
    lower_bound = np.maximum(0.0, y_pred - half_width).round(2)
    upper_bound = (y_pred + half_width).round(2)

    return lower_bound, upper_bound, round(residual_std, 4)
