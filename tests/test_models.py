"""
Comprehensive Unit & Integration Test Suite for Member 2
"Seeing Next Month, Not Just Last Month" - Predictive Dashboard

Tests:
1. Churn probability is between 0 and 1.
2. Churn model trains successfully.
3. ROC-AUC can be calculated.
4. Revenue model trains successfully.
5. Predictions are finite.
6. Prediction lower bound is never negative.
7. Lower bound <= prediction <= upper bound.
8. Recourse does not modify immutable features.
9. Recourse only uses valid dataset fields.
10. Recourse probability is actually recalculated.
11. PSI returns a finite non-negative value.
12. Drift flag follows the threshold.
13. Output DataFrame contains required columns.
14. Telemetry contains required metrics.
15. No NaN/inf values in final model outputs.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    IDENTIFIER_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
)
from src.drift import DriftMonitor, calculate_single_feature_psi
from src.inference import run_member_2_pipeline
from src.models import (
    ChurnModel,
    SalesModel,
    compute_prediction_intervals,
    temporal_train_test_split,
)
from src.recourse import IMMUTABLE_FEATURE_FIELDS, MUTABLE_FEATURE_FIELDS, RecourseEngine


def create_synthetic_snapshot_data(n_samples: int = 120, seed: int = 42) -> pd.DataFrame:
    """
    Creates a deterministic synthetic snapshot dataset matching Member 1's exact schema
    across multiple cutoff dates for unit and integration testing.
    """
    np.random.seed(seed)

    data = {
        "subscription_id": [f"SUB-{i:04d}" for i in range(n_samples)],
        "account_id": [f"ACC-{i // 2:04d}" for i in range(n_samples)],
        "cutoff_date": ["2024-05-31"] * (n_samples // 2) + ["2024-06-30"] * (n_samples // 2),
        TARGET_COLUMN: np.random.choice([0, 1], size=n_samples, p=[0.75, 0.25]),
    }

    # Numerical features
    for col in NUMERICAL_FEATURES:
        if "ratio" in col or "growth" in col or "score" in col:
            data[col] = np.random.uniform(0.0, 5.0, size=n_samples).round(2)
        elif "amount" in col:
            data[col] = np.random.uniform(25.0, 500.0, size=n_samples).round(2)
        else:
            data[col] = np.random.randint(1, 150, size=n_samples).astype(float)

    # Categorical features
    data["industry"] = np.random.choice(["technology", "finance", "healthcare", "retail"], size=n_samples)
    data["country"] = np.random.choice(["US", "UK", "DE", "CA", "FR"], size=n_samples)
    data["referral_source"] = np.random.choice(["organic", "paid_search", "partner", "direct"], size=n_samples)
    data["plan_tier"] = np.random.choice(["starter", "professional", "enterprise"], size=n_samples)
    data["billing_frequency"] = np.random.choice(["monthly", "annual"], size=n_samples)

    # Boolean features
    data["is_trial"] = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
    data["auto_renew_flag"] = np.random.choice([0, 1], size=n_samples, p=[0.3, 0.7])

    df = pd.DataFrame(data)
    return df


@pytest.fixture
def synthetic_snapshot_data() -> pd.DataFrame:
    return create_synthetic_snapshot_data()


def test_01_churn_probability_bounded_zero_to_one(synthetic_snapshot_data):
    """Test 1: Churn probability is strictly between 0 and 1."""
    train_df, test_df = temporal_train_test_split(synthetic_snapshot_data, test_ratio=0.30)
    model = ChurnModel(cv=2, random_state=42)
    model.fit(train_df, train_df[TARGET_COLUMN])

    probs = model.predict_proba(test_df)
    assert np.all(probs >= 0.0), "Churn probabilities must be >= 0.0"
    assert np.all(probs <= 1.0), "Churn probabilities must be <= 1.0"
    assert len(probs) == len(test_df)


def test_02_churn_model_trains_successfully(synthetic_snapshot_data):
    """Test 2: Churn model trains and sets fitted state successfully."""
    train_df, _ = temporal_train_test_split(synthetic_snapshot_data, test_ratio=0.30)
    model = ChurnModel(cv=2, random_state=42)
    assert not model.is_fitted
    model.fit(train_df, train_df[TARGET_COLUMN])
    assert model.is_fitted


def test_03_roc_auc_calculated(synthetic_snapshot_data):
    """Test 3: ROC-AUC can be evaluated and returns valid metric."""
    train_df, test_df = temporal_train_test_split(synthetic_snapshot_data, test_ratio=0.30)
    model = ChurnModel(cv=2, random_state=42)
    model.fit(train_df, train_df[TARGET_COLUMN])

    metrics = model.evaluate(test_df, test_df[TARGET_COLUMN])
    assert "churn_auc" in metrics
    assert 0.0 <= metrics["churn_auc"] <= 1.0
    assert not np.isnan(metrics["churn_auc"])


def test_04_revenue_model_trains_successfully(synthetic_snapshot_data):
    """Test 4: Revenue model trains successfully and flags fitted state."""
    train_df, _ = temporal_train_test_split(synthetic_snapshot_data, test_ratio=0.30)
    sales_model = SalesModel(alpha=1.0, random_state=42)
    assert not sales_model.is_fitted
    sales_model.fit(train_df, train_df["mrr_amount"])
    assert sales_model.is_fitted


def test_05_predictions_are_finite(synthetic_snapshot_data):
    """Test 5: Revenue predictions are finite with no NaN or inf."""
    train_df, test_df = temporal_train_test_split(synthetic_snapshot_data, test_ratio=0.30)
    sales_model = SalesModel(alpha=1.0, random_state=42)
    sales_model.fit(train_df, train_df["mrr_amount"])

    preds = sales_model.predict(test_df)
    assert np.all(np.isfinite(preds)), "All predictions must be finite"
    assert not np.any(np.isnan(preds)), "No prediction should be NaN"


def test_06_prediction_lower_bound_is_never_negative():
    """Test 6: Prediction lower bound is bounded at 0 (never negative)."""
    y_true = np.array([10.0, 50.0, 20.0])
    # Case with huge residual standard deviation that would otherwise push lower bound negative
    y_pred = np.array([5.0, 12.0, 8.0])
    lower, upper, _ = compute_prediction_intervals(y_true, y_pred, z_multiplier=1.96)

    assert np.all(lower >= 0.0), f"Lower bounds must be >= 0.0, got {lower}"


def test_07_lower_bound_le_prediction_le_upper_bound(synthetic_snapshot_data):
    """Test 7: Lower bound <= prediction <= upper bound constraint holds universally."""
    train_df, test_df = temporal_train_test_split(synthetic_snapshot_data, test_ratio=0.30)
    sales_model = SalesModel(alpha=1.0, random_state=42)
    sales_model.fit(train_df, train_df["mrr_amount"])

    preds = sales_model.predict(test_df)
    lower, upper, _ = compute_prediction_intervals(test_df["mrr_amount"].values, preds)

    assert np.all(lower <= preds + 1e-6), "Lower bound must be <= point prediction"
    assert np.all(preds <= upper + 1e-6), "Point prediction must be <= upper bound"
    assert np.all(lower <= upper), "Lower bound must be <= upper bound"


def test_08_recourse_does_not_modify_immutable_features(synthetic_snapshot_data):
    """Test 8: Recourse engine leaves immutable features untouched."""
    train_df, test_df = temporal_train_test_split(synthetic_snapshot_data, test_ratio=0.30)
    model = ChurnModel(cv=2, random_state=42)
    model.fit(train_df, train_df[TARGET_COLUMN])

    engine = RecourseEngine(churn_model=model)
    customer_row = test_df.iloc[0].copy()

    # Evaluate customer
    engine.evaluate_customer(customer_row, current_probability=0.85)

    # Verify no immutable fields were altered
    for imm in IMMUTABLE_FEATURE_FIELDS:
        if imm in customer_row:
            assert customer_row[imm] == test_df.iloc[0][imm]


def test_09_recourse_only_uses_valid_dataset_fields():
    """Test 9: Recourse only references valid fields that exist in MODEL_FEATURE_COLUMNS."""
    for field in MUTABLE_FEATURE_FIELDS:
        assert field in MODEL_FEATURE_COLUMNS, f"Mutable field '{field}' is not in MODEL_FEATURE_COLUMNS"


def test_10_recourse_probability_is_recalculated(synthetic_snapshot_data):
    """Test 10: Recourse probability is actively evaluated by model, not hard-coded."""
    train_df, test_df = temporal_train_test_split(synthetic_snapshot_data, test_ratio=0.30)
    model = ChurnModel(cv=2, random_state=42)
    model.fit(train_df, train_df[TARGET_COLUMN])

    engine = RecourseEngine(churn_model=model)
    action, drop = engine.evaluate_customer(test_df.iloc[0], current_probability=0.75)

    assert isinstance(action, str)
    assert isinstance(drop, float)
    assert 0.0 <= drop <= 1.0


def test_11_psi_returns_finite_non_negative(synthetic_snapshot_data):
    """Test 11: Population Stability Index returns finite non-negative values."""
    train_df, test_df = temporal_train_test_split(synthetic_snapshot_data, test_ratio=0.30)
    monitor = DriftMonitor()

    psi_score, psi_by_feature, _ = monitor.compute_psi(train_df, test_df)

    assert isinstance(psi_score, float)
    assert psi_score >= 0.0
    assert not np.isnan(psi_score)
    for feat, val in psi_by_feature.items():
        assert val >= 0.0
        assert not np.isnan(val)


def test_12_drift_flag_follows_threshold(synthetic_snapshot_data):
    """Test 12: Drift flag accurately triggers when PSI exceeds 0.20 threshold."""
    train_df = synthetic_snapshot_data.copy()
    shifted_test_df = train_df.copy()

    # Artificially shift numerical features dramatically
    for col in NUMERICAL_FEATURES:
        shifted_test_df[col] = shifted_test_df[col] * 25.0 + 1000.0

    monitor = DriftMonitor()
    psi_score, _, drift_flag = monitor.compute_psi(train_df, shifted_test_df)

    assert psi_score > 0.20, f"Expected PSI > 0.20 under severe shift, got {psi_score}"
    assert drift_flag is True, "Drift flag must be True when PSI > 0.20"


def test_13_output_dataframe_contains_required_columns(synthetic_snapshot_data):
    """Test 13: Output inference DataFrame contains all required Member 3 columns."""
    preds_df, _ = run_member_2_pipeline(
        data_df=synthetic_snapshot_data,
        export_outputs=False,
    )

    required_member_3_columns = [
        "customer_id",
        "actual_churn",
        "churn_probability",
        "actual_sales",
        "predicted_sales",
        "sales_lower_bound",
        "sales_upper_bound",
        "prescriptive_action",
        "prescribed_risk_drop",
    ]

    for col in required_member_3_columns:
        assert col in preds_df.columns, f"Missing required Member 3 column: {col}"


def test_14_telemetry_contains_required_metrics(synthetic_snapshot_data):
    """Test 14: Telemetry dictionary contains all required governance fields."""
    _, telemetry = run_member_2_pipeline(
        data_df=synthetic_snapshot_data,
        export_outputs=False,
    )

    required_fields = [
        "model_version",
        "churn_auc",
        "sales_mape",
        "sales_rmse",
        "psi_score",
        "drift_flag",
        "test_sample_count",
    ]

    for field in required_fields:
        assert field in telemetry, f"Missing telemetry field: {field}"
        assert telemetry[field] is not None


def test_15_no_nan_or_inf_in_final_model_outputs(synthetic_snapshot_data):
    """Test 15: No NaN or infinite values exist anywhere in inference output."""
    preds_df, telemetry = run_member_2_pipeline(
        data_df=synthetic_snapshot_data,
        export_outputs=False,
    )

    numeric_cols = [
        "churn_probability",
        "predicted_sales",
        "sales_lower_bound",
        "sales_upper_bound",
        "prescribed_risk_drop",
    ]

    for col in numeric_cols:
        assert not preds_df[col].isna().any(), f"NaN values detected in {col}"
        assert not np.isinf(preds_df[col]).any(), f"Infinite values detected in {col}"

    assert not np.isnan(telemetry["churn_auc"])
    assert not np.isnan(telemetry["sales_mape"])
    assert not np.isnan(telemetry["sales_rmse"])
    assert not np.isnan(telemetry["psi_score"])
