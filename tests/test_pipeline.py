"""
Unit and Integration Tests for the Data Engineering Pipeline
"""

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pytest

from src.config import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    IDENTIFIER_COLUMNS,
    LEAKAGE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
)
from src.pipeline import (
    audit_leakage,
    calculate_target,
    compute_account_features,
    compute_feature_usage_aggregations,
    compute_subscription_features,
    compute_support_ticket_aggregations,
    filter_active_subscriptions,
    get_preprocessor,
)


@pytest.fixture
def sample_data():
    """Generates synthetic test DataFrames for unit testing pipeline functions."""
    cutoff = pd.Timestamp("2024-06-30 23:59:59")

    accounts_df = pd.DataFrame(
        [
            {
                "account_id": "A1",
                "account_name": "Company_Alpha",
                "industry": "DevTools",
                "country": "US",
                "signup_date": pd.Timestamp("2024-01-01"),
                "referral_source": "organic",
                "plan_tier": "Pro",
                "seats": 10,
                "is_trial": False,
                "churn_flag": False,
            },
            {
                "account_id": "A2",
                "account_name": "Company_Beta",
                "industry": "FinTech",
                "country": "UK",
                "signup_date": pd.Timestamp("2024-04-01"),
                "referral_source": "ads",
                "plan_tier": "Basic",
                "seats": 5,
                "is_trial": True,
                "churn_flag": True,
            },
        ]
    )

    subscriptions_df = pd.DataFrame(
        [
            # Active subscription, churns in next 30d (2024-07-15)
            {
                "subscription_id": "S1",
                "account_id": "A1",
                "start_date": pd.Timestamp("2024-01-15"),
                "end_date": pd.Timestamp("2024-07-15"),
                "plan_tier": "Pro",
                "seats": 10,
                "mrr_amount": 500,
                "arr_amount": 6000,
                "is_trial": False,
                "upgrade_flag": False,
                "downgrade_flag": False,
                "churn_flag": True,
                "billing_frequency": "monthly",
                "auto_renew_flag": True,
            },
            # Active subscription, does NOT churn in next 30d (end_date in Oct 2024)
            {
                "subscription_id": "S2",
                "account_id": "A1",
                "start_date": pd.Timestamp("2024-03-01"),
                "end_date": pd.Timestamp("2024-10-15"),
                "plan_tier": "Pro",
                "seats": 15,
                "mrr_amount": 750,
                "arr_amount": 9000,
                "is_trial": False,
                "upgrade_flag": True,
                "downgrade_flag": False,
                "churn_flag": True,
                "billing_frequency": "monthly",
                "auto_renew_flag": True,
            },
            # Active subscription, never churns (end_date is NaT)
            {
                "subscription_id": "S3",
                "account_id": "A2",
                "start_date": pd.Timestamp("2024-04-10"),
                "end_date": pd.NaT,
                "plan_tier": "Basic",
                "seats": 5,
                "mrr_amount": 100,
                "arr_amount": 1200,
                "is_trial": True,
                "upgrade_flag": False,
                "downgrade_flag": False,
                "churn_flag": False,
                "billing_frequency": "monthly",
                "auto_renew_flag": False,
            },
            # Inactive subscription: ended BEFORE cutoff (2024-05-30)
            {
                "subscription_id": "S4",
                "account_id": "A2",
                "start_date": pd.Timestamp("2024-04-01"),
                "end_date": pd.Timestamp("2024-05-30"),
                "plan_tier": "Basic",
                "seats": 2,
                "mrr_amount": 50,
                "arr_amount": 600,
                "is_trial": False,
                "upgrade_flag": False,
                "downgrade_flag": False,
                "churn_flag": True,
                "billing_frequency": "monthly",
                "auto_renew_flag": False,
            },
            # Future subscription: started AFTER cutoff (2024-07-05)
            {
                "subscription_id": "S5",
                "account_id": "A2",
                "start_date": pd.Timestamp("2024-07-05"),
                "end_date": pd.NaT,
                "plan_tier": "Enterprise",
                "seats": 50,
                "mrr_amount": 2500,
                "arr_amount": 30000,
                "is_trial": False,
                "upgrade_flag": False,
                "downgrade_flag": False,
                "churn_flag": False,
                "billing_frequency": "annual",
                "auto_renew_flag": True,
            },
        ]
    )

    feature_usage_df = pd.DataFrame(
        [
            # Valid usage in 30d window and recent 14d
            {
                "usage_id": "U1",
                "subscription_id": "S1",
                "usage_date": pd.Timestamp("2024-06-25"),
                "feature_name": "feature_1",
                "usage_count": 10,
                "usage_duration_secs": 600,
                "error_count": 1,
                "is_beta_feature": False,
            },
            # Valid usage in 30d window and prior 14d
            {
                "usage_id": "U2",
                "subscription_id": "S1",
                "usage_date": pd.Timestamp("2024-06-10"),
                "feature_name": "feature_2",
                "usage_count": 5,
                "usage_duration_secs": 300,
                "error_count": 0,
                "is_beta_feature": True,
            },
            # Usage older than 30 days (2024-05-15) - must be excluded from 30d window
            {
                "usage_id": "U3",
                "subscription_id": "S1",
                "usage_date": pd.Timestamp("2024-05-15"),
                "feature_name": "feature_3",
                "usage_count": 20,
                "usage_duration_secs": 1200,
                "error_count": 0,
                "is_beta_feature": False,
            },
            # Future usage after cutoff (2024-07-05) - must NOT leak
            {
                "usage_id": "U4",
                "subscription_id": "S1",
                "usage_date": pd.Timestamp("2024-07-05"),
                "feature_name": "feature_1",
                "usage_count": 100,
                "usage_duration_secs": 5000,
                "error_count": 5,
                "is_beta_feature": False,
            },
            # Usage before subscription start_date (2024-01-01 < 2024-01-15) - must NOT be included
            {
                "usage_id": "U5",
                "subscription_id": "S1",
                "usage_date": pd.Timestamp("2024-01-01"),
                "feature_name": "feature_1",
                "usage_count": 50,
                "usage_duration_secs": 2000,
                "error_count": 0,
                "is_beta_feature": False,
            },
        ]
    )

    support_tickets_df = pd.DataFrame(
        [
            # T1: Valid ticket in 60d window, closed BEFORE cutoff (2024-06-02), first response in 30 mins (2024-06-01 10:30 <= cutoff)
            {
                "ticket_id": "T1",
                "account_id": "A1",
                "submitted_at": pd.Timestamp("2024-06-01 10:00:00"),
                "closed_at": pd.Timestamp("2024-06-02 10:00:00"),
                "resolution_time_hours": 24.0,
                "priority": "urgent",
                "first_response_time_minutes": 30,
                "satisfaction_score": 4.0,
                "escalation_flag": True,
            },
            # T2: Submitted before cutoff (2024-06-30 22:00:00), closed AFTER cutoff (2024-07-02 12:00:00).
            # First response time is 180 min (2024-07-01 01:00:00 > cutoff).
            # Satisfaction is 5.0 (closed in future, must NOT leak).
            # Resolution time is 96h (closed in future, must NOT leak).
            {
                "ticket_id": "T2",
                "account_id": "A1",
                "submitted_at": pd.Timestamp("2024-06-30 22:00:00"),
                "closed_at": pd.Timestamp("2024-07-02 12:00:00"),
                "resolution_time_hours": 96.0,
                "priority": "medium",
                "first_response_time_minutes": 180,
                "satisfaction_score": 5.0,
                "escalation_flag": False,
            },
            # T3: Ticket older than 60 days (2024-04-01) - excluded from 60d window
            {
                "ticket_id": "T3",
                "account_id": "A1",
                "submitted_at": pd.Timestamp("2024-04-01"),
                "closed_at": pd.Timestamp("2024-04-02 10:00:00"),
                "resolution_time_hours": 24.0,
                "priority": "low",
                "first_response_time_minutes": 120,
                "satisfaction_score": 5.0,
                "escalation_flag": False,
            },
            # T4: Future ticket submitted AFTER cutoff (2024-07-10) - must NOT leak
            {
                "ticket_id": "T4",
                "account_id": "A1",
                "submitted_at": pd.Timestamp("2024-07-10"),
                "closed_at": pd.Timestamp("2024-07-11 10:00:00"),
                "resolution_time_hours": 24.0,
                "priority": "urgent",
                "first_response_time_minutes": 15,
                "satisfaction_score": 1.0,
                "escalation_flag": True,
            },
            # T5: Ticket before account signup (2023-12-01 < 2024-01-01) - must NOT be included
            {
                "ticket_id": "T5",
                "account_id": "A1",
                "submitted_at": pd.Timestamp("2023-12-01"),
                "closed_at": pd.Timestamp("2023-12-02 10:00:00"),
                "resolution_time_hours": 24.0,
                "priority": "high",
                "first_response_time_minutes": 45,
                "satisfaction_score": 3.0,
                "escalation_flag": False,
            },
        ]
    )

    return {
        "cutoff": cutoff,
        "accounts": accounts_df,
        "subscriptions": subscriptions_df,
        "feature_usage": feature_usage_df,
        "support_tickets": support_tickets_df,
    }


def test_active_subscription_filtering(sample_data):
    """Test: Only subscriptions active at cutoff_date are included."""
    cutoff = sample_data["cutoff"]
    subs = sample_data["subscriptions"]

    active = filter_active_subscriptions(subs, cutoff)
    active_ids = set(active["subscription_id"])

    assert active_ids == {"S1", "S2", "S3"}
    assert "S4" not in active_ids  # ended before cutoff
    assert "S5" not in active_ids  # started after cutoff


def test_target_calculation(sample_data):
    """Test: Target calculation for churn in (cutoff, cutoff + 30 days]."""
    cutoff = sample_data["cutoff"]
    subs = sample_data["subscriptions"]
    active = filter_active_subscriptions(subs, cutoff)

    target = calculate_target(active, cutoff, horizon_days=30)
    target_dict = dict(zip(active["subscription_id"], target))

    assert target_dict["S1"] == 1  # end_date = 2024-07-15 (within 30 days)
    assert target_dict["S2"] == 0  # end_date = 2024-10-15 (outside 30 days)
    assert target_dict["S3"] == 0  # end_date = NaT (active)


def test_satisfaction_closed_after_cutoff_excluded(sample_data):
    """Test 1 & 2: Satisfaction from ticket closed after cutoff is excluded; closed on/before is included."""
    cutoff = sample_data["cutoff"]
    active = filter_active_subscriptions(sample_data["subscriptions"], cutoff)
    accounts = sample_data["accounts"]
    tickets = sample_data["support_tickets"]

    support_feats = compute_support_ticket_aggregations(active, accounts, tickets, cutoff, window_days=60)
    s1_row = support_feats[support_feats["subscription_id"] == "S1"].iloc[0]

    # T1 was closed before cutoff (satisfaction_score = 4.0) -> INCLUDED
    # T2 was closed after cutoff (satisfaction_score = 5.0) -> EXCLUDED
    # Result: only T1 contributes -> avg = 4.0, count = 1
    assert s1_row["satisfaction_response_count_60d"] == 1
    assert s1_row["avg_satisfaction_score_60d"] == 4.0


def test_first_response_time_after_cutoff_excluded(sample_data):
    """Test 3 & 4: First response time occurring after cutoff is excluded; on/before cutoff is included."""
    cutoff = sample_data["cutoff"]
    active = filter_active_subscriptions(sample_data["subscriptions"], cutoff)
    accounts = sample_data["accounts"]
    tickets = sample_data["support_tickets"]

    support_feats = compute_support_ticket_aggregations(active, accounts, tickets, cutoff, window_days=60)
    s1_row = support_feats[support_feats["subscription_id"] == "S1"].iloc[0]

    # T1 response was at 2024-06-01 10:30 (before cutoff) -> INCLUDED (30 mins)
    # T2 response was at 2024-07-01 01:00 (after cutoff) -> EXCLUDED
    # Result: only T1 response time contributes -> avg = 30.0 mins
    assert s1_row["avg_first_response_time_minutes_60d"] == 30.0
    # Both T1 and T2 were submitted by cutoff -> ticket_count_60d = 2
    assert s1_row["ticket_count_60d"] == 2


def test_resolution_time_after_cutoff_excluded(sample_data):
    """Test 5: Resolution time for tickets closed after cutoff is excluded."""
    cutoff = sample_data["cutoff"]
    active = filter_active_subscriptions(sample_data["subscriptions"], cutoff)
    accounts = sample_data["accounts"]
    tickets = sample_data["support_tickets"]

    support_feats = compute_support_ticket_aggregations(active, accounts, tickets, cutoff, window_days=60)
    s1_row = support_feats[support_feats["subscription_id"] == "S1"].iloc[0]

    # T1 closed before cutoff (24.0h). T2 closed after cutoff (96.0h).
    # Result: only T1 resolution time is used -> avg = 24.0h
    assert s1_row["avg_resolution_time_hours_60d"] == 24.0


def test_no_future_usage_leakage(sample_data):
    """Test: Feature usage after cutoff date is strictly excluded."""
    cutoff = sample_data["cutoff"]
    active = filter_active_subscriptions(sample_data["subscriptions"], cutoff)
    usage = sample_data["feature_usage"]

    usage_feats = compute_feature_usage_aggregations(active, usage, cutoff, window_days=30)
    s1_row = usage_feats[usage_feats["subscription_id"] == "S1"].iloc[0]

    # Future usage (U4: count=100) must NOT be included in usage_count_30d
    # Valid usage in 30d: U1 (count=10) + U2 (count=5) = 15
    assert s1_row["usage_count_30d"] == 15
    assert s1_row["error_count_30d"] == 1


def test_correct_30d_usage_window(sample_data):
    """Test: Usage older than 30 days and usage before subscription start are excluded."""
    cutoff = sample_data["cutoff"]
    active = filter_active_subscriptions(sample_data["subscriptions"], cutoff)
    usage = sample_data["feature_usage"]

    usage_feats = compute_feature_usage_aggregations(active, usage, cutoff, window_days=30)
    s1_row = usage_feats[usage_feats["subscription_id"] == "S1"].iloc[0]

    # U3 (older than 30d) and U5 (before start_date) must be excluded
    assert s1_row["usage_duration_secs_30d"] == 600 + 300  # U1 (600) + U2 (300) = 900
    assert s1_row["unique_features_30d"] == 2


def test_velocity_calculation_finite_and_safe(sample_data):
    """Test: Usage velocity produces finite values and handles zeros safely."""
    cutoff = sample_data["cutoff"]
    active = filter_active_subscriptions(sample_data["subscriptions"], cutoff)
    usage = sample_data["feature_usage"]

    usage_feats = compute_feature_usage_aggregations(active, usage, cutoff, window_days=30)

    # S1: recent 14d (U1 count=10), prior 14d (U2 count=5)
    # Velocity = (10 - 5) / (5 + 1.0) = 5 / 6 = 0.8333...
    s1_val = usage_feats[usage_feats["subscription_id"] == "S1"]["usage_growth_14d_vs_prior14d"].iloc[0]
    assert np.isclose(s1_val, 5.0 / 6.0)

    # S2 & S3 have 0 usage in both windows -> velocity must be 0.0 (no NaN, no Inf)
    for sub_id in ["S2", "S3"]:
        val = usage_feats[usage_feats["subscription_id"] == sub_id]["usage_growth_14d_vs_prior14d"].iloc[0]
        assert val == 0.0
        assert not np.isnan(val)
        assert not np.isinf(val)


def test_upgrade_and_downgrade_flags_excluded_from_model_features():
    """Test 6 & 7: upgrade_flag and downgrade_flag are NOT in MODEL_FEATURE_COLUMNS."""
    assert "upgrade_flag" not in MODEL_FEATURE_COLUMNS
    assert "downgrade_flag" not in MODEL_FEATURE_COLUMNS


def test_final_model_feature_count_is_25():
    """Test 8: Final model feature count is exactly 25."""
    assert len(NUMERICAL_FEATURES) == 18
    assert len(CATEGORICAL_FEATURES) == 5
    assert len(BOOLEAN_FEATURES) == 2
    assert len(MODEL_FEATURE_COLUMNS) == 25


def test_no_leakage_columns_in_model_features():
    """Test 10: No prohibited leakage columns exist in MODEL_FEATURE_COLUMNS."""
    for col in LEAKAGE_COLUMNS:
        assert col not in MODEL_FEATURE_COLUMNS, f"Prohibited leakage column in features: {col}"
    assert "end_date" not in MODEL_FEATURE_COLUMNS
    assert "churn_flag" not in MODEL_FEATURE_COLUMNS


def test_preprocessor_structure():
    """Test: ColumnTransformer is configured correctly and not fitted."""
    preprocessor = get_preprocessor()
    assert hasattr(preprocessor, "transformers")
    names = [name for name, _, _ in preprocessor.transformers]
    assert "num" in names
    assert "cat" in names
    assert "bool" in names
