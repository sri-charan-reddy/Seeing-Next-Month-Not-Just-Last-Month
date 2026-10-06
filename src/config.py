"""
Project Configuration and Global Constants for
"Seeing Next Month, Not Just Last Month" - Predictive Dashboard
"""

from pathlib import Path
from typing import List

# Base Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_DATASET_DIR = RAW_DATA_DIR / "dataset" if (RAW_DATA_DIR / "dataset").exists() else RAW_DATA_DIR
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Raw Data File Paths
ACCOUNTS_CSV = RAW_DATASET_DIR / "ravenstack_accounts.csv"
SUBSCRIPTIONS_CSV = RAW_DATASET_DIR / "ravenstack_subscriptions.csv"
FEATURE_USAGE_CSV = RAW_DATASET_DIR / "ravenstack_feature_usage.csv"
SUPPORT_TICKETS_CSV = RAW_DATASET_DIR / "ravenstack_support_tickets.csv"
CHURN_EVENTS_CSV = RAW_DATASET_DIR / "ravenstack_churn_events.csv"

# Processed Output Files
PROCESSED_SNAPSHOTS_CSV = PROCESSED_DATA_DIR / "subscription_churn_snapshots.csv"

# Modeling Snapshot Parameters
SNAPSHOT_START = "2023-03-31"
SNAPSHOT_END = "2024-11-30"
PREDICTION_HORIZON_DAYS = 30
USAGE_WINDOW_DAYS = 30
SUPPORT_WINDOW_DAYS = 60

# Column Categories & Feature Lists
IDENTIFIER_COLUMNS: List[str] = [
    "subscription_id",
    "account_id",
    "cutoff_date",
]

TARGET_COLUMN: str = "churn_next_30d"
SALES_TARGET_COLUMN: str = "future_mrr_30d"

NUMERICAL_FEATURES: List[str] = [
    "account_age_days",
    "active_tenure_days",
    "seats",
    "mrr_amount",
    "arr_amount",
    "usage_count_30d",
    "usage_duration_secs_30d",
    "error_count_30d",
    "unique_features_30d",
    "beta_feature_ratio_30d",
    "usage_growth_14d_vs_prior14d",
    "ticket_count_60d",
    "urgent_ticket_count_60d",
    "escalation_count_60d",
    "avg_resolution_time_hours_60d",
    "avg_first_response_time_minutes_60d",
    "avg_satisfaction_score_60d",
    "satisfaction_response_count_60d",
]

# Forward Sales/Revenue Model Features:
# Current contract revenue features (mrr_amount, arr_amount) are strictly excluded
# to prevent identity-feature shortcuts when predicting forward 30-day MRR.
SALES_NUMERICAL_FEATURES: List[str] = [
    c for c in NUMERICAL_FEATURES if c not in ["mrr_amount", "arr_amount"]
]

CATEGORICAL_FEATURES: List[str] = [
    "industry",
    "country",
    "referral_source",
    "plan_tier",
    "billing_frequency",
]

# Note: upgrade_flag and downgrade_flag are removed to avoid static lifecycle leakage
BOOLEAN_FEATURES: List[str] = [
    "is_trial",
    "auto_renew_flag",
]

MODEL_FEATURE_COLUMNS: List[str] = (
    NUMERICAL_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES
)

CHURN_FEATURE_COLUMNS: List[str] = MODEL_FEATURE_COLUMNS

SALES_FEATURE_COLUMNS: List[str] = (
    SALES_NUMERICAL_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES
)

# Known Leakage / Prohibited Feature Columns
LEAKAGE_COLUMNS: List[str] = [
    "end_date",
    "churn_flag",
    "churn_date",
    "reason_code",
    "refund_amount_usd",
    "preceding_upgrade_flag",
    "preceding_downgrade_flag",
    "is_reactivation",
    "feedback_text",
    "churn_event_id",
    "account_name",
    "upgrade_flag",
    "downgrade_flag",
]
