"""
Data Engineering Pipeline for "Seeing Next Month, Not Just Last Month" - Predictive Dashboard

Constructs a point-in-time subscription-level monthly snapshot dataset
with historical trailing features and leakage-free target formulation.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    ACCOUNTS_CSV,
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    CHURN_EVENTS_CSV,
    FEATURE_USAGE_CSV,
    IDENTIFIER_COLUMNS,
    LEAKAGE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    PREDICTION_HORIZON_DAYS,
    PROCESSED_DATA_DIR,
    PROCESSED_SNAPSHOTS_CSV,
    RAW_DATASET_DIR,
    SALES_FEATURE_COLUMNS,
    SALES_NUMERICAL_FEATURES,
    SALES_TARGET_COLUMN,
    SNAPSHOT_END,
    SNAPSHOT_START,
    SUBSCRIPTIONS_CSV,
    SUPPORT_TICKETS_CSV,
    SUPPORT_WINDOW_DAYS,
    TARGET_COLUMN,
    USAGE_WINDOW_DAYS,
)


def load_raw_data(
    data_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads the 4 relevant raw CSV tables without modifying them.

    Returns:
        Tuple of (accounts_df, subscriptions_df, feature_usage_df, support_tickets_df)
    """
    target_dir = data_dir or RAW_DATASET_DIR

    accounts_file = target_dir / "ravenstack_accounts.csv"
    subs_file = target_dir / "ravenstack_subscriptions.csv"
    usage_file = target_dir / "ravenstack_feature_usage.csv"
    tickets_file = target_dir / "ravenstack_support_tickets.csv"

    if not accounts_file.exists():
        raise FileNotFoundError(f"Accounts file not found at {accounts_file}")
    if not subs_file.exists():
        raise FileNotFoundError(f"Subscriptions file not found at {subs_file}")
    if not usage_file.exists():
        raise FileNotFoundError(f"Feature usage file not found at {usage_file}")
    if not tickets_file.exists():
        raise FileNotFoundError(f"Support tickets file not found at {tickets_file}")

    accounts_df = pd.read_csv(accounts_file)
    subscriptions_df = pd.read_csv(subs_file)
    feature_usage_df = pd.read_csv(usage_file)
    support_tickets_df = pd.read_csv(tickets_file)

    # Standardize and parse date columns
    accounts_df["signup_date"] = pd.to_datetime(accounts_df["signup_date"])
    subscriptions_df["start_date"] = pd.to_datetime(subscriptions_df["start_date"])
    subscriptions_df["end_date"] = pd.to_datetime(subscriptions_df["end_date"])
    feature_usage_df["usage_date"] = pd.to_datetime(feature_usage_df["usage_date"])
    support_tickets_df["submitted_at"] = pd.to_datetime(support_tickets_df["submitted_at"])
    support_tickets_df["closed_at"] = pd.to_datetime(support_tickets_df["closed_at"])

    # Standardize boolean types
    for b_col in ["is_trial", "upgrade_flag", "downgrade_flag", "auto_renew_flag"]:
        if b_col in subscriptions_df.columns:
            subscriptions_df[b_col] = subscriptions_df[b_col].astype(bool)
    if "is_beta_feature" in feature_usage_df.columns:
        feature_usage_df["is_beta_feature"] = feature_usage_df["is_beta_feature"].astype(bool)
    if "escalation_flag" in support_tickets_df.columns:
        support_tickets_df["escalation_flag"] = support_tickets_df["escalation_flag"].astype(bool)

    return accounts_df, subscriptions_df, feature_usage_df, support_tickets_df


def get_month_end_cutoffs(
    start_date: str = SNAPSHOT_START, end_date: str = SNAPSHOT_END
) -> List[pd.Timestamp]:
    """
    Generates all calendar month-end cutoff timestamps between start_date and end_date inclusive.
    """
    cutoffs = pd.date_range(start=start_date, end=end_date, freq="ME")
    return list(cutoffs)


def filter_active_subscriptions(
    subscriptions_df: pd.DataFrame, cutoff_date: pd.Timestamp
) -> pd.DataFrame:
    """
    Filters subscriptions that are active at the cutoff date:
    start_date <= cutoff_date AND (end_date IS NULL OR end_date > cutoff_date)
    """
    mask = (subscriptions_df["start_date"] <= cutoff_date) & (
        subscriptions_df["end_date"].isna() | (subscriptions_df["end_date"] > cutoff_date)
    )
    return subscriptions_df[mask].copy()


def calculate_target(
    active_subs_df: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    horizon_days: int = PREDICTION_HORIZON_DAYS,
) -> pd.Series:
    """
    Calculates the binary target `churn_next_30d`:
    1 if end_date occurs in (cutoff_date, cutoff_date + horizon_days], else 0.
    """
    target_window_end = cutoff_date + pd.Timedelta(days=horizon_days)
    is_churn = (
        active_subs_df["end_date"].notna()
        & (active_subs_df["end_date"] > cutoff_date)
        & (active_subs_df["end_date"] <= target_window_end)
    ).astype(int)
    is_churn.name = TARGET_COLUMN
    return is_churn


def calculate_forward_sales_target(
    active_subs_df: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    horizon_days: int = PREDICTION_HORIZON_DAYS,
) -> pd.Series:
    """
    Calculates forward 30-day / next-month MRR target:
    - 0.0 if the subscription terminates/churns within (cutoff_date, cutoff_date + horizon_days]
    - current mrr_amount if the subscription remains active through the next 30 days
    Uses only ground-truth future subscription status as the regression target y.
    """
    target_window_end = cutoff_date + pd.Timedelta(days=horizon_days)
    is_terminating = (
        active_subs_df["end_date"].notna()
        & (active_subs_df["end_date"] > cutoff_date)
        & (active_subs_df["end_date"] <= target_window_end)
    )
    future_mrr = np.where(is_terminating, 0.0, active_subs_df["mrr_amount"].astype(float))
    return pd.Series(future_mrr, index=active_subs_df.index, name=SALES_TARGET_COLUMN)



def compute_account_features(
    active_subs_df: pd.DataFrame, accounts_df: pd.DataFrame, cutoff_date: pd.Timestamp
) -> pd.DataFrame:
    """
    Extracts account-level features without leaking account_name or accounts.churn_flag.
    """
    safe_account_cols = ["account_id", "industry", "country", "referral_source", "signup_date"]
    acc_sub = accounts_df[safe_account_cols].copy()

    merged = active_subs_df[["subscription_id", "account_id"]].merge(
        acc_sub, on="account_id", how="left"
    )

    merged["account_age_days"] = (cutoff_date - merged["signup_date"]).dt.days.clip(lower=0)

    return merged[["subscription_id", "industry", "country", "referral_source", "account_age_days"]]


def compute_subscription_features(
    active_subs_df: pd.DataFrame, cutoff_date: pd.Timestamp
) -> pd.DataFrame:
    """
    Extracts subscription contract parameters known at cutoff_date.
    Explicitly excludes end_date, subscriptions.churn_flag, and static upgrade/downgrade flags.
    """
    df = active_subs_df.copy()
    df["active_tenure_days"] = (cutoff_date - df["start_date"]).dt.days.clip(lower=0)

    # Convert booleans to integer 0/1 for explicit modeling representation
    df["is_trial"] = df["is_trial"].astype(int)
    df["auto_renew_flag"] = df["auto_renew_flag"].astype(int)

    feature_cols = [
        "subscription_id",
        "account_id",
        "plan_tier",
        "seats",
        "mrr_amount",
        "arr_amount",
        "is_trial",
        "billing_frequency",
        "auto_renew_flag",
        "active_tenure_days",
    ]

    return df[feature_cols]


def compute_feature_usage_aggregations(
    active_subs_df: pd.DataFrame,
    feature_usage_df: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    window_days: int = USAGE_WINDOW_DAYS,
) -> pd.DataFrame:
    """
    Aggregates usage records in trailing 30-day window and calculates 14d velocity.
    Enforces subscription.start_date <= usage_date <= cutoff_date.
    """
    window_start_30d = cutoff_date - pd.Timedelta(days=window_days)
    window_14d_recent_start = cutoff_date - pd.Timedelta(days=14)
    window_14d_prior_start = cutoff_date - pd.Timedelta(days=28)

    # Pre-filter candidate usage records
    valid_usage = feature_usage_df[
        (feature_usage_df["usage_date"] <= cutoff_date)
        & (feature_usage_df["usage_date"] >= window_14d_prior_start)
    ].copy()

    # Join start_date from active_subs
    sub_starts = active_subs_df[["subscription_id", "start_date"]].set_index("subscription_id")
    valid_usage = valid_usage.join(sub_starts, on="subscription_id", how="inner")

    # Enforce usage_date >= subscription.start_date
    valid_usage = valid_usage[valid_usage["usage_date"] >= valid_usage["start_date"]].copy()

    # 30-day window slice
    usage_30d = valid_usage[valid_usage["usage_date"] >= window_start_30d]

    # Group aggregations for 30d window
    agg_30d = usage_30d.groupby("subscription_id").agg(
        usage_count_30d=("usage_count", "sum"),
        usage_duration_secs_30d=("usage_duration_secs", "sum"),
        error_count_30d=("error_count", "sum"),
        unique_features_30d=("feature_name", "nunique"),
        beta_usage_count=("usage_count", lambda s: s[usage_30d.loc[s.index, "is_beta_feature"]].sum()),
    )

    agg_30d["beta_feature_ratio_30d"] = np.where(
        agg_30d["usage_count_30d"] > 0,
        agg_30d["beta_usage_count"] / agg_30d["usage_count_30d"],
        0.0,
    )
    agg_30d = agg_30d.drop(columns=["beta_usage_count"])

    # 14d Recent vs Prior 14d Velocity
    recent_14d = valid_usage[valid_usage["usage_date"] > window_14d_recent_start]
    recent_agg = recent_14d.groupby("subscription_id")["usage_count"].sum().rename("recent_usage_14d")

    prior_14d = valid_usage[
        (valid_usage["usage_date"] > window_14d_prior_start)
        & (valid_usage["usage_date"] <= window_14d_recent_start)
    ]
    prior_agg = prior_14d.groupby("subscription_id")["usage_count"].sum().rename("prior_usage_14d")

    # Align with active subscriptions
    res = active_subs_df[["subscription_id"]].set_index("subscription_id")
    res = res.join(agg_30d).join(recent_agg).join(prior_agg).fillna(0.0)

    # Safe velocity formula: (recent - prior) / (prior + 1.0)
    res["usage_growth_14d_vs_prior14d"] = (
        res["recent_usage_14d"] - res["prior_usage_14d"]
    ) / (res["prior_usage_14d"] + 1.0)

    cols_to_keep = [
        "usage_count_30d",
        "usage_duration_secs_30d",
        "error_count_30d",
        "unique_features_30d",
        "beta_feature_ratio_30d",
        "usage_growth_14d_vs_prior14d",
    ]

    return res[cols_to_keep].reset_index()


def compute_support_ticket_aggregations(
    active_subs_df: pd.DataFrame,
    accounts_df: pd.DataFrame,
    support_tickets_df: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    window_days: int = SUPPORT_WINDOW_DAYS,
) -> pd.DataFrame:
    """
    Aggregates support ticket activity in trailing 60-day window per account.
    Strictly guards against post-cutoff satisfaction scores, first response times, and resolution times.
    """
    window_start_60d = cutoff_date - pd.Timedelta(days=window_days)

    # Candidate tickets submitted in 60d window on or before cutoff
    valid_tickets = support_tickets_df[
        (support_tickets_df["submitted_at"] <= cutoff_date)
        & (support_tickets_df["submitted_at"] >= window_start_60d)
    ].copy()

    # Join account signup_date
    acc_signups = accounts_df[["account_id", "signup_date"]].set_index("account_id")
    valid_tickets = valid_tickets.join(acc_signups, on="account_id", how="inner")

    # Enforce submitted_at >= account.signup_date
    valid_tickets = valid_tickets[valid_tickets["submitted_at"] >= valid_tickets["signup_date"]].copy()

    # Priority condition
    valid_tickets["is_urgent"] = valid_tickets["priority"].str.lower().isin(["urgent", "high"])

    # Resolution time guard: only consider resolution if closed_at <= cutoff_date
    valid_tickets["safe_resolution_time"] = np.where(
        valid_tickets["closed_at"] <= cutoff_date,
        valid_tickets["resolution_time_hours"],
        np.nan,
    )

    # Satisfaction score guard: satisfaction is submitted upon/after closure; must only count if closed_at <= cutoff_date
    valid_tickets["safe_satisfaction_score"] = np.where(
        valid_tickets["closed_at"] <= cutoff_date,
        valid_tickets["satisfaction_score"],
        np.nan,
    )

    # First response time guard: only consider if response actually occurred on or before cutoff_date
    first_response_timestamp = valid_tickets["submitted_at"] + pd.to_timedelta(
        valid_tickets["first_response_time_minutes"], unit="m"
    )
    valid_tickets["safe_first_response_time"] = np.where(
        first_response_timestamp <= cutoff_date,
        valid_tickets["first_response_time_minutes"],
        np.nan,
    )

    # Group by account_id
    ticket_aggs = valid_tickets.groupby("account_id").agg(
        ticket_count_60d=("ticket_id", "count"),
        urgent_ticket_count_60d=("is_urgent", "sum"),
        escalation_count_60d=("escalation_flag", "sum"),
        avg_resolution_time_hours_60d=("safe_resolution_time", "mean"),
        avg_first_response_time_minutes_60d=("safe_first_response_time", "mean"),
        avg_satisfaction_score_60d=("safe_satisfaction_score", "mean"),
        satisfaction_response_count_60d=("safe_satisfaction_score", "count"),
    )

    # Map to active subscriptions
    sub_accounts = active_subs_df[["subscription_id", "account_id"]]
    merged = sub_accounts.merge(ticket_aggs, on="account_id", how="left")

    fill_zeros = [
        "ticket_count_60d",
        "urgent_ticket_count_60d",
        "escalation_count_60d",
        "avg_resolution_time_hours_60d",
        "avg_first_response_time_minutes_60d",
        "avg_satisfaction_score_60d",
        "satisfaction_response_count_60d",
    ]
    merged[fill_zeros] = merged[fill_zeros].fillna(0.0)

    cols_to_keep = ["subscription_id"] + fill_zeros
    return merged[cols_to_keep]


def build_monthly_snapshot(
    accounts_df: pd.DataFrame,
    subscriptions_df: pd.DataFrame,
    feature_usage_df: pd.DataFrame,
    support_tickets_df: pd.DataFrame,
    cutoff_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Assembles a complete, leakage-free modeling snapshot for a single cutoff date.
    """
    active_subs = filter_active_subscriptions(subscriptions_df, cutoff_date)
    if active_subs.empty:
        return pd.DataFrame()

    # 1. Target calculations
    churn_target = calculate_target(active_subs, cutoff_date)
    sales_target = calculate_forward_sales_target(active_subs, cutoff_date)

    # 2. Subscriptions features
    sub_feats = compute_subscription_features(active_subs, cutoff_date)

    # 3. Account features
    acc_feats = compute_account_features(active_subs, accounts_df, cutoff_date)

    # 4. Usage features
    usage_feats = compute_feature_usage_aggregations(active_subs, feature_usage_df, cutoff_date)

    # 5. Support features
    support_feats = compute_support_ticket_aggregations(
        active_subs, accounts_df, support_tickets_df, cutoff_date
    )

    # Merge all feature tables on subscription_id
    snapshot = (
        sub_feats.merge(acc_feats, on="subscription_id", how="inner")
        .merge(usage_feats, on="subscription_id", how="inner")
        .merge(support_feats, on="subscription_id", how="inner")
    )

    snapshot["cutoff_date"] = cutoff_date.strftime("%Y-%m-%d")
    snapshot[TARGET_COLUMN] = churn_target.values
    snapshot[SALES_TARGET_COLUMN] = sales_target.values

    # Reorder columns: Identifiers + Features + Targets
    ordered_cols = (
        IDENTIFIER_COLUMNS + MODEL_FEATURE_COLUMNS + [TARGET_COLUMN, SALES_TARGET_COLUMN]
    )
    return snapshot[ordered_cols]


def audit_leakage(df: pd.DataFrame) -> None:
    """
    Audits the generated snapshot DataFrame to guarantee no leakage columns exist
    among the model features.
    """
    feature_columns = set(MODEL_FEATURE_COLUMNS)
    df_columns = set(df.columns)

    for prohibited in LEAKAGE_COLUMNS:
        if prohibited in feature_columns:
            raise ValueError(f"Leakage Audit FAILED: '{prohibited}' is present in MODEL_FEATURE_COLUMNS.")
        if (
            prohibited in df_columns
            and prohibited not in IDENTIFIER_COLUMNS
            and prohibited not in [TARGET_COLUMN, SALES_TARGET_COLUMN]
        ):
            raise ValueError(f"Leakage Audit FAILED: Prohibited column '{prohibited}' found in dataset.")

    if "churn_flag" in df_columns:
        raise ValueError("Leakage Audit FAILED: 'churn_flag' is present in the dataset.")
    if "end_date" in df_columns:
        raise ValueError("Leakage Audit FAILED: 'end_date' is present in the dataset.")
    if "upgrade_flag" in feature_columns or "downgrade_flag" in feature_columns:
        raise ValueError("Leakage Audit FAILED: upgrade_flag/downgrade_flag found in feature columns.")


def build_modeling_dataset(
    accounts_df: Optional[pd.DataFrame] = None,
    subscriptions_df: Optional[pd.DataFrame] = None,
    feature_usage_df: Optional[pd.DataFrame] = None,
    support_tickets_df: Optional[pd.DataFrame] = None,
    start_date: str = SNAPSHOT_START,
    end_date: str = SNAPSHOT_END,
) -> pd.DataFrame:
    """
    Constructs the complete multi-cohort snapshot modeling DataFrame across all month-end cutoffs.
    """
    if any(df is None for df in [accounts_df, subscriptions_df, feature_usage_df, support_tickets_df]):
        accounts_df, subscriptions_df, feature_usage_df, support_tickets_df = load_raw_data()

    cutoffs = get_month_end_cutoffs(start_date, end_date)
    snapshot_frames: List[pd.DataFrame] = []

    for cutoff in cutoffs:
        snapshot = build_monthly_snapshot(
            accounts_df=accounts_df,
            subscriptions_df=subscriptions_df,
            feature_usage_df=feature_usage_df,
            support_tickets_df=support_tickets_df,
            cutoff_date=cutoff,
        )
        if not snapshot.empty:
            snapshot_frames.append(snapshot)

    if not snapshot_frames:
        return pd.DataFrame()

    full_dataset = pd.concat(snapshot_frames, ignore_index=True)

    # Perform leakage audit
    audit_leakage(full_dataset)

    return full_dataset


def get_preprocessor() -> ColumnTransformer:
    """
    Creates and returns an unfitted scikit-learn ColumnTransformer configured
    for numerical standard scaling, categorical one-hot encoding, and boolean passthrough.

    IMPORTANT: This preprocessor is NOT fitted during data construction.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("bool", "passthrough", BOOLEAN_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def get_sales_preprocessor() -> ColumnTransformer:
    """
    Creates and returns an unfitted scikit-learn ColumnTransformer configured
    specifically for the forward Sales/MRR Forecaster.
    Excludes current contract revenue features (mrr_amount, arr_amount) to prevent
    identity-shortcut artifacts while predicting future 30-day MRR.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), SALES_NUMERICAL_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("bool", "passthrough", BOOLEAN_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor



def save_processed_data(
    df: pd.DataFrame, output_path: Path = PROCESSED_SNAPSHOTS_CSV
) -> Path:
    """
    Saves the processed modeling DataFrame to data/processed/.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def run_pipeline() -> pd.DataFrame:
    """
    Executes the full pipeline: loads raw data, generates snapshot dataset,
    audits leakage, and saves the output to data/processed/.
    """
    df = build_modeling_dataset()
    save_processed_data(df)
    return df


if __name__ == "__main__":
    df = run_pipeline()
    print(f"Pipeline executed successfully. Generated {len(df):,} snapshot rows.")
