"""
Member 2: End-to-End Inference & Governance Pipeline
"Seeing Next Month, Not Just Last Month" - Predictive Dashboard

Orchestrates:
1. Temporal train/holdout dataset partitioning
2. Calibrated Churn Classifier training & ROC-AUC evaluation
3. Ridge Sales/MRR Forecaster training & MAPE/RMSE evaluation
4. 95% Residual Prediction Interval generation
5. Prescriptive Recourse Engine evaluation
6. Population Stability Index (PSI) drift monitoring
7. Member 3 schema contract compliance & artifact export
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd

from src.config import (
    MODEL_FEATURE_COLUMNS,
    OUTPUTS_DIR,
    PROCESSED_SNAPSHOTS_CSV,
    TARGET_COLUMN,
)
from src.drift import DriftMonitor
from src.models import (
    ChurnModel,
    SalesModel,
    compute_prediction_intervals,
    temporal_train_test_split,
)
from src.recourse import RecourseEngine


def run_member_2_pipeline(
    data_df: Optional[pd.DataFrame] = None,
    train_df: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None,
    revenue_col: str = "mrr_amount",
    export_outputs: bool = True,
    model_version: str = "v1.0-histgradboost-ridge",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes the complete Member 2 ML modeling, uncertainty estimation,
    counterfactual recourse, and drift governance pipeline.

    Returns:
        Tuple of (predictions_df, model_telemetry_dict) formatted strictly
        for downstream consumption by Member 3 (Supabase & Power BI).
    """
    # 1. Dataset Resolution & Temporal Partitioning
    if train_df is None or test_df is None:
        if data_df is None:
            if PROCESSED_SNAPSHOTS_CSV.exists():
                data_df = pd.read_csv(PROCESSED_SNAPSHOTS_CSV)
            else:
                raise FileNotFoundError(
                    f"Processed snapshot dataset not found at {PROCESSED_SNAPSHOTS_CSV}. "
                    "Run Member 1 pipeline first or pass data_df directly."
                )

        train_set, test_set = temporal_train_test_split(data_df, test_ratio=0.20)
    else:
        train_set = train_df.copy()
        test_set = test_df.copy()

    if train_set.empty or test_set.empty:
        raise ValueError("Train or test split is empty.")

    # 2. Churn Classification (HistGradientBoosting + CalibratedClassifierCV)
    y_train_churn = train_set[TARGET_COLUMN].astype(int)
    y_test_churn = test_set[TARGET_COLUMN].astype(int)

    churn_model = ChurnModel(random_state=42)
    churn_model.fit(train_set, y_train_churn)

    churn_metrics = churn_model.evaluate(test_set, y_test_churn)
    churn_probs = churn_model.predict_proba(test_set)

    # 3. Sales / MRR Forecasting (Ridge Regression)
    if revenue_col not in train_set.columns:
        raise KeyError(f"Specified revenue column '{revenue_col}' not found in training dataset.")

    y_train_sales = train_set[revenue_col].astype(float)
    y_test_sales = test_set[revenue_col].astype(float)

    sales_model = SalesModel(alpha=1.0, random_state=42)
    sales_model.fit(train_set, y_train_sales)

    sales_metrics = sales_model.evaluate(test_set, y_test_sales)
    predicted_sales = sales_model.predict(test_set)

    # 4. 95% Residual Prediction Intervals (+/- 1.96 * sigma)
    lower_bounds, upper_bounds, residual_std = compute_prediction_intervals(
        y_true=y_test_sales.values,
        y_pred=predicted_sales,
        z_multiplier=1.96,
    )

    # 5. Prescriptive Counterfactual Recourse Engine
    recourse_engine = RecourseEngine(churn_model=churn_model, risk_threshold=0.60)
    prescriptive_actions, prescribed_risk_drops = recourse_engine.batch_generate_recourse(
        df=test_set,
        probabilities=churn_probs,
    )

    # 6. Population Stability Index (PSI) Drift Monitoring
    drift_monitor = DriftMonitor()
    psi_score, psi_by_feature, drift_flag = drift_monitor.compute_psi(
        train_df=train_set,
        holdout_df=test_set,
    )

    # 7. Member 3 Schema Output Alignment
    # Determine customer_id from account_id or subscription_id
    if "customer_id" in test_set.columns:
        customer_ids = test_set["customer_id"].astype(str).tolist()
    elif "account_id" in test_set.columns:
        customer_ids = test_set["account_id"].astype(str).tolist()
    elif "subscription_id" in test_set.columns:
        customer_ids = test_set["subscription_id"].astype(str).tolist()
    else:
        customer_ids = [f"CUST-{i:05d}" for i in range(len(test_set))]

    predictions_df = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "actual_churn": y_test_churn.values,
            "churn_probability": np.round(churn_probs, 4),
            "actual_sales": np.round(y_test_sales.values, 2),
            "predicted_sales": np.round(predicted_sales, 2),
            "sales_lower_bound": lower_bounds,
            "sales_upper_bound": upper_bounds,
            "prescriptive_action": prescriptive_actions,
            "prescribed_risk_drop": prescribed_risk_drops,
        }
    )

    # If subscription_id is distinct from customer_id, retain it for tracking
    if "subscription_id" in test_set.columns and "subscription_id" not in predictions_df.columns:
        predictions_df["subscription_id"] = test_set["subscription_id"].values

    # 8. Model Governance Telemetry Payload
    model_telemetry: Dict[str, Any] = {
        "model_version": model_version,
        "churn_auc": churn_metrics["churn_auc"],
        "sales_mape": sales_metrics["sales_mape"],
        "sales_rmse": sales_metrics["sales_rmse"],
        "psi_score": psi_score,
        "drift_flag": bool(drift_flag),
        "test_sample_count": len(test_set),
        "notes": (
            f"Dual model pipeline: HistGradientBoosting (Calibrated CV) + Ridge regression. "
            f"Residual sigma={residual_std:.2f}. Drift status: {drift_monitor.get_drift_status_label(psi_score)}."
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "auxiliary_metrics": {
            "precision": churn_metrics["precision"],
            "recall": churn_metrics["recall"],
            "f1": churn_metrics["f1"],
            "brier_score": churn_metrics["brier_score"],
            "psi_by_feature": psi_by_feature,
        },
    }

    # 9. Artifact Serialization
    if export_outputs:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = OUTPUTS_DIR / "customer_predictions.csv"
        telemetry_path = OUTPUTS_DIR / "model_telemetry.json"

        predictions_df.to_csv(csv_path, index=False)
        with open(telemetry_path, "w") as f:
            json.dump(model_telemetry, f, indent=2)

    return predictions_df, model_telemetry


if __name__ == "__main__":
    print("[*] Running Member 2 ML Inference Pipeline...")
    try:
        preds_df, telemetry = run_member_2_pipeline()
        print(f"[+] Successfully generated inferences for {len(preds_df):,} customers.")
        print(f"    Churn ROC-AUC : {telemetry['churn_auc']:.4f}")
        print(f"    Sales MAPE    : {telemetry['sales_mape']:.4f}")
        print(f"    Sales RMSE    : ${telemetry['sales_rmse']:.2f}")
        print(f"    PSI Score     : {telemetry['psi_score']:.4f} (Drift: {telemetry['drift_flag']})")
    except Exception as exc:
        print(f"[-] Pipeline execution stopped: {exc}")
