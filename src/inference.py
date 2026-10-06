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
    SALES_FEATURE_COLUMNS,
    SALES_TARGET_COLUMN,
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
    revenue_col: Optional[str] = None,
    export_outputs: bool = True,
    model_version: str = "v1.0-histgradboost-ridge",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes the complete Member 2 ML modeling, uncertainty estimation,
    counterfactual recourse, and drift governance pipeline.

    Incorporates approved architectural updates:
    - Operational Top-10% Risk Queue derived strictly from training distribution
    - Forward 30-day MRR target (next-month revenue)
    - Sales regression features exclude current contract mrr_amount / arr_amount
    - Defensible sales evaluation metrics (RMSE, MAE, non-zero MAPE, WMAPE)
    - Recourse restricted to operational risk queue with validated risk reduction
    - PSI governance with explicit disclosure for expected calendar aging
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

    # Resolve forward sales regression target column
    target_sales_col = revenue_col
    if target_sales_col is None:
        if SALES_TARGET_COLUMN in train_set.columns:
            target_sales_col = SALES_TARGET_COLUMN
        elif "mrr_amount" in train_set.columns and TARGET_COLUMN in train_set.columns:
            # Construct forward 30d MRR target if not pre-computed:
            # 0.0 if churn occurs in next 30d, else current mrr_amount
            train_set[SALES_TARGET_COLUMN] = np.where(
                train_set[TARGET_COLUMN] == 1, 0.0, train_set["mrr_amount"].astype(float)
            )
            test_set[SALES_TARGET_COLUMN] = np.where(
                test_set[TARGET_COLUMN] == 1, 0.0, test_set["mrr_amount"].astype(float)
            )
            target_sales_col = SALES_TARGET_COLUMN
        else:
            target_sales_col = "mrr_amount"

    # 2. Churn Classification (HistGradientBoosting + CalibratedClassifierCV)
    y_train_churn = train_set[TARGET_COLUMN].astype(int)
    y_test_churn = test_set[TARGET_COLUMN].astype(int)

    churn_model = ChurnModel(random_state=42)
    churn_model.fit(train_set, y_train_churn)

    # Operational Top-10% Risk Queue threshold derived strictly from training distribution
    risk_threshold = churn_model.compute_training_risk_threshold(train_set, percentile=0.90)

    churn_metrics = churn_model.evaluate(test_set, y_test_churn)
    churn_probs = churn_model.predict_proba(test_set)

    # Operational risk queue holdout statistics
    in_risk_queue = churn_probs >= risk_threshold
    queue_count = int(in_risk_queue.sum())
    total_test_churners = int(y_test_churn.sum())
    captured_churners = int((y_test_churn[in_risk_queue] == 1).sum()) if queue_count > 0 else 0
    queue_recall = captured_churners / total_test_churners if total_test_churners > 0 else 0.0
    queue_precision = captured_churners / queue_count if queue_count > 0 else 0.0
    baseline_churn_rate = total_test_churners / len(test_set) if len(test_set) > 0 else 0.0
    queue_lift = queue_precision / baseline_churn_rate if baseline_churn_rate > 0 else 0.0

    # 3. Forward Sales / MRR Forecasting (Ridge Regression on Sales Features)
    if target_sales_col not in train_set.columns:
        raise KeyError(f"Specified revenue column '{target_sales_col}' not found in training dataset.")

    y_train_sales = train_set[target_sales_col].astype(float)
    y_test_sales = test_set[target_sales_col].astype(float)

    sales_model = SalesModel(feature_columns=SALES_FEATURE_COLUMNS, alpha=1.0, random_state=42)
    sales_model.fit(train_set, y_train_sales)

    sales_metrics = sales_model.evaluate(test_set, y_test_sales)
    predicted_sales = sales_model.predict(test_set)

    # 4. 95% Residual Prediction Intervals (+/- 1.96 * sigma)
    lower_bounds, upper_bounds, residual_std = compute_prediction_intervals(
        y_true=y_test_sales.values,
        y_pred=predicted_sales,
        z_multiplier=1.96,
    )

    # 5. Prescriptive Counterfactual Recourse Engine (Top-10% Risk Cohort)
    recourse_engine = RecourseEngine(churn_model=churn_model, risk_threshold=risk_threshold)
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

    # Identify features experiencing expected calendar aging
    calendar_aging_features = [
        f for f in ["account_age_days", "active_tenure_days"]
        if psi_by_feature.get(f, 0.0) > drift_monitor.CRITICAL_DRIFT_THRESHOLD
    ]

    notes_parts = [
        f"Dual model pipeline: HistGradientBoosting (Calibrated CV) + Ridge forward MRR regression.",
        f"Residual sigma={residual_std:.2f}. Drift status: {drift_monitor.get_drift_status_label(psi_score)}.",
        f"Top-10% risk threshold={risk_threshold:.6f} captured {captured_churners}/{total_test_churners} holdout churners (lift: {queue_lift:.2f}x).",
    ]
    if calendar_aging_features:
        notes_parts.append(
            f"Feature-level calendar aging observed in {', '.join(calendar_aging_features)}; treated as expected temporal drift."
        )
    telemetry_notes = " ".join(notes_parts)

    # 7. Member 3 Schema Output Alignment
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

    if "subscription_id" in test_set.columns and "subscription_id" not in predictions_df.columns:
        predictions_df["subscription_id"] = test_set["subscription_id"].values

    # 8. Model Governance Telemetry Payload
    model_telemetry: Dict[str, Any] = {
        "model_version": model_version,
        "churn_auc": churn_metrics["churn_auc"],
        "sales_mape": sales_metrics["sales_mape"],
        "sales_rmse": sales_metrics["sales_rmse"],
        "sales_mae": sales_metrics.get("sales_mae"),
        "sales_wmape": sales_metrics.get("sales_wmape"),
        "psi_score": psi_score,
        "drift_flag": bool(drift_flag),
        "test_sample_count": len(test_set),
        "risk_threshold": round(risk_threshold, 6),
        "risk_queue_count": queue_count,
        "risk_queue_recall": round(queue_recall, 4),
        "risk_queue_precision": round(queue_precision, 4),
        "risk_queue_lift": round(queue_lift, 2),
        "notes": telemetry_notes,
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
        member2_csv_path = OUTPUTS_DIR / "predictions_holdout_member_2.csv"
        telemetry_path = OUTPUTS_DIR / "model_telemetry.json"

        predictions_df.to_csv(csv_path, index=False)
        predictions_df.to_csv(member2_csv_path, index=False)
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
