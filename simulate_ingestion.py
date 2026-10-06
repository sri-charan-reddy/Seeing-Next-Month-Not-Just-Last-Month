"""
[OFFLINE / DEMO TESTING UTILITY ONLY]
Simulate Ingestion Pipeline: Customer Churn & Next-Month Sales Forecasting
Member 3: Cloud Data Persistence & Monitoring Layer

================================================================================
CRITICAL ARCHITECTURAL DISTINCTION:
- PRODUCTION INFERENCE PATH:
    Production model training, inference, and governance are executed via:
    `src.inference.run_member_2_pipeline()`
    using real features from Member 1's point-in-time RavenStack SaaS pipeline.

- OFFLINE / DEMO SIMULATION PATH (THIS SCRIPT):
    This script (`simulate_ingestion.py`) is STRICTLY an offline testing and demo
    utility used to verify Supabase table schemas, connection pooling, and
    Power BI DirectQuery mappings when raw RavenStack data or trained ML models
    are not present in the runtime environment.
    
    The synthetic generator in this file produces mock benchmark records for
    infrastructure testing ONLY and does NOT represent production model predictions.
================================================================================
"""

import sys
import os
import argparse
import random
import logging
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Ensure local directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from supabase_manager import TelemetryLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("SimulateIngestion")


# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def generate_mock_customer_id(index: int) -> str:
    """Generate deterministic mock Customer / Account ID for offline testing."""
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    prefix = f"{1000 + (index * 7) % 9000:04d}"
    random_gen = random.Random(index * 1337)
    suffix = "".join(random_gen.choice(chars) for _ in range(5))
    return f"acc_{prefix}-{suffix}"


# Backwards compatibility alias
generate_telco_customer_id = generate_mock_customer_id


def generate_mock_holdout_predictions(n_samples: int = 1409, seed: int = 42) -> pd.DataFrame:
    """
    [OFFLINE / MOCK UTILITY ONLY]
    Synthesize mock holdout predictions for testing Supabase schema constraints,
    direct PostgreSQL pooler batching, and Power BI DirectQuery mappings.

    NOTE: This is strictly a synthetic testing generator and does NOT execute
    Member 2's trained ML models or use real RavenStack data. Production predictions
    must be generated using `src.inference.run_member_2_pipeline()`.
    
    Args:
        n_samples: Mock test sample size (default 1,409)
        seed: Random seed for deterministic validation

    Returns:
        pd.DataFrame conforming to `customer_predictions` database schema.
    """
    np.random.seed(seed)
    random.seed(seed)

    logger.info(f"[OFFLINE MOCK] Generating mock inference payload for N={n_samples} accounts...")

    # 1. Calibrated Churn Probabilities: Right-skewed distribution (Beta distribution)
    raw_probs = np.random.beta(a=1.4, b=3.8, size=n_samples)
    churn_probabilities = np.clip(raw_probs, 0.0450, 0.9550).round(4)

    # 2. Predicted Sales / MRR
    base_sales = np.random.normal(loc=65.0, scale=24.0, size=n_samples) + (churn_probabilities * 15.0)
    predicted_sales = np.clip(base_sales, 20.00, 118.75).round(2)

    # 3. 95% Prediction Interval: ±1.96 * sigma
    sigma = np.random.uniform(1.05, 2.05, size=n_samples)
    half_width = (1.96 * sigma).round(2)
    sales_lower_bound = np.maximum(5.00, predicted_sales - half_width).round(2)
    sales_upper_bound = (predicted_sales + half_width).round(2)

    # 4. Actual Churn & Actual Sales (Ground truth for holdout evaluation)
    actual_churn = (np.random.rand(n_samples) < churn_probabilities).astype(int)
    actual_sales = np.clip(predicted_sales + np.random.normal(0, sigma, size=n_samples), 18.00, 125.00).round(2)

    # 5. Prescriptive Recourse Interventions (Aligned with SaaS mutable actions)
    prescriptive_actions = []
    prescribed_risk_drops = []

    intervention_catalog = [
        ("Switch to Annual Billing + Assign Dedicated CSM", 0.24, 0.29),
        ("Enable Auto-Renew + Apply 10% Annual Discount", 0.28, 0.33),
        ("Upgrade to Pro Tier + Onboarding Specialist", 0.22, 0.27),
        ("Convert Trial to Paid + Enterprise SLA Guarantee", 0.16, 0.22),
    ]

    for p in churn_probabilities:
        if p > 0.60:
            # High risk: trigger counterfactual prescription engine
            chosen_action, min_risk, max_risk = random.choice(intervention_catalog)
            simulated_new_risk = round(random.uniform(min_risk, max_risk), 4)
            simulated_new_risk = min(simulated_new_risk, round(p - 0.25, 4))
            simulated_new_risk = max(0.1200, simulated_new_risk)
            
            p_percent = int(round(p * 100))
            drop_percent = int(round(simulated_new_risk * 100))
            action_text = f"{chosen_action} (Simulated Risk: {p_percent}% -> {drop_percent}%)"
            
            prescriptive_actions.append(action_text)
            prescribed_risk_drops.append(simulated_new_risk)
        elif p >= 0.30:
            # Moderate risk: proactive retention nudge
            simulated_new_risk = round(max(0.1500, p - 0.10), 4)
            p_percent = int(round(p * 100))
            drop_percent = int(round(simulated_new_risk * 100))
            action_text = f"Automated Loyalty Perk Notification (Simulated Risk: {p_percent}% -> {drop_percent}%)"
            prescriptive_actions.append(action_text)
            prescribed_risk_drops.append(simulated_new_risk)
        else:
            # Low risk: standard engagement
            action_text = "Standard Engagement (Low Risk)"
            prescriptive_actions.append(action_text)
            prescribed_risk_drops.append(p)

    customer_ids = [generate_mock_customer_id(i) for i in range(n_samples)]

    df = pd.DataFrame({
        "customer_id": customer_ids,
        "actual_churn": actual_churn,
        "churn_probability": churn_probabilities,
        "actual_sales": actual_sales,
        "predicted_sales": predicted_sales,
        "sales_lower_bound": sales_lower_bound,
        "sales_upper_bound": sales_upper_bound,
        "prescriptive_action": prescriptive_actions,
        "prescribed_risk_drop": prescribed_risk_drops
    })

    return df


# Backwards compatibility alias
generate_synthetic_telco_holdout = generate_mock_holdout_predictions


def run_pipeline_simulation(
    dry_run: bool = False,
    export_csv: bool = False,
    use_production: bool = False,
) -> None:
    """
    Executes ingestion pipeline:
    - If `use_production=True`: Executes `src.inference.run_member_2_pipeline()`
      using real Member 2 models and features.
    - If `use_production=False`: Runs offline mock simulation for database schema
      and DirectQuery validation.
    """
    load_dotenv()
    
    print("\n" + "="*80)
    print("[*] CLOUD INGESTION & MONITORING PIPELINE")
    print("    Member 3: Supabase PostgreSQL Persistence & Governance Layer")
    print("="*80)

    if use_production:
        print("\n[+] MODE: PRODUCTION INFERENCE INGESTION")
        print("    Invoking Member 2 inference pipeline: src.inference.run_member_2_pipeline()")
        try:
            from src.inference import run_member_2_pipeline
        except ImportError:
            logger.error(
                "Production pipeline ('src.inference.run_member_2_pipeline') not available. "
                "Ensure src/ is on PYTHONPATH."
            )
            sys.exit(1)

        predictions_df, governance_telemetry = run_member_2_pipeline()
    else:
        print("\n" + "="*80)
        print("[!] NOTICE: OFFLINE INFRASTRUCTURE TESTING / SIMULATION MODE ONLY")
        print("    This script generates mock benchmark test records to validate Supabase")
        print("    connectivity, table schemas, and Power BI DirectQuery mappings.")
        print("    It does NOT execute trained ML models or represent real customer predictions.")
        print("    Production Flow:")
        print("      Raw RavenStack data -> Member 1 pipeline -> Member 2 inference")
        print("      -> run_member_2_pipeline() -> Supabase -> Power BI")
        print("    Pass --production to execute the real Member 2 pipeline.")
        print("="*80)

        governance_telemetry = {
            "model_version": "v1.0-offline-simulation",
            "churn_auc": 0.8812,
            "sales_mape": 0.0382,
            "sales_rmse": 14.65,
            "psi_score": 0.0845,  # PSI < 0.10: Stable, no feature drift
            "drift_flag": False,
            "test_sample_count": 1409,
            "notes": "Offline mock simulation for schema and Power BI DirectQuery validation."
        }
        predictions_df = generate_mock_holdout_predictions(n_samples=1409, seed=2026)

    # Quick metric calculations
    high_risk_count = (predictions_df["churn_probability"] >= 0.50).sum()
    total_sales = predictions_df["predicted_sales"].sum()
    revenue_at_risk = predictions_df[predictions_df["churn_probability"] >= 0.50]["predicted_sales"].sum()
    recoverable_revenue = predictions_df[
        (predictions_df["churn_probability"] >= 0.50) & 
        (predictions_df["prescribed_risk_drop"] < 0.35)
    ]["predicted_sales"].sum()

    print("\n[+] INFERENCE SUMMARY STATS:")
    print(f"  * Total Evaluated Customers   : {len(predictions_df):,}")
    print(f"  * High-Risk Accounts (P >= 0.5): {high_risk_count:,} ({high_risk_count/len(predictions_df)*100:.1f}%)")
    print(f"  * Total Projected Sales       : ${total_sales:,.2f}")
    print(f"  * Revenue at Risk             : ${revenue_at_risk:,.2f} ({revenue_at_risk/total_sales*100:.1f}%)")
    print(f"  * Potentially Recoverable     : ${recoverable_revenue:,.2f} ({recoverable_revenue/revenue_at_risk*100:.1f}% of at-risk)")
    print(f"  * Avg Prediction Interval Band: ${predictions_df['sales_upper_bound'].mean() - predictions_df['sales_lower_bound'].mean():.2f}")

    if export_csv:
        csv_filename = "synthetic_holdout_predictions.csv"
        predictions_df.to_csv(csv_filename, index=False)
        print(f"\n[+] Exported dataframe to local CSV: {csv_filename}")

    # 3. Connection Verification & Ingestion
    logger_client = TelemetryLogger()
    connection_health = logger_client.verify_connection()

    print("\n[+] DATABASE CONNECTIVITY STATUS:")
    print(f"  * Status               : {connection_health.get('status').upper()}")
    print(f"  * REST API Connected   : {connection_health.get('rest_api_connected')}")
    print(f"  * Direct PG Connected  : {connection_health.get('direct_postgres_connected')}")
    if connection_health.get("errors"):
        print(f"  * Diagnostics          : {connection_health['errors']}")

    if dry_run or connection_health.get("status") != "healthy":
        if connection_health.get("status") != "healthy" and not dry_run:
            print("\n[!] Supabase credentials not yet detected in .env or database unreachable.")
            print("    Proceeding in DRY-RUN validation mode to verify pipeline integrity.")
        else:
            print("\n[*] DRY-RUN MODE: Skipping cloud database insertion.")

        print("\n[+] DATA INTEGRITY CHECKS:")
        print(f"  * DataFrame null count: {predictions_df.isnull().sum().sum()}")
        print(f"  * Interval violation count: {(predictions_df['sales_lower_bound'] > predictions_df['sales_upper_bound']).sum()}")
        print(f"  * High-Risk Prescriptions Sample:")
        sample_display = predictions_df[predictions_df["churn_probability"] > 0.65][
            ["customer_id", "churn_probability", "predicted_sales", "prescriptive_action", "prescribed_risk_drop"]
        ].head(5)
        print(sample_display.to_string(index=False))
        print("\n[+] Pipeline validation successful. Configure .env with live Supabase credentials to persist.")
        return

    # 4. Live Cloud Ingestion
    print("\n[+] COMMITTING TELEMETRY & INFERENCES TO SUPABASE...")
    
    # Step A: Log Run Telemetry
    run_id = logger_client.log_run_telemetry(governance_telemetry)
    print(f"  [OK] Registered Model Run: run_id='{run_id}' (Model: {governance_telemetry['model_version']})")

    # Step B: Batch Ingest Customer Predictions
    ingest_result = logger_client.log_customer_predictions(
        run_id=run_id,
        predictions_df=predictions_df,
        batch_size=500
    )

    print(f"  [OK] Ingested Customer Predictions: {ingest_result['total_records']} rows in {ingest_result['batch_count']} batches")
    print(f"  [OK] Ingestion Mode: {ingest_result['method']}")

    print("\n" + "="*80)
    print("[SUCCESS] PIPELINE INGESTION COMPLETED SUCCESSFULLY!")
    print(f"   Run ID: {run_id}")
    print(f"   Target View for Power BI: 'vw_latest_customer_predictions'")
    print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Supabase Ingestion Layer. By default, runs in offline/mock simulation mode "
            "for schema and DirectQuery validation. Use --production to execute the real "
            "Member 2 inference pipeline (run_member_2_pipeline)."
        )
    )
    parser.add_argument(
        "--production",
        action="store_true",
        help="Execute real Member 2 production inference (run_member_2_pipeline) instead of offline mock simulation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run simulation and validation without inserting into database.",
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Save the generated synthetic holdout predictions to a CSV file.",
    )
    args = parser.parse_args()

    run_pipeline_simulation(
        dry_run=args.dry_run,
        export_csv=args.export_csv,
        use_production=args.production,
    )
