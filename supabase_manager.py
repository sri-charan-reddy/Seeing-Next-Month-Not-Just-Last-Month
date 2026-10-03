"""
Enterprise Cloud Data Persistence & Monitoring Layer for Supabase (PostgreSQL)
Microsoft Hackathon - Member 3: Cloud Data Persistence & Monitoring Layer
Handles model governance logging, drift metric tracking, and batched customer predictions ingestion.
"""

import os
import sys
import logging
import uuid
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Try importing Supabase PostgREST client and psycopg2
try:
    from supabase import create_client, Client
    HAS_SUPABASE = True
except ImportError:
    HAS_SUPABASE = False

try:
    import psycopg2
    from psycopg2.extras import execute_values
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

# Configure structured enterprise logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("TelemetryLogger")


class TelemetryLogger:
    """
    Enterprise telemetry and predictions logging client for Supabase PostgreSQL.
    Supports both Supabase REST API (via supabase-py) and direct PostgreSQL connection pooling (via psycopg2).
    """

    REQUIRED_TELEMETRY_FIELDS = {
        "model_version",
        "churn_auc",
        "sales_mape",
        "sales_rmse",
        "psi_score",
        "test_sample_count"
    }

    REQUIRED_PREDICTION_COLUMNS = {
        "customer_id",
        "churn_probability",
        "predicted_sales",
        "sales_lower_bound",
        "sales_upper_bound",
        "prescriptive_action",
        "prescribed_risk_drop"
    }

    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        database_url: Optional[str] = None,
        env_file: Optional[str] = None
    ):
        """
        Initialize the TelemetryLogger with credentials from arguments or environment variables.

        Args:
            supabase_url: Supabase Project URL (e.g., https://xyz.supabase.co)
            supabase_key: Supabase Service Role Key (recommended for backend ingestion) or anon key
            database_url: Optional direct PostgreSQL URI (e.g., postgresql://postgres:pw@host:5432/postgres)
            env_file: Path to custom .env file if not located at default root
        """
        # Load environment variables
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()

        self.supabase_url = (
            supabase_url 
            or os.getenv("SUPABASE_URL") 
            or os.getenv("NEXT_PUBLIC_SUPABASE_URL")
        )
        self.supabase_key = (
            supabase_key 
            or os.getenv("SUPABASE_SERVICE_KEY") 
            or os.getenv("SUPABASE_SERVICE_ROLE_KEY") 
            or os.getenv("SUPABASE_KEY")
        )
        self.database_url = (
            database_url 
            or os.getenv("DATABASE_URL") 
            or os.getenv("SUPABASE_DB_URL")
        )

        self.client: Optional[Any] = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the Supabase client if credentials are provided."""
        if not HAS_SUPABASE:
            logger.warning("Package 'supabase' is not installed. Rest API client unavailable.")
            return

        if self.supabase_url and self.supabase_key:
            try:
                self.client = create_client(self.supabase_url, self.supabase_key)
                logger.info(f"Initialized Supabase client for endpoint: {self.supabase_url}")
            except Exception as e:
                logger.error(f"Failed to initialize Supabase client: {str(e)}")
                self.client = None
        else:
            logger.warning(
                "SUPABASE_URL or SUPABASE_SERVICE_KEY not provided. "
                "Ensure credentials are set in .env or constructor."
            )

    def verify_connection(self) -> Dict[str, Any]:
        """
        Performs a health check verifying connectivity to Supabase.
        Returns a dictionary with status details and latency.
        """
        status: Dict[str, Any] = {
            "status": "unhealthy",
            "rest_api_connected": False,
            "direct_postgres_connected": False,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "errors": []
        }

        # 1. Test Supabase PostgREST API
        if self.client:
            try:
                start_time = datetime.now()
                # Query limit 1 on model_telemetry or health endpoint
                res = self.client.table("model_telemetry").select("run_id").limit(1).execute()
                latency_ms = (datetime.now() - start_time).total_seconds() * 1000
                status["rest_api_connected"] = True
                status["rest_latency_ms"] = round(latency_ms, 2)
                logger.info(f"Supabase REST API check SUCCESS ({latency_ms:.1f}ms)")
            except Exception as e:
                error_msg = f"Supabase REST API health-check failed: {str(e)}"
                logger.error(error_msg)
                status["errors"].append(error_msg)

        # 2. Test direct PostgreSQL connection if configured
        if HAS_PSYCOPG2 and self.database_url:
            try:
                start_time = datetime.now()
                conn = psycopg2.connect(self.database_url, connect_timeout=5)
                cursor = conn.cursor()
                cursor.execute("SELECT 1;")
                cursor.fetchone()
                cursor.close()
                conn.close()
                latency_ms = (datetime.now() - start_time).total_seconds() * 1000
                status["direct_postgres_connected"] = True
                status["postgres_latency_ms"] = round(latency_ms, 2)
                logger.info(f"Direct PostgreSQL check SUCCESS ({latency_ms:.1f}ms)")
            except Exception as e:
                error_msg = f"Direct PostgreSQL connection failed: {str(e)}"
                logger.warning(error_msg)
                status["errors"].append(error_msg)

        # Overall status
        if status["rest_api_connected"] or status["direct_postgres_connected"]:
            status["status"] = "healthy"
        else:
            if not self.client and not self.database_url:
                status["errors"].append("No credentials provided for either REST API or Direct DB.")

        return status

    def log_run_telemetry(self, metrics_dict: Dict[str, Any]) -> str:
        """
        Inserts a run-level record into `model_telemetry`.
        Calculates drift_flag automatically if not supplied (true if psi_score > 0.20).

        Args:
            metrics_dict: Dictionary containing governance metrics:
                - model_version (str)
                - churn_auc (float)
                - sales_mape (float)
                - sales_rmse (float)
                - psi_score (float)
                - test_sample_count (int)
                - optional: run_id, notes, drift_flag

        Returns:
            str: run_id of the registered telemetry record.
        """
        # Validate required fields
        missing_fields = self.REQUIRED_TELEMETRY_FIELDS - set(metrics_dict.keys())
        if missing_fields:
            raise ValueError(f"Missing required telemetry metrics: {missing_fields}")

        # Ensure run_id exists
        run_id = str(metrics_dict.get("run_id") or uuid.uuid4())
        psi_score = float(metrics_dict["psi_score"])
        drift_flag = metrics_dict.get("drift_flag", psi_score > 0.20)

        record = {
            "run_id": run_id,
            "model_version": str(metrics_dict["model_version"]),
            "churn_auc": round(float(metrics_dict["churn_auc"]), 4),
            "sales_mape": round(float(metrics_dict["sales_mape"]), 4),
            "sales_rmse": round(float(metrics_dict["sales_rmse"]), 2),
            "psi_score": round(psi_score, 4),
            "drift_flag": bool(drift_flag),
            "test_sample_count": int(metrics_dict["test_sample_count"]),
            "notes": metrics_dict.get("notes", "Automated ingestion via TelemetryLogger"),
            "created_at": metrics_dict.get("created_at", datetime.now(timezone.utc).isoformat())
        }

        logger.info(
            f"Logging telemetry for run_id='{run_id}', version='{record['model_version']}', "
            f"AUC={record['churn_auc']}, MAPE={record['sales_mape']}, PSI={record['psi_score']} "
            f"(Drift Flag: {record['drift_flag']})"
        )

        # 1. Primary insertion via Supabase client
        if self.client:
            try:
                response = self.client.table("model_telemetry").insert(record).execute()
                if response.data and len(response.data) > 0:
                    logger.info(f"Successfully committed run_id='{run_id}' to model_telemetry via REST API.")
                    return run_id
                else:
                    logger.warning("Supabase insert returned empty response data; falling back if possible.")
            except Exception as e:
                logger.error(f"Error inserting model_telemetry via Supabase REST: {str(e)}")
                if not (HAS_PSYCOPG2 and self.database_url):
                    raise

        # 2. Fallback / Direct insertion via psycopg2
        if HAS_PSYCOPG2 and self.database_url:
            try:
                conn = psycopg2.connect(self.database_url)
                with conn.cursor() as cur:
                    insert_query = """
                        INSERT INTO model_telemetry 
                        (run_id, model_version, churn_auc, sales_mape, sales_rmse, psi_score, drift_flag, test_sample_count, notes, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (run_id) DO UPDATE SET
                            churn_auc = EXCLUDED.churn_auc,
                            sales_mape = EXCLUDED.sales_mape,
                            sales_rmse = EXCLUDED.sales_rmse,
                            psi_score = EXCLUDED.psi_score,
                            drift_flag = EXCLUDED.drift_flag;
                    """
                    cur.execute(insert_query, (
                        record["run_id"],
                        record["model_version"],
                        record["churn_auc"],
                        record["sales_mape"],
                        record["sales_rmse"],
                        record["psi_score"],
                        record["drift_flag"],
                        record["test_sample_count"],
                        record["notes"],
                        record["created_at"]
                    ))
                conn.commit()
                conn.close()
                logger.info(f"Successfully committed run_id='{run_id}' to model_telemetry via direct psycopg2.")
                return run_id
            except Exception as e:
                logger.error(f"Error inserting model_telemetry via psycopg2: {str(e)}")
                raise

        if not self.client and not self.database_url:
            raise RuntimeError(
                "Neither Supabase client nor DATABASE_URL is configured. "
                "Set SUPABASE_URL and SUPABASE_SERVICE_KEY in .env."
            )

        return run_id

    def log_customer_predictions(
        self,
        run_id: str,
        predictions_df: pd.DataFrame,
        batch_size: int = 500
    ) -> Dict[str, Any]:
        """
        Batch-inserts customer predictions and counterfactual prescriptive interventions.

        Args:
            run_id: Foreign key referencing the model_telemetry entry.
            predictions_df: pandas DataFrame containing inference results from Member 2.
            batch_size: Chunk size for ingestion (default: 500 records per HTTP batch).

        Returns:
            Dict containing ingestion summary (total_records, batch_count, run_id).
        """
        if predictions_df is None or predictions_df.empty:
            raise ValueError("predictions_df cannot be empty or None")

        # Validate DataFrame columns
        missing_cols = self.REQUIRED_PREDICTION_COLUMNS - set(predictions_df.columns)
        if missing_cols:
            raise ValueError(f"DataFrame missing required prediction columns: {missing_cols}")

        total_records = len(predictions_df)
        logger.info(f"Preparing batch ingestion of {total_records} customer predictions for run_id='{run_id}'...")

        # Sanitize and prepare data
        df = predictions_df.copy()
        df["run_id"] = str(run_id)

        # Handle optional columns
        if "actual_churn" not in df.columns:
            df["actual_churn"] = None
        if "actual_sales" not in df.columns:
            df["actual_sales"] = None

        # Clean NaN/None and round numerical values for PostgreSQL NUMERIC compliance
        df["churn_probability"] = df["churn_probability"].astype(float).round(4)
        df["predicted_sales"] = df["predicted_sales"].astype(float).round(2)
        df["sales_lower_bound"] = df["sales_lower_bound"].astype(float).round(2)
        df["sales_upper_bound"] = df["sales_upper_bound"].astype(float).round(2)
        df["prescribed_risk_drop"] = df["prescribed_risk_drop"].astype(float).round(4)
        df["prescriptive_action"] = df["prescriptive_action"].astype(str)
        df["customer_id"] = df["customer_id"].astype(str)

        # Convert NaN to None for SQL NULL compatibility
        df = df.replace({np.nan: None})

        # Columns to ingest
        target_columns = [
            "run_id",
            "customer_id",
            "actual_churn",
            "churn_probability",
            "actual_sales",
            "predicted_sales",
            "sales_lower_bound",
            "sales_upper_bound",
            "prescriptive_action",
            "prescribed_risk_drop"
        ]

        cleaned_records = df[target_columns].to_dict(orient="records")

        # Ingestion execution
        inserted_count = 0
        total_batches = (total_records + batch_size - 1) // batch_size

        # High-speed psycopg2 path if available
        if HAS_PSYCOPG2 and self.database_url:
            try:
                inserted_count = self._insert_via_psycopg2(cleaned_records, target_columns, batch_size)
                return {
                    "status": "success",
                    "method": "psycopg2_direct",
                    "run_id": run_id,
                    "total_records": inserted_count,
                    "batch_count": total_batches
                }
            except Exception as e:
                logger.warning(f"Direct PostgreSQL batch insert failed ({str(e)}). Falling back to REST API.")

        # REST API batch insertion
        if not self.client:
            raise RuntimeError("Supabase client is not initialized. Please verify credentials in .env.")

        for i in range(0, total_records, batch_size):
            batch = cleaned_records[i : i + batch_size]
            batch_num = (i // batch_size) + 1
            try:
                res = self.client.table("customer_predictions").insert(batch).execute()
                inserted_count += len(batch)
                logger.info(
                    f"Uploaded batch {batch_num}/{total_batches} "
                    f"({inserted_count}/{total_records} records committed)"
                )
            except Exception as e:
                logger.error(f"Failed to insert batch {batch_num} for run_id='{run_id}': {str(e)}")
                raise

        logger.info(f"Ingestion complete: {inserted_count} predictions successfully written to Supabase.")
        return {
            "status": "success",
            "method": "supabase_rest",
            "run_id": run_id,
            "total_records": inserted_count,
            "batch_count": total_batches
        }

    def _insert_via_psycopg2(
        self,
        records: List[Dict[str, Any]],
        columns: List[str],
        batch_size: int
    ) -> int:
        """High-performance direct PostgreSQL bulk insertion using execute_values."""
        conn = psycopg2.connect(self.database_url)
        inserted = 0
        try:
            with conn.cursor() as cur:
                col_names = ", ".join(columns)
                insert_query = f"INSERT INTO customer_predictions ({col_names}) VALUES %s"

                for i in range(0, len(records), batch_size):
                    batch = records[i : i + batch_size]
                    values = [[rec[col] for col in columns] for rec in batch]
                    execute_values(cur, insert_query, values, page_size=batch_size)
                    inserted += len(batch)
                    logger.info(f"[psycopg2] Bulk inserted {inserted}/{len(records)} records")

            conn.commit()
            return inserted
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


if __name__ == "__main__":
    # Self-test if run directly
    logger.info("Running TelemetryLogger connection verification...")
    logger_instance = TelemetryLogger()
    health = logger_instance.verify_connection()
    logger.info(f"Connection Health Summary: {health}")
