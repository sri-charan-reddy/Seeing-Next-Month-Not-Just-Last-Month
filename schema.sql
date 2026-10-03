CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE IF NOT EXISTS model_telemetry (
    run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_version VARCHAR(100) NOT NULL,
    churn_auc NUMERIC(5,4) NOT NULL
        CHECK (churn_auc BETWEEN 0.0000 AND 1.0000),
    sales_mape NUMERIC(5,4) NOT NULL
        CHECK (sales_mape >= 0.0000),
    sales_rmse NUMERIC(8,2) NOT NULL
        CHECK (sales_rmse >= 0.00),
    psi_score NUMERIC(5,4) NOT NULL
        CHECK (psi_score >= 0.0000),
    drift_flag BOOLEAN NOT NULL DEFAULT FALSE,
    test_sample_count INTEGER NOT NULL
        CHECK (test_sample_count > 0),
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc', now())
);

CREATE TABLE IF NOT EXISTS customer_predictions (
    id BIGSERIAL PRIMARY KEY,
    run_id UUID NOT NULL
        REFERENCES model_telemetry(run_id)
        ON DELETE CASCADE,
    customer_id VARCHAR(50) NOT NULL,
    actual_churn INTEGER
        CHECK (actual_churn IN (0, 1)),
    churn_probability NUMERIC(5,4) NOT NULL
        CHECK (churn_probability BETWEEN 0.0000 AND 1.0000),
    actual_sales NUMERIC(8,2)
        CHECK (actual_sales >= 0.00),
    predicted_sales NUMERIC(8,2) NOT NULL
        CHECK (predicted_sales >= 0.00),
    sales_lower_bound NUMERIC(8,2) NOT NULL,
    sales_upper_bound NUMERIC(8,2) NOT NULL,
    prescriptive_action TEXT NOT NULL,
    prescribed_risk_drop NUMERIC(5,4) NOT NULL
        CHECK (prescribed_risk_drop BETWEEN 0.0000 AND 1.0000),
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc', now())
);

ALTER TABLE customer_predictions
DROP CONSTRAINT IF EXISTS chk_sales_prediction_interval;

ALTER TABLE customer_predictions
ADD CONSTRAINT chk_sales_prediction_interval
CHECK (sales_upper_bound >= sales_lower_bound);

CREATE INDEX IF NOT EXISTS idx_customer_predictions_run_id
ON customer_predictions(run_id);

CREATE INDEX IF NOT EXISTS idx_customer_predictions_churn_prob
ON customer_predictions(churn_probability DESC);

CREATE INDEX IF NOT EXISTS idx_customer_predictions_customer_id
ON customer_predictions(customer_id);

CREATE INDEX IF NOT EXISTS idx_customer_predictions_run_churn
ON customer_predictions(run_id, churn_probability DESC);

CREATE INDEX IF NOT EXISTS idx_customer_predictions_created_at
ON customer_predictions(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_customer_predictions_risk_drop
ON customer_predictions(prescribed_risk_drop);

CREATE INDEX IF NOT EXISTS idx_model_telemetry_created_at
ON model_telemetry(created_at DESC);

CREATE OR REPLACE VIEW vw_latest_customer_predictions AS
WITH latest_run AS (
    SELECT
        run_id,
        model_version,
        churn_auc,
        sales_mape,
        sales_rmse,
        psi_score,
        drift_flag,
        created_at AS run_timestamp
    FROM model_telemetry
    ORDER BY created_at DESC
    LIMIT 1
)
SELECT
    cp.id,
    cp.run_id,
    lr.model_version,
    lr.churn_auc,
    lr.sales_mape,
    lr.sales_rmse,
    lr.psi_score,
    lr.drift_flag,
    lr.run_timestamp,
    cp.customer_id,
    cp.actual_churn,
    cp.churn_probability,
    CASE
        WHEN cp.churn_probability >= 0.75
            THEN 'Tier 1 - Critical Risk (>75%)'
        WHEN cp.churn_probability >= 0.50
            THEN 'Tier 2 - Elevated Risk (50-75%)'
        WHEN cp.churn_probability >= 0.25
            THEN 'Tier 3 - Moderate Risk (25-50%)'
        ELSE 'Tier 4 - Stable (<25%)'
    END AS risk_tier,
    cp.actual_sales,
    cp.predicted_sales,
    cp.sales_lower_bound,
    cp.sales_upper_bound,
    cp.sales_upper_bound - cp.sales_lower_bound AS interval_spread,
    cp.prescriptive_action,
    cp.prescribed_risk_drop,
    cp.churn_probability - cp.prescribed_risk_drop AS risk_reduction_delta,
    CASE
        WHEN cp.churn_probability >= 0.50
         AND cp.prescribed_risk_drop < 0.35
            THEN TRUE
        ELSE FALSE
    END AS is_recoverable_revenue,
    cp.created_at AS prediction_timestamp
FROM customer_predictions cp
INNER JOIN latest_run lr
    ON cp.run_id = lr.run_id;

CREATE OR REPLACE VIEW vw_model_drift_governance AS
SELECT
    run_id,
    model_version,
    churn_auc,
    sales_mape,
    sales_rmse,
    psi_score,
    drift_flag,
    CASE
        WHEN psi_score < 0.10
            THEN 'Stable (No Shift)'
        WHEN psi_score <= 0.20
            THEN 'Moderate Shift (Monitor)'
        ELSE 'Critical Drift (Action Required)'
    END AS drift_status_label,
    test_sample_count,
    created_at AS run_timestamp
FROM model_telemetry
ORDER BY created_at DESC;

CREATE OR REPLACE VIEW vw_high_risk_retention_queue AS
SELECT
    cp.customer_id,
    cp.churn_probability,
    cp.predicted_sales,
    cp.churn_probability * cp.predicted_sales AS expected_revenue_loss,
    cp.prescriptive_action,
    cp.prescribed_risk_drop,
    cp.churn_probability - cp.prescribed_risk_drop AS expected_risk_reduction,
    cp.run_id,
    cp.created_at
FROM customer_predictions cp
WHERE cp.churn_probability >= 0.50
ORDER BY cp.churn_probability * cp.predicted_sales DESC;

ALTER TABLE model_telemetry
ENABLE ROW LEVEL SECURITY;

ALTER TABLE customer_predictions
ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "service_role_model_telemetry"
ON model_telemetry;

DROP POLICY IF EXISTS "authenticated_read_model_telemetry"
ON model_telemetry;

DROP POLICY IF EXISTS "anon_read_model_telemetry"
ON model_telemetry;

DROP POLICY IF EXISTS "service_role_customer_predictions"
ON customer_predictions;

DROP POLICY IF EXISTS "authenticated_read_customer_predictions"
ON customer_predictions;

DROP POLICY IF EXISTS "anon_read_customer_predictions"
ON customer_predictions;

CREATE POLICY "service_role_model_telemetry"
ON model_telemetry
FOR ALL
TO service_role
USING (TRUE)
WITH CHECK (TRUE);

CREATE POLICY "authenticated_read_model_telemetry"
ON model_telemetry
FOR SELECT
TO authenticated
USING (TRUE);

CREATE POLICY "anon_read_model_telemetry"
ON model_telemetry
FOR SELECT
TO anon
USING (TRUE);

CREATE POLICY "service_role_customer_predictions"
ON customer_predictions
FOR ALL
TO service_role
USING (TRUE)
WITH CHECK (TRUE);

CREATE POLICY "authenticated_read_customer_predictions"
ON customer_predictions
FOR SELECT
TO authenticated
USING (TRUE);

CREATE POLICY "anon_read_customer_predictions"
ON customer_predictions
FOR SELECT
TO anon
USING (TRUE);

GRANT SELECT ON ALL TABLES IN SCHEMA public
TO anon, authenticated;

GRANT SELECT ON ALL VIEWS IN SCHEMA public
TO anon, authenticated;
