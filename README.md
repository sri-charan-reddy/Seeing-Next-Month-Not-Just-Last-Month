# Enterprise Cloud Persistence & Monitoring Layer (Member 3)
## Customer Churn & Next-Month Sales Forecasting Platform (Microsoft Hackathon)

### 📌 Architecture Summary & Role Scope
In our 3-member enterprise architecture:
- **Member 1 (Data Engineering):** Point-in-time snapshot feature engineering on raw RavenStack SaaS data, leakage prevention, and temporal velocity metrics.
- **Member 2 (ML Models & Recourse):** Dual ML modeling (Calibrated Churn Classifier, Ridge Sales Regressor with 95% prediction intervals), Counterfactual Prescriptive Recourse Engine, and Population Stability Index (PSI) drift monitoring.
- **Member 3 (This Module - Persistence & Serving):** Cloud Data Persistence & Monitoring Layer using **Supabase (PostgreSQL)**, automated pipeline ingestion scripts, and structured views/DAX specifications for the **Microsoft Power BI executive dashboard**.

---

### 🔄 Production Flow vs. Offline Testing

#### 1. Production Pipeline Flow:
```text
Raw RavenStack Data (4 CSVs)
  │
  ▼ [Member 1: src/pipeline.py]
Point-in-Time Features & Preprocessor (25 features, zero leakage)
  │
  ▼ [Member 2: src/inference.py -> run_member_2_pipeline()]
Trained Models, Uncertainty Intervals, Counterfactual Recourse & Drift
  │
  ▼ [Member 3: supabase_manager.py -> TelemetryLogger]
Supabase Cloud PostgreSQL (customer_predictions & model_telemetry)
  │
  ▼ [Member 3: powerbi_setup_guide.md]
Microsoft Power BI DirectQuery Executive Dashboard
```

#### 2. Offline Simulation Flow (`simulate_ingestion.py`):
`simulate_ingestion.py` is **STRICTLY an offline testing utility** for validating database connection pooling, table schemas, and Power BI DirectQuery mappings without requiring raw SaaS data or trained ML weights. It does **NOT** represent production model predictions.

---

## 📂 Repository File Structure

```text
├── schema.sql                         # Production-grade idempotent PostgreSQL DDL schema & views
├── supabase_manager.py                # TelemetryLogger module (supabase-py & psycopg2 dual-engine)
├── simulate_ingestion.py              # Offline holdout test ingestion simulation utility
├── powerbi_setup_guide.md             # Complete Power BI setup, DAX measures, and visual specs
├── synthetic_holdout_predictions.csv  # Pre-generated holdout inference payload for offline testing
├── .env.example                       # Environment variable template
├── requirements.txt                   # Production Python dependencies
└── README.md                          # Architecture documentation & execution guide
```

---

## ⚡ Quick Start: 3-Step Setup

### Step 1: Initialize Database Schema in Supabase
1. Open your **Supabase Dashboard** -> **SQL Editor**.
2. Copy and paste the contents of [`schema.sql`](file:///c:/Users/btash/OneDrive/Desktop/microsoft%20hackathon/schema.sql).
3. Click **Run**. This creates:
   - `model_telemetry` table (run-level governance, AUC, MAPE, RMSE, PSI score, drift flag).
   - `customer_predictions` table (customer-level predictions, intervals, counterfactual recourse).
   - B-tree indexes for high-speed Power BI queries.
   - Analytical Views: `vw_latest_customer_predictions`, `vw_model_drift_governance`, `vw_high_risk_retention_queue`.
   - Row-Level Security (RLS) policies.

### Step 2: Configure Environment Variables
Create a `.env` file in the project root (copied from [`.env.example`](file:///c:/Users/btash/OneDrive/Desktop/microsoft%20hackathon/.env.example)):
```env
SUPABASE_URL="https://<your-project-id>.supabase.co"
SUPABASE_SERVICE_KEY="<your-supabase-service-role-key>"

# Optional for ultra-fast direct pooler bulk ingestion
DATABASE_URL="postgresql://postgres:<password>@db.<your-project-id>.supabase.co:5432/postgres"
```

### Step 3: Run Ingestion or Simulation

**Option A: Real Production Pipeline Ingestion** (Requires raw RavenStack data & trained Member 2 models):
```bash
python simulate_ingestion.py --production
```
*Or call programmatically:*
```python
from src.inference import run_member_2_pipeline
from supabase_manager import TelemetryLogger

predictions_df, telemetry = run_member_2_pipeline()
logger = TelemetryLogger()
run_id = logger.log_run_telemetry(telemetry)
logger.log_customer_predictions(run_id, predictions_df)
```

**Option B: Offline Mock Simulation** (For testing database connectivity & Power BI DirectQuery mappings):
```bash
# Offline Dry-Run & CSV Export:
python simulate_ingestion.py --dry-run --export-csv

# Live Ingestion into Supabase with mock payload:
python simulate_ingestion.py
```

---

## 🔌 Integration Contract with Member 1 & 2

When Member 2 completes model evaluation, integrate the pipeline in Python with just **3 lines of code**:

```python
from supabase_manager import TelemetryLogger

# 1. Initialize Logger
logger = TelemetryLogger()

# 2. Log Model Governance Telemetry
run_id = logger.log_run_telemetry({
    "model_version": "v1.0-histgradboost",
    "churn_auc": 0.8812,
    "sales_mape": 0.0382,
    "sales_rmse": 14.65,
    "psi_score": 0.0845,
    "test_sample_count": len(test_df),
    "notes": "Calibrated dual-model with counterfactual recourse"
})

# 3. Batch-Insert Granular Inferences & Prescriptions
logger.log_customer_predictions(
    run_id=run_id,
    predictions_df=member_2_output_df, # Must contain required schema columns
    batch_size=500
)
```

### Required Columns in `predictions_df`:
- `customer_id` (`str`): Unique Customer / Account ID (e.g., `'acc_00123'`)
- `actual_churn` (`int`, optional): 0 or 1 ground truth
- `churn_probability` (`float`): Calibrated probability in $[0.0000, 1.0000]$
- `actual_sales` (`float`, optional): Actual billed amount
- `predicted_sales` (`float`): Point forecast
- `sales_lower_bound` (`float`): $\text{pred} - 1.96\sigma$
- `sales_upper_bound` (`float`): $\text{pred} + 1.96\sigma$
- `prescriptive_action` (`str`): Counterfactual recommendation text
- `prescribed_risk_drop` (`float`): Simulated risk post-intervention

---

## 📊 Power BI DirectQuery & DAX Configuration

Refer to [`powerbi_setup_guide.md`](file:///c:/Users/btash/OneDrive/Desktop/microsoft%20hackathon/powerbi_setup_guide.md) for full dashboard design specifications:

### Core DAX Measures:
1. **Total Expected Sales:**
   ```dax
   Total Expected Sales = SUM(customer_predictions[predicted_sales])
   ```
2. **Total Revenue At Risk:**
   ```dax
   Total Revenue At Risk = 
   CALCULATE(
       SUM(customer_predictions[predicted_sales]), 
       customer_predictions[churn_probability] >= 0.50
   )
   ```
3. **Total Recoverable Revenue:**
   ```dax
   Total Recoverable Revenue = 
   CALCULATE(
       SUM(customer_predictions[predicted_sales]), 
       customer_predictions[churn_probability] >= 0.50, 
       customer_predictions[prescribed_risk_drop] < 0.35
   )
   ```
4. **Drift Status Text:**
   ```dax
   Drift Status Text = 
   IF(SELECTEDVALUE(model_telemetry[drift_flag], FALSE), "⚠️ CRITICAL DRIFT DETECTED", "✅ MODELS STABLE")
   ```
