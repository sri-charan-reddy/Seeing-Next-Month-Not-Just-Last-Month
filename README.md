# "Seeing Next Month, Not Just Last Month" — Predictive Dashboard
## Microsoft Hackathon Project

### Overview
A forward-looking enterprise predictive platform providing SaaS subscription dynamics, dual ML forecasts (Calibrated Churn & Ridge MRR), counterfactual prescriptive retention recourse, and live cloud persistence for Microsoft Power BI.

---

### 📌 Architecture Summary & 3-Member Role Scope
- **Member 1 (Data Engineering):** Point-in-time snapshot feature engineering on raw RavenStack SaaS data, leakage prevention, and temporal velocity metrics.
- **Member 2 (ML Models & Recourse):** Dual ML modeling (Calibrated Churn Classifier, Ridge Sales Regressor with 95% prediction intervals), Counterfactual Prescriptive Recourse Engine, and Population Stability Index (PSI) drift monitoring.
- **Member 3 (Persistence & Serving):** Cloud Data Persistence & Monitoring Layer using **Supabase (PostgreSQL)**, automated pipeline ingestion scripts, and structured views/DAX specifications for the **Microsoft Power BI executive dashboard**.

---

### 🔄 End-to-End Production Flow

```text
RavenStack SaaS Data (data/raw/)
        │
        ▼
Member 1: Data Engineering (src/pipeline.py)
        │
        ▼
Point-in-Time Features (21 cutoffs, 25 features, zero leakage)
        │
        ▼
Member 2: ML + Recourse + Drift (src/inference.py)
        │
        ▼
Predictions / Intervals / Actions / Telemetry
        │
        ▼
Member 3: Supabase (supabase_manager.py -> PostgreSQL)
        │
        ▼
Power BI Dashboard (DirectQuery Views & Executive DAX)
```

---

### Repository File Structure

```text
Microsoft hackathon project/
│
├── data/
│   ├── raw/                           # Raw RavenStack SaaS CSVs (omitted from git for privacy)
│   └── processed/                     # Cleaned, leakage-free point-in-time snapshot tables
│
├── src/                               # Core Python Modules
│   ├── __init__.py                    # Package initializer
│   ├── config.py                      # Global paths, features, and leakage configurations
│   ├── pipeline.py                    # Member 1: Point-in-time snapshot feature engineering
│   ├── models.py                      # Member 2: Dual ML models (Calibrated Churn + Ridge Sales)
│   ├── recourse.py                    # Member 2: Counterfactual Prescriptive Recourse Engine
│   ├── drift.py                       # Member 2: Population Stability Index (PSI) drift monitoring
│   └── inference.py                   # Member 2: End-to-end inference & Member 3 schema alignment
│
├── tests/                             # Automated Test Suites (27 tests)
│   ├── test_pipeline.py               # Member 1 pipeline tests (12 tests)
│   └── test_models.py                 # Member 2 ML, recourse, and drift tests (15 tests)
│
├── outputs/                           # Generated inference outputs & model telemetry
│   ├── customer_predictions.csv
│   └── model_telemetry.json
│
├── schema.sql                         # Member 3: Supabase PostgreSQL DDL schema & views
├── supabase_manager.py                # Member 3: TelemetryLogger module (Supabase & psycopg2)
├── simulate_ingestion.py              # Member 3: Offline ingestion simulation & testing utility
├── dashboard.py                       # Member 3: Executive Streamlit monitoring dashboard
├── powerbi_setup_guide.md             # Member 3: Power BI DirectQuery DAX & visual specs
├── synthetic_holdout_predictions.csv  # Member 3: Offline test inference fixture
├── requirements.txt                   # Unified project dependencies
├── README.md                          # Comprehensive project documentation
├── .gitignore                         # Git ignore rules
├── .env.example                       # Environment template for cloud persistence
└── main.py                            # Main pipeline entry point
```

---

## ⚡ Setup & Execution

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run Complete Automated Test Suite (27 Tests)
```bash
PYTHONPATH=. pytest -v
```

### 3. Run Member 1 Feature Pipeline
When raw RavenStack data is available in `data/raw/`:
```bash
python -m src.pipeline
```

### 4. Run Member 2 ML Modeling & Inference Pipeline
```bash
python -m src.inference
```

### 5. Run Full End-to-End Pipeline
```bash
python main.py
```

### 6. Cloud Ingestion & Testing (Member 3)

**Option A: Real Production Ingestion** (Requires processed SaaS data & trained models):
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

**Option B: Offline Mock Simulation** (Validates Supabase schemas & DirectQuery views):
```bash
# Offline Dry-Run & CSV Export:
python simulate_ingestion.py --dry-run --export-csv

# Live Ingestion into Supabase with mock payload:
python simulate_ingestion.py
```

---

## 🧱 Member 1: Data Engineering, Leakage Guard & Velocity Features

- **Ingestion & Point-in-Time Cutoffs:** Constructs monthly snapshots across 21 month-end cutoffs (`2023-03-31` to `2024-11-30`).
- **Target Formulation:** `churn_next_30d` formulated strictly looking forward 30 days post-cutoff.
- **Strict Leakage Prevention:** Audits and excludes prohibited columns (`end_date`, `churn_flag`, `upgrade_flag`, `downgrade_flag`) and future-dated usage/tickets.
- **Velocity Features:** `usage_growth_14d_vs_prior14d` captures short-term vs. prior usage momentum safely without division-by-zero errors.
- **Service Depth:** Tracks `unique_features_30d` and `beta_feature_ratio_30d`.
- **25 Model Features:** Exactly 18 numerical, 5 categorical, and 2 boolean features.
- **Scikit-Learn Preprocessing:** Unfitted `ColumnTransformer` with `StandardScaler` (numerical), `OneHotEncoder(handle_unknown="ignore")` (categorical), and passthrough (boolean).

---

## 🤖 Member 2: Dual ML Models, Prescriptive Recourse & Drift Governance

- **Calibrated Churn Classifier (`ChurnModel`):**
  - Base Estimator: `HistGradientBoostingClassifier(learning_rate=0.08, max_leaf_nodes=31)`
  - Calibration: `CalibratedClassifierCV(method="sigmoid", cv=3)`
  - Output: `churn_probability` strictly bounded in $[0.0000, 1.0000]$
  - Evaluation: ROC-AUC (primary metric), precision, recall, F1, Brier score
- **Sales / MRR Regressor (`SalesModel`):**
  - Model: `Ridge(alpha=1.0)` with $L_2$ regularization predicting `mrr_amount`
  - Output: `predicted_sales` with non-negative projection ($\max(0.0, \hat{y})$)
  - Evaluation: MAPE and RMSE
- **95% Residual Prediction Intervals:**
  - Bounds: $[\max(0.0, \hat{y} - 1.96\sigma), \hat{y} + 1.96\sigma]$ based on holdout residual standard deviation $\sigma$.
- **Temporal Validation:** `temporal_train_test_split()` enforces strict chronological snapshot splitting across cutoff dates (no future snapshots in training).
- **Prescriptive Counterfactual Recourse Engine (`RecourseEngine`):**
  - Triggers for high-risk accounts ($P(\text{churn}) > 0.60$).
  - Modifies strictly mutable subscription fields (`billing_frequency`, `auto_renew_flag`, `plan_tier`, `is_trial`).
  - Preserves all immutable features (demographics, tenure, historical usage, past tickets).
  - Recalculates churn probability through the fitted pipeline targeting risk $< 0.35$.
- **Population Stability Index (PSI) Drift Monitor (`DriftMonitor`):**
  - Evaluates feature-level PSI across all 25 features between training baseline and holdout/production cohorts.
  - Governance thresholds: Stable ($< 0.10$), Moderate Shift ($0.10 - 0.20$), Critical Drift ($> 0.20 \rightarrow \text{drift\_flag} = \text{True}$).

---

## ☁️ Member 3: Cloud Persistence & Power BI Infrastructure

- **PostgreSQL DDL Schema (`schema.sql`):** Creates `model_telemetry` and `customer_predictions` tables with B-tree indexes, foreign keys, check constraints, and RLS policies.
- **Analytical Views:**
  - `vw_latest_customer_predictions`: Joins latest model run with customer-level predictions, risk tiers, and recoverable revenue flags.
  - `vw_model_drift_governance`: Tracks run-level AUC, MAPE, RMSE, PSI score, and drift labels.
  - `vw_high_risk_retention_queue`: Prioritizes accounts with $P \ge 0.50$ by expected revenue loss.
- **Telemetry Logger (`supabase_manager.py`):** Dual-engine client supporting both Supabase PostgREST API and psycopg2 direct pooler bulk ingestion.
- **Power BI DAX Specifications (`powerbi_setup_guide.md`):** Core DAX measures for Total Expected Sales, Total Revenue At Risk, Total Recoverable Revenue, and Drift Status Text.
- **Uncertainty & Disclosure Cards:** Dedicated dashboard cards for model accuracy metrics (AUC, MAPE, RMSE) and prediction interval ranges.

---

## 📊 Limitations & Real-Data Availability Disclosure

- **Data Availability:** The proprietary raw RavenStack SaaS CSV tables (`ravenstack_accounts.csv`, `ravenstack_subscriptions.csv`, `ravenstack_feature_usage.csv`, `ravenstack_support_tickets.csv`) are omitted from this repository in accordance with data privacy guidelines.
- **Performance Metrics Disclosure:** Real-data model performance metrics (e.g. production ROC-AUC, MAPE, RMSE) have **not** been computed and remain pending until the raw RavenStack dataset is provided in `data/raw/` and processed.
- **Deterministic Validation:** All 27 automated unit and integration tests are executed and validated against deterministic test fixtures conforming exactly to the Member 1 schema.
