# Microsoft Power BI Enterprise Setup & Modeling Guide
## Customer Churn & Next-Month Sales Forecasting Platform
**Hackathon Track:** Enterprise AI & Cloud Analytics  
**Architecture Role:** Member 3 — Cloud Data Persistence, Governance, & BI Serving Layer  
**Backend:** Supabase PostgreSQL (Managed Cloud DB)  
**Target Dashboard:** Microsoft Power BI Desktop / Power BI Service  

---

## 1. Architecture Overview & Data Flow

```mermaid
graph LR
    subgraph Data & Feature Pipeline (Member 1 & 2)
        A[Raw SaaS / RavenStack Ingestion] --> B[Temporal Velocity Pipeline]
        B --> C[Calibrated Churn Classifier]
        B --> D[Ridge Sales Regressor ±1.96σ]
        C --> E[Counterfactual Recourse Engine]
    end

    subgraph Supabase Cloud Persistence (Member 3)
        E --> F[supabase_manager.py]
        D --> F
        F -->|Batch Ingestion| G[(model_telemetry)]
        F -->|Indexed Ingestion| H[(customer_predictions)]
        G --> I[Analytical Views: vw_latest_customer_predictions]
        H --> I
    end

    subgraph Microsoft Power BI Executive Dashboard
        I -->|Native PG Connector| J[DirectQuery / Import Engine]
        J --> K[Executive KPI Cards]
        J --> L[Prediction Interval Shaded Band]
        J --> M[Retention Action Priority Queue]
        J --> N[Model Drift PSI Monitor]
        J --> O[Honest Accuracy Governance Note]
    end
```

---

## 2. Supabase PostgreSQL Connection Setup in Power BI Desktop

### 2.1 Retrieving Credentials from Supabase
1. Log in to your **Supabase Dashboard** (`https://supabase.com/dashboard`).
2. Navigate to your project -> **Project Settings** (gear icon) -> **Database**.
3. Under **Connection parameters**, locate:
   - **Host:** `db.<project-ref>.supabase.co` (or pooler: `aws-0-<region>.pooler.supabase.com`)
   - **Database name:** `postgres`
   - **Port:** `5432` (Direct connection) or `6543` (Connection Pooler / Transaction mode)
   - **User:** `postgres` (or `postgres.<project-ref>` when using the pooler)
   - **Password:** The database password configured during Supabase project creation.

---

### 2.2 Connecting in Power BI Desktop
1. Open **Power BI Desktop**.
2. Click **Get Data** -> Select **PostgreSQL database** -> Click **Connect**.
3. In the PostgreSQL database dialog:
   - **Server:** Enter the Host followed by the port:
     ```text
     db.<project-ref>.supabase.co:5432
     ```
     *(If connecting via pooler, enter `aws-0-<region>.pooler.supabase.com:6543`)*
   - **Database:** `postgres`
   - **Data Connectivity mode:** Choose **Import** (recommended for high interactive dashboard responsiveness) or **DirectQuery** (for real-time streaming inference).
   - Under **Advanced options**, leave SQL statement blank to import tables/views natively, or select `public.vw_latest_customer_predictions`.
4. In the authentication prompt:
   - Select **Database** tab on the left.
   - **User name:** `postgres` (or `postgres.<project-ref>` for pooler)
   - **Password:** `<your-database-password>`
   - **Encryption:** Power BI defaults to SSL (Supabase requires SSL). Click **OK** / **Connect**.
   - If prompted with an untrusted certificate warning: Click **OK** (Supabase uses cloud-signed SSL).

---

### 2.3 Selecting Tables & Views in Navigator
In the **Navigator** window, check the following objects:
1. `model_telemetry` (governance metrics and drift history)
2. `customer_predictions` (granular inferences)
3. `vw_latest_customer_predictions` *(Pre-joined view for rapid dashboard builds)*

Click **Transform Data** (Power Query) or **Load**.

---

## 3. Data Model & Relationships (Star Schema)

When importing both `model_telemetry` and `customer_predictions`, establish the relationship in the **Model View**:

| Primary Table (Dimension / Header) | Foreign Key Table (Fact) | Join Column | Cardinality | Cross Filter Direction |
| :--- | :--- | :--- | :--- | :--- |
| `model_telemetry` | `customer_predictions` | `run_id` | **1 to Many (1:*)** | **Single (`model_telemetry` filters `customer_predictions`)** |

> **Pro-Tip:** If using `vw_latest_customer_predictions`, the telemetry metadata is already denormalized into every row of the latest run, simplifying DAX and boosting DirectQuery performance.

---

## 4. DAX Measures Specification

Create a dedicated measure table named `_Measures` in Power BI:
`Modeling` -> `New Table` -> `_Measures = {0}`.

Add the following DAX calculations:

### Measure 1: Total Expected Next-Month MRR
Calculates the aggregate point forecast for next-month billing across the evaluated customer base.
```dax
Total Expected Next-Month MRR = 
SUM(customer_predictions[predicted_sales])
```
- **Format:** Currency (`$#,##0.00`)

---

### Measure 2: Revenue In Operational Risk Queue
Aggregates the forward MRR forecast for customers in the training-derived Top-10% operational risk queue.
```dax
Revenue In Operational Risk Queue = 
CALCULATE(
    SUM(customer_predictions[predicted_sales]), 
    customer_predictions[churn_probability] >= MAX(model_telemetry[risk_threshold])
)
```
- **Format:** Currency (`$#,##0.00`)

---

### Measure 3: Total Recoverable Revenue
Quantifies forward MRR in the operational risk queue where counterfactual intervention achieves genuine risk reduction.
```dax
Total Recoverable Revenue = 
CALCULATE(
    SUM(customer_predictions[predicted_sales]), 
    customer_predictions[churn_probability] >= MAX(model_telemetry[risk_threshold]), 
    customer_predictions[prescribed_risk_drop] < customer_predictions[churn_probability]
)
```
- **Format:** Currency (`$#,##0.00`)

---

### Measure 4: Drift Status & Calendar Aging Disclosure
Dynamic operational alert card indicating model stability and calendar aging status.
```dax
Drift Status Text = 
IF(
    SELECTEDVALUE(model_telemetry[drift_flag], FALSE), 
    "⚠️ CRITICAL DRIFT DETECTED", 
    "✅ MODELS STABLE (Overall PSI < 0.20; Calendar Aging Documented)"
)
```
- **Conditional Formatting:**
  - Font Color: Red (`#DC2626`) if overall drift detected, Emerald Green (`#059669`) if stable.

---

### Additional Executive DAX Measures

#### Measure 5: Operational Queue Selection Rate (%)
```dax
Queue Selection Rate % = 
DIVIDE(
    CALCULATE(COUNTROWS(customer_predictions), customer_predictions[churn_probability] >= MAX(model_telemetry[risk_threshold])),
    COUNTROWS(customer_predictions),
    0
)
```
- **Format:** Percentage (`0.0%`)

#### Measure 6: Average Prediction Interval Width ($ Spread)
```dax
Avg Interval Spread = 
AVERAGEX(
    customer_predictions, 
    customer_predictions[sales_upper_bound] - customer_predictions[sales_lower_bound]
)
```
- **Format:** Currency (`$#,##0.00`)

---

## 5. Executive Dashboard Visual Layout & Configuration

The executive dashboard layout is organized into 5 primary visual zones:

```
+-------------------------------------------------------------------------------------------------------------+
| [HEADER] Enterprise Churn Intelligence & Next-Month Sales Forecaster | Model: Dual Pipeline (HistGB + Ridge)|
+-------------------------------------------------------------------------------------------------------------+
| [KPI 1] Expected Next MRR       [KPI 2] Risk Queue MRR    [KPI 3] Recoverable MRR    [KPI 4] Drift & Aging  |
|        Forward 30-Day Sum             Top-10% Cohort            Actionable Drop            STABLE (PSI <0.20)|
+-------------------------------------------------------------+-----------------------------------------------+
| [VISUAL A] 95% Confidence Sales Prediction Band             | [VISUAL B] Population Stability Drift Gauge   |
|  - Line & Clustered Column Chart                            |  - Gauge Visual (PSI Target: 0.10, Max: 0.25) |
|  - Y: Lower Bound, Predicted Sales, Upper Bound             |  - Overall PSI vs Threshold 0.20              |
+-------------------------------------------------------------+-----------------------------------------------+
| [VISUAL C] Top-10% Operational Retention Priority Queue     | [VISUAL D] Mandatory Honest Accuracy Note     |
|  - Columns: Customer ID, Churn %, Forecast, Recourse, Drop  |  - Leakage-Free Point-In-Time Pipeline        |
|  - Sorted by churn_probability Descending                   |  - Expected Calendar Aging Governance         |
+-------------------------------------------------------------------------------------------------------------+
```

### Visual A: KPI Metric Cards (Top Ribbon)
- **Visual Type:** New Card Visual or Multi-row Card.
- **Card 1:** `[Total Expected Next-Month MRR]` (Forward 30-day forecast).
- **Card 2:** `[Revenue In Operational Risk Queue]` (Top-10% cohort forward MRR).
- **Card 3:** `[Total Recoverable Revenue]` (Cohort with validated simulated risk reduction).
- **Card 4:** `[Drift Status Text]` (Dynamic status with green/amber pill background).

---

### Visual B: Sales Forecast with Shaded 95% Prediction Interval Band
- **Visual Type:** **Line Chart** or **Area Chart**.
- **X-Axis:** `customer_id` (Sorted by `predicted_sales` Ascending) or Spend deciles.
- **Values:**
  1. `sales_upper_bound` (Line: Light Slate, Stroke Width: 1)
  2. `predicted_sales` (Line: Primary Microsoft Blue `#0078D4`, Stroke Width: 3)
  3. `sales_lower_bound` (Line: Light Slate, Stroke Width: 1, bounded at $0)
- **Shaded Band Technique:**
  - In Power BI Analytics pane -> Turn on **Error Bars** on `predicted_sales`:
    - Upper Bound: `sales_upper_bound`
    - Lower Bound: `sales_lower_bound`
    - Marker: Off, Bar: Off, **Shaded Area:** On (Transparency: 80%, Color: `#0078D4`).
  - This displays the empirical $\pm 1.96\sigma$ uncertainty band.

---

### Visual C: Top-10% Operational Retention Action Table
- **Visual Type:** **Table** or **Matrix**.
- **Data Source:** `vw_high_risk_retention_queue` or `customer_predictions`.
- **Columns Included:**
  1. `customer_id`
  2. `churn_probability` (Format as `0.00%`, Data Bars: Red gradient)
  3. `predicted_sales` (Next-month MRR point forecast, `$#,##0.00`)
  4. `prescriptive_action` (Full counterfactual prescription text, e.g., *"Switch to Annual Billing Plan with 15% Discount (Simulated Risk: 1.84% -> 1.02%)"*)
  5. `prescribed_risk_drop` (Post-intervention simulated risk, `0.00%`)
- **Filters on Visual:** Filtered to the Top-10% operational risk queue (`churn_probability >= risk_threshold`).
- **Default Sort:** `churn_probability` Descending.

---

### Visual D: PSI Drift Gauge & Governance
- **Visual Type:** **Gauge Visual**.
- **Value:** `SELECTEDVALUE(model_telemetry[psi_score])`.
- **Minimum Value:** `0.00`.
- **Maximum Value:** `0.30`.
- **Target Value:** `0.10` (Safe threshold; alarm at `0.20`).
- **Conditional Color Coding:**
  - Value `< 0.10`: `#10B981` (Green — Stable)
  - Value `0.10 - 0.20`: `#F59E0B` (Amber — Moderate Shift)
  - Value `> 0.20`: `#EF4444` (Red — Critical Drift Flagged)

---

### Visual E: Mandatory Model Governance & Transparency Disclosure
Place a dedicated Card or Callout component prominently in the dashboard:

```text
========================================================================================
                      [GOVERNANCE & VALIDATION AUDIT TRAIL]
========================================================================================
* Forward Target Formulation: Regression target is NEXT 30-DAY MRR (0 if churned,
  contract MRR if retained). Current contract mrr_amount/arr_amount are strictly
  excluded from feature inputs to eliminate identity-shortcut artifacts.
* Operational Churn Policy: Base churn rate is low (~1.6%). Operational risk queue
  targets the Top-10% highest-risk cohort using a frozen threshold derived strictly
  from the training risk distribution, without touching holdout labels.
* Defensible Sales Metrics: Sales performance tracked via RMSE, MAE, Non-zero MAPE
  (calculated strictly where y > 0 to eliminate division-by-zero artifacts), and WMAPE.
* Expected Calendar Aging: Overall Population Stability Index (PSI) remains stable
  (< 0.10). Higher feature-level PSI in account_age_days and active_tenure_days is
  explicitly classified as Expected Calendar Aging rather than data corruption.
* Honest Model Reporting: Features are strictly pre-cutoff trailing point-in-time
  aggregations with 0 lookahead leakage. Model scores reflect genuine signal without
  synthetic inflation.
========================================================================================
```

---

## 6. Performance Optimization Tips for Supabase & DirectQuery

1. **Leverage the Indexed View:** Direct Power BI to `vw_latest_customer_predictions`. It automatically limits reads to the single most recent model run, leveraging the `idx_customer_predictions_run_churn` index.
2. **Turn on Query Folding:** Ensure filters (such as `churn_probability >= risk_threshold`) are applied in Power Query steps so they fold into PostgreSQL `WHERE` clauses executed server-side in Supabase.
3. **Connection Mode Recommendation:**
   - For hackathon demonstration: Use **Import Mode** with a scheduled refresh or manual refresh on trigger.
   - For real-time production: Connect via the Supabase **Connection Pooler (port 6543)** in **DirectQuery** mode.
