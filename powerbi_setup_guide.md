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

### Measure 1: Total Expected Sales
Calculates the aggregate point forecast for next-month billing across the evaluated customer base.
```dax
Total Expected Sales = 
SUM(customer_predictions[predicted_sales])
```
- **Format:** Currency (`$#,##0.00`)

---

### Measure 2: Total Revenue At Risk
Aggregates the sales forecast for customers exhibiting a churn probability of 50% or higher.
```dax
Total Revenue At Risk = 
CALCULATE(
    SUM(customer_predictions[predicted_sales]), 
    customer_predictions[churn_probability] >= 0.50
)
```
- **Format:** Currency (`$#,##0.00`)

---

### Measure 3: Total Recoverable Revenue
Quantifies the portion of revenue at risk that can realistically be salvaged through counterfactual prescriptive interventions (where post-intervention simulated risk falls below 35%).
```dax
Total Recoverable Revenue = 
CALCULATE(
    SUM(customer_predictions[predicted_sales]), 
    customer_predictions[churn_probability] >= 0.50, 
    customer_predictions[prescribed_risk_drop] < 0.35
)
```
- **Format:** Currency (`$#,##0.00`)

---

### Measure 4: Drift Status Text
Dynamic operational alert card indicating whether feature drift (Population Stability Index > 0.20) has degraded model reliability.
```dax
Drift Status Text = 
IF(
    SELECTEDVALUE(model_telemetry[drift_flag], FALSE), 
    "⚠️ CRITICAL DRIFT DETECTED", 
    "✅ MODELS STABLE (PSI < 0.20)"
)
```
- **Conditional Formatting:**
  - Font Color: Red (`#DC2626`) if drift detected, Emerald Green (`#059669`) if stable.

---

### Additional Executive DAX Measures

#### Measure 5: Recoverable Revenue Ratio (%)
```dax
Recoverable Revenue Ratio = 
DIVIDE([Total Recoverable Revenue], [Total Revenue At Risk], 0)
```
- **Format:** Percentage (`0.0%`)

#### Measure 6: Churn Rate % (Projected)
```dax
Projected High-Risk Churn Rate = 
DIVIDE(
    CALCULATE(COUNTROWS(customer_predictions), customer_predictions[churn_probability] >= 0.50),
    COUNTROWS(customer_predictions),
    0
)
```
- **Format:** Percentage (`0.0%`)

#### Measure 7: Average Prediction Interval Width ($ Spread)
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
| [HEADER] Enterprise Churn Intelligence & Next-Month Sales Forecaster | Model: v1.0-histgradboost            |
+-------------------------------------------------------------------------------------------------------------+
| [KPI 1] Total Expected Sales    [KPI 2] Revenue At Risk    [KPI 3] Recoverable Revenue   [KPI 4] Drift Status|
|        $96,063.80                     $13,455.75 (14.0%)         $5,426.17 (40.3%)           MODELS STABLE   |
+-------------------------------------------------------------+-----------------------------------------------+
| [VISUAL A] 95% Confidence Sales Prediction Band             | [VISUAL B] Population Stability Drift Gauge   |
|  - Line & Clustered Column Chart                            |  - Gauge Visual (PSI Target: 0.10, Max: 0.25) |
|  - Y: Lower Bound, Predicted Sales, Upper Bound             |  - Current PSI: 0.0845 (Healthy Baseline)     |
+-------------------------------------------------------------+-----------------------------------------------+
| [VISUAL C] Counterfactual Retention Action Priority Queue   | [VISUAL D] Mandatory Honest Accuracy Note     |
|  - Columns: Customer ID, Churn %, Forecast, Recourse, Drop  |  - Verified Leakage Prevention                |
|  - Conditional Heatmap highlighting high-yield interventions|  - Temporal Split Validation Summary          |
+-------------------------------------------------------------------------------------------------------------+
```

### Visual A: KPI Metric Cards (Top Ribbon)
- **Visual Type:** New Card Visual or Multi-row Card.
- **Card 1:** `[Total Expected Sales]` (Callout value: `$96.1K`, Subtitle: "Holdout 1,409 Accounts").
- **Card 2:** `[Total Revenue At Risk]` (Callout value: `$13.5K`, Color: Soft Crimson `#E11D48`).
- **Card 3:** `[Total Recoverable Revenue]` (Callout value: `$5.4K`, Subtitle: "40.3% Recovery Potential", Color: Emerald `#10B981`).
- **Card 4:** `[Drift Status Text]` (Dynamic status with green/red pill background).

---

### Visual B: Sales Forecast with Shaded 95% Prediction Interval Band
- **Visual Type:** **Line Chart** or **Area Chart**.
- **X-Axis:** `customer_id` (Sorted by `predicted_sales` Ascending) or Sub-cohort Spend deciles.
- **Values:**
  1. `sales_upper_bound` (Line: Light Slate, Stroke Width: 1)
  2. `predicted_sales` (Line: Primary Microsoft Blue `#0078D4`, Stroke Width: 3)
  3. `sales_lower_bound` (Line: Light Slate, Stroke Width: 1)
- **Shaded Band Technique:**
  - In Power BI Analytics pane -> Turn on **Error Bars** on `predicted_sales`:
    - Upper Bound: `sales_upper_bound`
    - Lower Bound: `sales_lower_bound`
    - Marker: Off, Bar: Off, **Shaded Area:** On (Transparency: 80%, Color: `#0078D4`).
  - This displays the exact $\pm 1.96\sigma$ uncertainty band ($4-$8 interval width).

---

### Visual C: Retention Action Table (Executive Priority Queue)
- **Visual Type:** **Table** or **Matrix**.
- **Data Source:** `vw_high_risk_retention_queue` or `customer_predictions`.
- **Columns Included:**
  1. `customer_id`
  2. `churn_probability` (Format as `0.0%`, Data Bars: Red gradient)
  3. `predicted_sales` (Format as `$#,##0.00`)
  4. `prescriptive_action` (Full counterfactual prescription text, e.g., *"Switch to Annual Billing + Assign Dedicated CSM (Simulated Risk: 82% -> 27%)"*)
  5. `prescribed_risk_drop` (Format as `0.0%`, Color font: Green `#047857`)
- **Filters on Visual:** `churn_probability >= 0.50`.
- **Default Sort:** `churn_probability` Descending.

---

### Visual D: PSI Drift Gauge (Model Governance & Reliability)
- **Visual Type:** **Gauge Visual**.
- **Value:** `SELECTEDVALUE(model_telemetry[psi_score])` (e.g., `0.0845`).
- **Minimum Value:** `0.00`.
- **Maximum Value:** `0.30`.
- **Target Value:** `0.10` (Safe threshold).
- **Conditional Color Coding:**
  - Value `< 0.10`: `#10B981` (Green — No Shift)
  - Value `0.10 - 0.20`: `#F59E0B` (Amber — Moderate Shift)
  - Value `> 0.20`: `#EF4444` (Red — Critical Drift Flagged)

---

### Visual E: The Mandatory "Honest Accuracy Note" Card
Place a dedicated Text/Card component prominently in the governance section of the dashboard:

```text
========================================================================================
                      [GOVERNANCE & VALIDATION AUDIT TRAIL]
========================================================================================
* Temporal Leakage Prevention: Features engineered strictly on pre-cutoff historical
  windows. Velocity aggregations omit future billing cycles to avoid lookahead bias.
* Calibrated Probabilities: Churn likelihood calibrated via Isotonic Regression;
  raw tree probabilities mapped to real frequentist risk. Holdout AUC: 0.8812.
* Dual-Model Uncertainty: Next-month sales generated via Ridge Regressor (MAPE: 3.82%,
  RMSE: $14.65) with empirical 95% prediction intervals (±1.96*sigma spread $4 to $8).
* Drift Monitoring: Real-time Population Stability Index (PSI = 0.0845) actively
  benchmarked against training distribution. Drift threshold established at PSI = 0.20.
========================================================================================
```

---

## 6. Performance Optimization Tips for Supabase & DirectQuery

1. **Leverage the Indexed View:** Direct Power BI to `vw_latest_customer_predictions`. It automatically limits reads to the single most recent model run, leveraging the `idx_customer_predictions_run_churn` index.
2. **Turn on Query Folding:** Ensure filters (such as `churn_probability >= 0.50`) are applied in Power Query steps so they fold into PostgreSQL `WHERE` clauses executed server-side in Supabase.
3. **Connection Mode Recommendation:**
   - For hackathon demonstration: Use **Import Mode** with a scheduled refresh or manual refresh on trigger.
   - For real-time production: Connect via the Supabase **Connection Pooler (port 6543)** in **DirectQuery** mode.
