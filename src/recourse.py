"""
Member 2: Prescriptive Counterfactual Recourse Engine
"Seeing Next Month, Not Just Last Month" - Predictive Dashboard

Generates actionable operational interventions for high-risk customers (P(churn) > 0.60).
Guarantees:
- Modifies ONLY operationally mutable subscription features
- Strictly preserves all immutable features (demographics, tenure, historical usage, past tickets)
- Recalculates churn probability through the fitted pipeline
- Verifies post-intervention risk threshold (< 0.35)
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.config import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    MODEL_FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
)


# Explicitly documented Mutable vs Immutable Feature Partitioning
MUTABLE_FEATURE_FIELDS = {
    "billing_frequency",  # Can offer switch from monthly to annual billing
    "auto_renew_flag",   # Can offer incentive to enable automated renewal
    "plan_tier",         # Can offer tier upgrade or premium bundle
    "is_trial",          # Can convert trial to paid production plan
}

IMMUTABLE_FEATURE_FIELDS = set(MODEL_FEATURE_COLUMNS) - MUTABLE_FEATURE_FIELDS


class RecourseEngine:
    """
    Counterfactual prescriptive recourse optimizer for SaaS customer retention.
    """

    def __init__(
        self,
        churn_model: Any,
        risk_threshold: float = 0.60,
        target_risk_bound: float = 0.35,
    ):
        self.churn_model = churn_model
        self.risk_threshold = risk_threshold
        self.target_risk_bound = target_risk_bound

    def _generate_candidate_interventions(
        self, row: pd.Series
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Generates candidate interventions based strictly on mutable operational fields.
        """
        candidates: List[Tuple[str, Dict[str, Any]]] = []

        curr_billing = str(row.get("billing_frequency", "")).lower()
        curr_auto_renew = int(row.get("auto_renew_flag", 0))
        curr_tier = str(row.get("plan_tier", "")).lower()
        curr_trial = int(row.get("is_trial", 0))

        # 1. Annual contract switch
        if curr_billing != "annual":
            candidates.append(
                (
                    "Switch to Annual Billing Plan with 15% Discount",
                    {"billing_frequency": "annual"},
                )
            )

        # 2. Enable automated renewal
        if curr_auto_renew == 0:
            candidates.append(
                (
                    "Enroll in Automated Renewal with Loyalty Credit",
                    {"auto_renew_flag": 1},
                )
            )

        # 3. Combined Annual Commitment + Auto-Renew
        if curr_billing != "annual" or curr_auto_renew == 0:
            candidates.append(
                (
                    "Lock-in Annual Commitment + Guaranteed Auto-Renew Support",
                    {"billing_frequency": "annual", "auto_renew_flag": 1},
                )
            )

        # 4. Plan tier upgrade with enterprise success support
        if curr_tier in ["starter", "free", "basic"]:
            candidates.append(
                (
                    "Upgrade to Professional Tier with Dedicated CS Manager",
                    {"plan_tier": "professional", "auto_renew_flag": 1},
                )
            )
        elif curr_tier in ["professional", "pro", "growth"]:
            candidates.append(
                (
                    "Upgrade to Enterprise Tier with Priority SLA & Tech Support",
                    {"plan_tier": "enterprise", "billing_frequency": "annual", "auto_renew_flag": 1},
                )
            )

        # 5. Convert trial
        if curr_trial == 1:
            candidates.append(
                (
                    "Transition Trial to Paid Annual Plan with Onboarding Specialist",
                    {"is_trial": 0, "billing_frequency": "annual", "auto_renew_flag": 1},
                )
            )

        # Fallback candidate if already annual and auto-renew
        if not candidates:
            candidates.append(
                (
                    "Proactive Executive Outreach & Account Optimization Review",
                    {"auto_renew_flag": 1},
                )
            )

        return candidates

    def evaluate_customer(
        self,
        customer_row: pd.Series,
        current_probability: float,
    ) -> Tuple[str, float]:
        """
        Evaluates a single customer. If current_probability > risk_threshold,
        tests mutable counterfactual interventions and recalculates risk.
        """
        if current_probability <= self.risk_threshold:
            return "Standard Retention Monitoring (Low Risk)", current_probability

        candidates = self._generate_candidate_interventions(customer_row)
        best_action: Optional[str] = None
        best_simulated_risk: float = current_probability

        # Evaluate candidate interventions by generating counterfactual rows
        for action_name, modifications in candidates:
            # Create counterfactual row copy
            counterfactual = customer_row.copy()
            for key, val in modifications.items():
                counterfactual[key] = val

            # Verify no immutable fields were changed
            for imm in IMMUTABLE_FEATURE_FIELDS:
                if imm in customer_row:
                    assert counterfactual[imm] == customer_row[imm], (
                        f"Safety violation: immutable field {imm} modified!"
                    )

            # Re-run modified row through the fitted model
            cf_df = pd.DataFrame([counterfactual])
            new_prob = float(self.churn_model.predict_proba(cf_df)[0])

            # Select intervention that achieves the lowest risk
            if new_prob < best_simulated_risk:
                best_simulated_risk = new_prob
                best_action = action_name

        # Check if successful intervention was found meeting the target risk bound (< 0.35)
        orig_pct = int(round(current_probability * 100))
        if best_action and best_simulated_risk < self.target_risk_bound:
            new_pct = int(round(best_simulated_risk * 100))
            action_text = f"{best_action} (Simulated Risk: {orig_pct}% -> {new_pct}%)"
            return action_text, round(best_simulated_risk, 4)
        elif best_action and best_simulated_risk < current_probability:
            # Reduced risk but did not fully drop below 0.35
            new_pct = int(round(best_simulated_risk * 100))
            action_text = f"{best_action} [Partial Drop] (Simulated Risk: {orig_pct}% -> {new_pct}%)"
            return action_text, round(best_simulated_risk, 4)
        else:
            return "No validated intervention found (Risk exceeds actionable threshold)", current_probability

    def batch_generate_recourse(
        self,
        df: pd.DataFrame,
        probabilities: np.ndarray,
    ) -> Tuple[List[str], List[float]]:
        """
        Batch processes customer dataframe and generates prescriptive actions.
        """
        prescriptive_actions: List[str] = []
        prescribed_risk_drops: List[float] = []

        for idx, (_, row) in enumerate(df.iterrows()):
            prob = float(probabilities[idx])
            action, drop = self.evaluate_customer(row, prob)
            prescriptive_actions.append(action)
            prescribed_risk_drops.append(round(drop, 4))

        return prescriptive_actions, prescribed_risk_drops
