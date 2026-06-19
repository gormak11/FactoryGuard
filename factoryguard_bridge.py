"""
factoryguard_bridge.py
─────────────────────────────────────────────────────────────────────────────
Bridges EDA notebook (eda.ipynb) ML output → FactoryGuard pipeline.

HOW TO USE
──────────
# 1. Load your saved model
import joblib
model = joblib.load("factoryguard_failure_model.pkl")

# 2. Get a sample row (any single-row DataFrame with the 8 feature columns)
sample = X_test.iloc[[0]]

# 3. Build the ML output dict
from factoryguard_bridge import build_ml_output, ml_output_to_reasoning_input, run_pipeline

ml_output = build_ml_output(sample, model, X.columns, machine_id="M24")

# 4. Convert to FactoryGuard reasoning input
reasoning_input = ml_output_to_reasoning_input(ml_output, machine_context={
    "operator_present": True,
    "batch_criticality": "HIGH",   # LOW / MEDIUM / HIGH / CRITICAL
    "current_operator":  "J. Singh",
    "shift":             "Night",
})

# 5. Run the full pipeline (reasoning → governance)
from factoryguard_bridge import run_pipeline
result = run_pipeline(reasoning_input)

print(result["verdict"])           # AUTO_APPROVE | HUMAN_APPROVAL_REQUIRED | EMERGENCY_ALERT
print(result["rationale"])
print(result["flagged_sensors"])
print(result["failure_probability"])
"""

import uuid
import json
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# SENSOR CONFIG
# Normal operating ranges from the ai4i2020 dataset.
# Edit these if your factory floor has different thresholds.
# ─────────────────────────────────────────────────────────────────────────────
NORMAL_RANGES = {
    "Air_temperature_K":     [295, 305],
    "Process_temperature_K": [305, 313],
    "Rotational_speed_rpm":  [1168, 2886],
    "Torque_Nm":             [3.8, 76.6],
    "Tool_wear_min":         [0, 253],
}

SENSOR_UNITS = {
    "Air_temperature_K":     "K",
    "Process_temperature_K": "K",
    "Rotational_speed_rpm":  "RPM",
    "Torque_Nm":             "Nm",
    "Tool_wear_min":         "min",
}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — build_ml_output()
# Converts one row of model input features → structured ML output dict.
# This is the "final ML output format" with machine_id, failure_probability,
# anomaly_score, and flagged_sensors all in one place.
# ─────────────────────────────────────────────────────────────────────────────
def build_ml_output(sample_row, model, X_columns, machine_id: str = "M-UNKNOWN") -> dict:
    """
    Parameters
    ----------
    sample_row   : single-row pandas DataFrame (from X_test.iloc[[n]])
    model        : fitted XGBClassifier (loaded from factoryguard_failure_model.pkl)
    X_columns    : model.feature_names_in_ or X.columns from the notebook
    machine_id   : string identifier for this machine, e.g. "M24"

    Returns
    -------
    ml_output dict with fields:
        machine_id, failure_probability, anomaly_score,
        flagged_sensors, predicted_failure, top_features, sensor_data
    """
    failure_prob = float(model.predict_proba(sample_row)[0][1])
    predicted    = bool(model.predict(sample_row)[0])
    importances  = dict(zip(X_columns, model.feature_importances_))

    # ── flagged_sensors: any sensor outside its normal operating range ──
    flagged_sensors = []
    sensor_data     = {}

    for col, (low, high) in NORMAL_RANGES.items():
        val = float(sample_row[col].values[0])
        flagged = not (low <= val <= high)

        sensor_data[col] = {
            "value":        round(val, 3),
            "unit":         SENSOR_UNITS[col],
            "normal_range": [low, high],
            "flagged":      flagged,
        }
        if flagged:
            flagged_sensors.append(col)

    # ── top 5 features by XGBoost importance ──
    top_features = []
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1])[:5]:
        val = float(sample_row[feat].values[0])
        deviation = 0.0
        if feat in NORMAL_RANGES:
            low, high = NORMAL_RANGES[feat]
            if val > high:
                deviation = round(val - high, 2)
            elif val < low:
                deviation = round(val - low, 2)
        top_features.append({
            "feature_name":          feat,
            "feature_value":         round(val, 3),
            "importance_score":      round(float(imp), 4),
            "deviation_from_normal": deviation,
        })

    return {
        "request_id":          str(uuid.uuid4()),
        "machine_id":          machine_id,
        "failure_probability": round(failure_prob, 4),   # ← your key field
        "anomaly_score":       round(failure_prob, 4),   # ← alias used by some consumers
        "predicted_failure":   predicted,
        "flagged_sensors":     flagged_sensors,           # ← your key field
        "model_version":       "xgb-v1",
        "model_confidence":    round(failure_prob, 4),
        "top_features":        top_features,
        "sensor_data":         sensor_data,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — ml_output_to_reasoning_input()
# Converts the ml_output dict → the exact schema that make_input() /
# ReasoningAgent.analyze() expects in FactoryGuard.
# ─────────────────────────────────────────────────────────────────────────────
def ml_output_to_reasoning_input(ml_output: dict, machine_context: Optional[dict] = None) -> dict:
    """
    Parameters
    ----------
    ml_output       : dict returned by build_ml_output()
    machine_context : dict with keys like operator_present, batch_criticality,
                      current_operator, shift, etc.
                      Defaults to safe fallback values if not provided.

    Returns
    -------
    reasoning_input dict ready for ReasoningAgent.analyze()
    """
    machine_context = machine_context or {}

    sd = ml_output["sensor_data"]

    # Map our column names → FactoryGuard's expected sensor_data keys
    sensor_data_fg = {
        "air_temperature": {
            "value":        sd["Air_temperature_K"]["value"],
            "unit":         "K",
            "normal_range": NORMAL_RANGES["Air_temperature_K"],
        },
        "process_temperature": {
            "value":        sd["Process_temperature_K"]["value"],
            "unit":         "K",
            "normal_range": NORMAL_RANGES["Process_temperature_K"],
        },
        "rotational_speed": {
            "value":        sd["Rotational_speed_rpm"]["value"],
            "unit":         "RPM",
            "normal_range": NORMAL_RANGES["Rotational_speed_rpm"],
        },
        "torque": {
            "value":        sd["Torque_Nm"]["value"],
            "unit":         "Nm",
            "normal_range": NORMAL_RANGES["Torque_Nm"],
        },
        "tool_wear": {
            "value":        sd["Tool_wear_min"]["value"],
            "unit":         "min",
            "normal_range": NORMAL_RANGES["Tool_wear_min"],
        },
    }

    return {
        "request_id": ml_output["request_id"],
        "machine_id": ml_output["machine_id"],

        "ml_prediction": {
            "failure_probability": ml_output["failure_probability"],
            "predicted_failure":   ml_output["predicted_failure"],
            "model_version":       ml_output["model_version"],
            "model_confidence":    ml_output["model_confidence"],
            "top_features":        ml_output["top_features"],
        },

        "sensor_data": sensor_data_fg,

        # RAG evidence — populate from your retrieval system if available,
        # otherwise FactoryGuard's ReasoningAgent handles empty gracefully.
        "rag_evidence": {
            "retrieval_confidence": 0.0,
            "similar_incidents":    [],
            "relevant_sops":        [],
            "maintenance_history":  [],
        },

        "machine_context": {
            "machine_type":             machine_context.get("machine_type", "CNC Mill"),
            "current_production_batch": machine_context.get("current_production_batch", "UNKNOWN"),
            "batch_criticality":        machine_context.get("batch_criticality", "HIGH"),
            "last_maintenance_date":    machine_context.get("last_maintenance_date", "2026-01-01T00:00:00Z"),
            "hours_since_maintenance":  machine_context.get("hours_since_maintenance", 0),
            "current_operator":         machine_context.get("current_operator", "UNKNOWN"),
            "shift":                    machine_context.get("shift", "Day"),
            "operator_present":         machine_context.get("operator_present", True),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — run_pipeline()
# Runs reasoning_input end-to-end through ReasoningAgent → GovernanceEngine.
# Returns a flat summary dict with the fields you care most about.
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(
    reasoning_input: dict,
    policy_overrides: Optional[dict] = None,
    recurrence_count_last_24h: int = 0,
    force_mock: bool = False,
) -> dict:
    """
    Parameters
    ----------
    reasoning_input            : dict from ml_output_to_reasoning_input()
    policy_overrides           : e.g. {"force_human_review_all": True}
    recurrence_count_last_24h  : how many HIGH/CRITICAL alerts this machine
                                 triggered in the last 24 h (triggers EMERGENCY_ALERT at 3+)
    force_mock                 : True = skip Anthropic API call (demo/CI mode)

    Returns
    -------
    Flat summary dict:
        machine_id, failure_probability, anomaly_score, flagged_sensors,
        verdict, rationale, action_approved, escalation_level,
        governance_checks_passed, governance_checks_failed,
        raw_reasoning, raw_governance
    """
    from reasoning_agent.reasoning_agent import ReasoningAgent
    from governance.governance_engine import GovernanceEngine

    agent  = ReasoningAgent(force_mock=force_mock)
    engine = GovernanceEngine()

    reasoning_output  = agent.analyze(reasoning_input)
    machine_context   = reasoning_input.get("machine_context", {})
    governance_result = engine.evaluate(
        reasoning_output,
        machine_context=machine_context,
        policy_overrides=policy_overrides or {},
        recurrence_count_last_24h=recurrence_count_last_24h,
    )

    decision = governance_result["decision"]
    har      = governance_result["human_approval_request"]
    audit    = governance_result["audit_log"]

    return {
        # ── ML fields ──
        "machine_id":          reasoning_input["machine_id"],
        "failure_probability": reasoning_input["ml_prediction"]["failure_probability"],
        "anomaly_score":       reasoning_input["ml_prediction"]["failure_probability"],
        "flagged_sensors":     [
            k for k, v in reasoning_input["sensor_data"].items()
            if v["value"] < v["normal_range"][0] or v["value"] > v["normal_range"][1]
        ],

        # ── Governance verdict ──
        "verdict":         decision["verdict"],
        "rationale":       decision["decision_rationale"],
        "action_approved": decision["action_approved"],

        # ── Escalation ──
        "human_review_required": har["required"],
        "escalation_level":      har.get("escalation_level", "NONE"),
        "urgency":               har.get("urgency", "NONE"),

        # ── Audit ──
        "governance_checks_passed": audit["governance_checks_passed"],
        "governance_checks_failed": audit["governance_checks_failed"],

        # ── Raw outputs (for debugging / logging) ──
        "raw_reasoning":   reasoning_output,
        "raw_governance":  governance_result,
    }


# ─────────────────────────────────────────────────────────────────────────────
# QUICK DEMO — run this file directly to test the bridge without FactoryGuard
# python factoryguard_bridge.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pandas as pd

    print("=" * 60)
    print("FactoryGuard Bridge — self-test (mock data, no model needed)")
    print("=" * 60)

    # Simulate a failed machine row (values out of range)
    mock_row = pd.DataFrame([{
        "Air_temperature_K":     312.0,   # ← above 305 → flagged
        "Process_temperature_K": 316.0,   # ← above 313 → flagged
        "Rotational_speed_rpm":  1400.0,
        "Torque_Nm":             75.0,
        "Tool_wear_min":         240.0,
        "Type_H":                0,
        "Type_L":                1,
        "Type_M":                0,
    }])

    # Minimal mock model (replace with: model = joblib.load("factoryguard_failure_model.pkl"))
    class MockModel:
        feature_importances_ = [0.41, 0.20, 0.15, 0.12, 0.08, 0.02, 0.01, 0.01]
        def predict(self, X):       return [1]
        def predict_proba(self, X): return [[0.08, 0.92]]

    mock_model = MockModel()
    cols = list(mock_row.columns)

    ml_output = build_ml_output(mock_row, mock_model, cols, machine_id="M24")

    print("\n── ML Output ──")
    print(f"  machine_id:          {ml_output['machine_id']}")
    print(f"  failure_probability: {ml_output['failure_probability']}")
    print(f"  anomaly_score:       {ml_output['anomaly_score']}")
    print(f"  flagged_sensors:     {ml_output['flagged_sensors']}")
    print(f"  predicted_failure:   {ml_output['predicted_failure']}")

    reasoning_input = ml_output_to_reasoning_input(
        ml_output,
        machine_context={
            "operator_present":  True,
            "batch_criticality": "HIGH",
            "current_operator":  "J. Singh",
            "shift":             "Night",
        }
    )

    print("\n── reasoning_input (keys) ──")
    print(f"  {list(reasoning_input.keys())}")

    print("\n── To run the full pipeline, call: ──")
    print("  result = run_pipeline(reasoning_input)")
    print("  print(result['verdict'])")
    print("\nSelf-test complete. No FactoryGuard import needed for this check.")
