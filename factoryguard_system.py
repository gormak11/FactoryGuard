"""
factoryguard_system.py
─────────────────────────────────────────────────────────────────────────────
Unified FactoryGuard System Orchestrator.

Connects:
1. ML Anomaly Classifier (XGBoost model file loading)
2. FAISS Vector Database Search (RAG Evidence retrieval)
3. ARIA LLM Reasoning (LLM failure mode & FMEA analysis)
4. Safety Governance Engine (Deterministic policy checks & 8 gates)

Usage:
    from factoryguard_system import run_factoryguard_system
    import pandas as pd

    # Get telemetry data
    sample_telemetry = pd.DataFrame([{
        "Air_temperature_K":     302.5,
        "Process_temperature_K": 310.2,
        "Rotational_speed_rpm":  1400.0,
        "Torque_Nm":             75.0,
        "Tool_wear_min":         240.0,
        "Type_H":                0,
        "Type_L":                1,
        "Type_M":                0,
    }])

    result = run_factoryguard_system(
        sample_telemetry,
        machine_id="M3",
        machine_context={"operator_present": True, "batch_criticality": "HIGH"}
    )
    print("Verdict:", result["decision"]["verdict"])
"""

import os
import joblib
import pandas as pd
import logging
from typing import Optional, Tuple
from datetime import datetime, timezone

# Import system modules
from factoryguard_bridge import build_ml_output, ml_output_to_reasoning_input
from reasoning_agent.reasoning_agent import ReasoningAgent
from governance.governance_engine import GovernanceEngine

# Import RAG module with flexible path mapping
try:
    from rags.rag.retrieve import retrieve_evidence
except ImportError:
    try:
        from rag.retrieve import retrieve_evidence
    except ImportError:
        logging.warning("RAG module retrieve_evidence could not be imported. Using empty stubs.")
        def retrieve_evidence(query: str, n_results: int = 2) -> dict:
            return {"incidents": [], "sops": [], "maintenance_logs": []}

logger = logging.getLogger("factoryguard.system")
logging.basicConfig(level=logging.INFO)


class SystemMockModel:
    """Fallback model if trained XGBoost model is missing."""
    feature_importances_ = [0.41, 0.20, 0.15, 0.12, 0.08, 0.02, 0.01, 0.01]
    
    def predict(self, X):
        # Predict failure if tool wear is high or temperatures are elevated
        row = X.iloc[0]
        if row.get("Tool_wear_min", 0) > 200 or row.get("Air_temperature_K", 0) > 310:
            return [1]
        return [0]
        
    def predict_proba(self, X):
        row = X.iloc[0]
        # Calculate dynamic probabilities
        prob = 0.05
        if row.get("Tool_wear_min", 0) > 200:
            prob += 0.50 + min(0.40, (row["Tool_wear_min"] - 200) / 100)
        if row.get("Air_temperature_K", 0) > 308:
            prob += 0.25
        prob = min(0.99, max(0.01, prob))
        return [[1.0 - prob, prob]]


def load_xgboost_model(model_path: str = "factoryguard_failure_model.pkl") -> Tuple[object, list]:
    """Loads the pre-trained XGBoost model or falls back to a MockModel if missing."""
    cols = [
        "Air_temperature_K", "Process_temperature_K", "Rotational_speed_rpm",
        "Torque_Nm", "Tool_wear_min", "Type_H", "Type_L", "Type_M"
    ]
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            # Try to get training columns from model
            if hasattr(model, "feature_names_in_"):
                cols = list(model.feature_names_in_)
            logger.info(f"Successfully loaded XGBoost model from '{model_path}'")
            return model, cols
        except Exception as e:
            logger.warning(f"Error loading model pickle '{model_path}': {e}. Falling back to MockModel.")
    else:
        logger.info(f"Model file '{model_path}' not found. Using MockModel for inference.")
        
    return SystemMockModel(), cols


def construct_anomaly_search_query(sensor_data: dict, machine_id: str) -> str:
    """Builds a semantic search query based on flagged telemetry anomalies."""
    flagged_sensors = [k for k, v in sensor_data.items() if v.get("flagged")]
    
    if not flagged_sensors:
        return f"Machine {machine_id} normal operation"
        
    # Heuristics to match RAG documents (vibration, overheating, bearing wear, etc.)
    if "rotational_speed" in flagged_sensors or "torque" in flagged_sensors:
        return f"Machine {machine_id} vibration anomaly spike"
    elif "air_temperature" in flagged_sensors or "process_temperature" in flagged_sensors:
        return f"Machine {machine_id} overheating temperature anomaly"
    elif "tool_wear" in flagged_sensors:
        # Check if high wear matches a bearing or wear incident
        val = sensor_data["tool_wear"]["value"]
        if val > 200:
            return f"Machine {machine_id} bearing wear tool failure"
        return f"Machine {machine_id} tool wear threshold exceeded"
        
    return f"Machine {machine_id} anomaly in " + ", ".join(flagged_sensors)


def query_rag_and_map_evidence(query: str, retrieval_confidence: float = 0.85) -> dict:
    """Retrieves context from FAISS and maps it to the Reasoning Agent input schema."""
    evidence = retrieve_evidence(query, n_results=2)
    
    similar_incidents = []
    for inc in evidence.get("incidents", []):
        similar_incidents.append({
            "incident_id": inc.get("id", "INC-UNKNOWN"),
            "description": f"Machine {inc.get('machine')} issue: {inc.get('issue')} (vibration: {inc.get('vibration', 'N/A')}, cost: ${inc.get('cost', 0)})",
            "machine_type": "CNC Mill",
            "failure_mode": "TWF" if "wear" in inc.get("issue", "").lower() else "RNF",
            "time_to_failure_hours": 5.5,
            "resolution": inc.get("resolution", "N/A"),
            "similarity_score": 0.85
        })
        
    relevant_sops = []
    for sop in evidence.get("sops", []):
        relevant_sops.append({
            "sop_id": sop.get("id", "SOP-UNKNOWN"),
            "title": sop.get("trigger", "Procedure trigger"),
            "content": sop.get("rule", "Procedure execution details"),
            "applicability": "CNC Mill class",
            "priority": "MANDATORY" if "immediate" in sop.get("rule", "").lower() or "requires" in sop.get("rule", "").lower() else "RECOMMENDED"
        })
        
    maintenance_history = []
    for log in evidence.get("maintenance_logs", []):
        maintenance_history.append({
            "date": log.get("date", datetime.now(timezone.utc).isoformat()),
            "action_taken": log.get("action", "Maintenance action"),
            "technician": log.get("technician", "Unknown"),
            "outcome": log.get("status", "completed")
        })
        
    has_results = len(similar_incidents) > 0 or len(relevant_sops) > 0
    return {
        "retrieval_confidence": retrieval_confidence if has_results else 0.0,
        "similar_incidents": similar_incidents,
        "relevant_sops": relevant_sops,
        "maintenance_history": maintenance_history,
    }


def run_factoryguard_system(
    sample_row: pd.DataFrame,
    machine_id: str = "M3",
    machine_context: Optional[dict] = None,
    policy_overrides: Optional[dict] = None,
    recurrence_count_last_24h: int = 0,
    force_mock: bool = True,
    model_path: str = "factoryguard_failure_model.pkl"
) -> dict:
    """
    Runs the complete connected pipeline:
    1. Loads XGBoost model & predicts failure probability.
    2. Flags sensor metrics out-of-range.
    3. Runs RAG search against FAISS database using semantic keywords.
    4. Submits prediction, sensors, and RAG context to ARIA.
    5. Checks the proposed recommendations using the Governance policy engine.
    """
    # 1. Model Inference
    model, cols = load_xgboost_model(model_path)
    ml_output = build_ml_output(sample_row, model, cols, machine_id=machine_id)
    
    # 2. Map basic reasoning schema input
    reasoning_input = ml_output_to_reasoning_input(ml_output, machine_context=machine_context)
    
    # 3. Perform RAG query
    query = construct_anomaly_search_query(reasoning_input["sensor_data"], machine_id)
    logger.info(f"Querying RAG system with text: '{query}'")
    rag_evidence = query_rag_and_map_evidence(query)
    
    # Inject RAG findings into reasoning input
    reasoning_input["rag_evidence"] = rag_evidence
    
    # 4. Probabilistic LLM Reasoning (ARIA)
    agent = ReasoningAgent(force_mock=force_mock)
    reasoning_output = agent.analyze(reasoning_input)
    
    # 5. Deterministic Governance Checks
    engine = GovernanceEngine()
    governance_result = engine.evaluate(
        reasoning_output,
        machine_context=reasoning_input["machine_context"],
        policy_overrides=policy_overrides or {},
        recurrence_count_last_24h=recurrence_count_last_24h
    )
    
    return governance_result


if __name__ == "__main__":
    print("=" * 60)
    print("FactoryGuard Integrated Pipeline System — Self-Test")
    print("=" * 60)
    
    # Test sample with tool wear anomaly
    test_row = pd.DataFrame([{
        "Air_temperature_K":     304.5,
        "Process_temperature_K": 312.0,
        "Rotational_speed_rpm":  1400.0,
        "Torque_Nm":             75.0,
        "Tool_wear_min":         245.0, # Flagged high tool wear anomaly
        "Type_H":                0,
        "Type_L":                1,
        "Type_M":                0,
    }])
    
    res = run_factoryguard_system(
        test_row,
        machine_id="M3",
        machine_context={"operator_present": True, "batch_criticality": "HIGH"},
        force_mock=True
    )
    
    print("\n[VERDICT]:", res["decision"]["verdict"])
    print("[RATIONALE]:", res["decision"]["decision_rationale"])
    print("[CHECKS PASSED]:", res["audit_log"]["governance_checks_passed"])
    print("[CHECKS FAILED]:", res["audit_log"]["governance_checks_failed"])
    print("[HUMAN APPROVAL REQUIRED]:", res["human_approval_request"]["required"])
    if res["human_approval_request"]["required"]:
        print("  - Urgency:", res["human_approval_request"]["urgency"])
        print("  - Escalation:", res["human_approval_request"]["escalation_level"])
        print("  - Context Summary:", res["human_approval_request"]["context_for_human"]["summary"])
