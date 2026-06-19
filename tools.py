"""
tools.py  —  Tools the agents call.

These are intentionally simple stand-ins. In the real system:
  * `retrieve_evidence`         -> your RAG retriever over incidents/SOPs/logs
  * `get_machine_baseline`      -> your time-series / asset DB
  * `lookup_recommended_actions`-> your SOP / playbook store
  * `get_action_catalog`        -> your allow-listed action registry

The agentic mechanics (tool binding, tool-calling loop, tool results feeding
back into the model) are identical regardless of what's behind these functions.
"""

from __future__ import annotations

from langchain_core.tools import tool


# --------------------------------------------------------------------------- #
# Diagnosis + Evidence Agent tools
# --------------------------------------------------------------------------- #

# Canned "knowledge base" — replace with a real vector store / RAG retriever.
_EVIDENCE_CORPUS = [
    {
        "id": "INC-1042",
        "type": "incident",
        "text": "M-07 bearing wear: vibration rose to 8+ mm/s with bearing "
                "housing temperature climbing ~12C above normal. Resolved by "
                "replacing the drive-end bearing.",
        "tags": ["vibration", "temperature", "bearing"],
    },
    {
        "id": "SOP-BRG-01",
        "type": "sop",
        "text": "Bearing fault response: reduce load, schedule bearing "
                "inspection, replace if vibration exceeds 7 mm/s. Do not "
                "increase speed while vibration is elevated.",
        "tags": ["vibration", "bearing", "load"],
    },
    {
        "id": "SOP-THERM-02",
        "type": "sop",
        "text": "Overheating response: reduce load, verify coolant flow, "
                "shut down if temperature exceeds 95C.",
        "tags": ["temperature", "power", "coolant"],
    },
    {
        "id": "SOP-LOAD-03",
        "type": "sop",
        "text": "Mechanical overload: power draw high with RPM dropping. "
                "Reduce load immediately and redistribute demand.",
        "tags": ["power", "rpm", "load"],
    },
    {
        "id": "LOG-M07-221",
        "type": "maintenance_log",
        "text": "M-07 drive-end bearing last replaced 14 months ago; "
                "lubrication interval overdue by 3 weeks at last check.",
        "tags": ["bearing", "lubrication", "maintenance"],
    },
]


@tool
def retrieve_evidence(query: str) -> str:
    """Search incidents, SOPs, and maintenance logs for evidence relevant to a
    fault. Pass a query describing the symptoms (e.g. 'high vibration and
    temperature on drive-end bearing'). Returns matching document snippets with
    their IDs so they can be cited as supporting evidence."""
    q = query.lower()
    hits = [
        d for d in _EVIDENCE_CORPUS
        if any(tag in q for tag in d["tags"]) or any(w in q for w in d["text"].lower().split()[:0])
    ]
    # Fallback keyword scan so the agent always gets something to reason over.
    if not hits:
        hits = [d for d in _EVIDENCE_CORPUS if any(w in d["text"].lower() for w in q.split())]
    if not hits:
        return "No matching evidence found. Try different symptom keywords."
    return "\n".join(f"[{d['id']} | {d['type']}] {d['text']}" for d in hits[:4])


@tool
def get_machine_baseline(machine_id: str) -> str:
    """Return the normal operating ranges for a machine's sensors, used to
    judge how abnormal a reading is."""
    return (
        f"{machine_id} normal ranges: vibration 1.3-2.7 mm/s, "
        f"temperature 60-70 C, pressure 4.5-5.5 bar, "
        f"rpm 1450-1550, power 67-83 kW."
    )


# --------------------------------------------------------------------------- #
# Action Proposal Agent tools
# --------------------------------------------------------------------------- #

_SOP_ACTIONS = {
    "bearing_wear": ["reduce_load", "inspect_bearing", "schedule_maintenance"],
    "overheating": ["reduce_load", "check_coolant", "shutdown"],
    "mechanical_overload": ["reduce_load", "reduce_speed"],
    "pressure_loss": ["inspect_seals", "shutdown"],
    "underspeed": ["inspect_drive", "schedule_maintenance"],
}


@tool
def lookup_recommended_actions(failure_mode: str) -> str:
    """Given a canonical failure mode tag (e.g. 'bearing_wear'), return the
    SOP-recommended remedial actions for that fault."""
    actions = _SOP_ACTIONS.get(failure_mode.lower().strip())
    if not actions:
        return (f"No SOP entry for '{failure_mode}'. Choose from the action "
                f"catalog and justify based on the diagnosis.")
    return f"SOP-recommended actions for {failure_mode}: {', '.join(actions)}."


@tool
def get_action_catalog() -> str:
    """Return the allow-listed set of actions that may be proposed. Only
    actions from this catalog are permitted."""
    catalog = [
        "increase_speed", "reduce_speed", "reduce_load", "shutdown",
        "inspect_bearing", "inspect_seals", "inspect_drive",
        "check_coolant", "schedule_maintenance", "continue_monitoring",
    ]
    return "Allowed actions: " + ", ".join(catalog)


DIAGNOSIS_TOOLS = [retrieve_evidence, get_machine_baseline]
ACTION_TOOLS = [lookup_recommended_actions, get_action_catalog]
