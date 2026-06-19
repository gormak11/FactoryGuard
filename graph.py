
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agents import action_node, diagnosis_node
from state import WorkflowState

CONFIDENCE_THRESHOLD = 0.6
MAX_DIAGNOSIS_ATTEMPTS = 2


def orchestrator(state: WorkflowState) -> dict:
    dx = state.get("diagnosis") or {}
    confidence = dx.get("confidence", 0.0)
    attempts = state.get("diagnosis_attempts", 0)

    if confidence < CONFIDENCE_THRESHOLD and attempts < MAX_DIAGNOSIS_ATTEMPTS:
        decision = "rediagnose"
        note = (f"confidence {confidence} < {CONFIDENCE_THRESHOLD}; "
                f"requesting more evidence (attempt {attempts}/{MAX_DIAGNOSIS_ATTEMPTS}).")
    else:
        decision = "act"
        note = f"diagnosis accepted (confidence {confidence}); handing off to Action agent."

    from langchain_core.messages import AIMessage
    return {"route": decision, "messages": [AIMessage(content=f"[Orchestrator] {note}")]}


def _route(state: WorkflowState) -> str:
    return state["route"]


def build_workflow():
    g = StateGraph(WorkflowState)

    g.add_node("diagnosis", diagnosis_node)
    g.add_node("orchestrator", orchestrator)
    g.add_node("action", action_node)

    g.add_edge(START, "diagnosis")
    g.add_edge("diagnosis", "orchestrator")
    g.add_conditional_edges(
        "orchestrator",
        _route,
        {"rediagnose": "diagnosis", "act": "action"},
    )
    g.add_edge("action", END)

    return g.compile()

workflow = build_workflow()
