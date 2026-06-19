
from __future__ import annotations

import json

from graph import workflow

# What the anomaly module would hand off
EXAMPLE_ALERT = {
    "machine_id": "M-07",
    "anomaly_score": 0.91,
    "flagged_sensors": [
        {"sensor": "vibration", "value": 8.1, "normal_range": (1.3, 2.7), "z_score": 17.4},
        {"sensor": "temperature", "value": 77.0, "normal_range": (60.0, 70.0), "z_score": 4.8},
    ],
}


def run(alert: dict) -> dict:
    initial: dict = {
        "alert": alert,
        "messages": [],
        "diagnosis": None,
        "action_proposal": None,
        "diagnosis_attempts": 0,
        "route": "",
    }
    return workflow.invoke(initial)


def pretty(final: dict) -> None:
    print("\n=== AGENT ACTIVITY LOG ===")
    for m in final["messages"]:
        print(" ", getattr(m, "content", m))

    print("\n=== DIAGNOSIS (Diagnosis + Evidence Agent) ===")
    print(json.dumps(final["diagnosis"], indent=2))

    print("\n=== PROPOSED ACTIONS (Action Proposal Agent) ===")
    print(json.dumps(final["action_proposal"], indent=2))


if __name__ == "__main__":
    final_state = run(EXAMPLE_ALERT)
    pretty(final_state)
