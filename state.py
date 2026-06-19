from __future__ import annotations

from typing import Annotated, Literal, Optional, TypedDict

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class FlaggedSensor(BaseModel):
    sensor: str
    value: float
    normal_range: tuple[float, float]
    z_score: float = Field(description="How many std-devs from normal.")


class AnomalyAlert(BaseModel):
    machine_id: str
    anomaly_score: float = Field(ge=0, le=1)
    flagged_sensors: list[FlaggedSensor]



class DiagnosisResult(BaseModel):
    root_cause: str = Field(description="Plain-language root cause.")
    failure_mode: str = Field(description="Canonical failure mode tag, e.g. 'bearing_wear'.")
    confidence: float = Field(ge=0, le=1)
    supporting_evidence: list[str] = Field(
        description="Doc IDs / log refs the diagnosis is grounded in."
    )
    reasoning: str



class ProposedAction(BaseModel):
    action: str
    rationale: str
    urgency: Literal["low", "medium", "high", "critical"]


class ActionProposal(BaseModel):
    actions: list[ProposedAction]
    summary: str


class WorkflowState(TypedDict):
    alert: dict
    messages: Annotated[list, add_messages]
    diagnosis: Optional[dict]
    action_proposal: Optional[dict]
    diagnosis_attempts: int
    route: str
