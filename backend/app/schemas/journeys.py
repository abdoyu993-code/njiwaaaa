from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class JourneyPlanRequest(BaseModel):
    goal: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description=(
            "Objectif exprimé librement, par exemple "
            "« j'ai acheté une voiture d'occasion »."
        ),
    )
    session_id: str | None = Field(
        default=None,
        max_length=64,
    )


class JourneyStepResponse(BaseModel):
    index: int
    title: str
    detail: str
    duration: str | None = None
    documents: list[str] = Field(default_factory=list)
    links: list[dict[str, str]] = Field(default_factory=list)
    forms: list[dict[str, str]] = Field(default_factory=list)
    source: str | None = None
    done: bool = False


class JourneyResponse(BaseModel):
    reference: str
    journey_key: str
    title: str
    summary: str
    domain: str
    goal: str
    deadline: str | None = None
    warning: str | None = None
    steps: list[JourneyStepResponse] = Field(
        default_factory=list
    )
    completed_count: int
    total_steps: int
    progress: int
    next_step: JourneyStepResponse | None = None
    created_at: datetime
    updated_at: datetime


class JourneyPlanResponse(BaseModel):
    detected: bool
    confidence: float = 0.0
    reason: str = ""
    journey: JourneyResponse | None = None
    suggestions: list[dict[str, Any]] = Field(
        default_factory=list
    )
    message: str = ""


class JourneyStepUpdateRequest(BaseModel):
    done: bool = True
