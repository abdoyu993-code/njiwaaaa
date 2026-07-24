from __future__ import annotations

from pydantic import BaseModel, Field

from backend.app.schemas.classification import (
    ClassificationResponse,
)


class DecisionRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="Question du citoyen",
    )

    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
    )


class DecisionResponse(BaseModel):
    text: str
    route: str
    reason: str
    classification: ClassificationResponse