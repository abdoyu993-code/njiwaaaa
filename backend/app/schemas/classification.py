from __future__ import annotations

from pydantic import BaseModel, Field


class ClassificationRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="Question envoyée par l'utilisateur",
    )

    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Nombre d'alternatives retournées",
    )


class ClassificationAlternative(BaseModel):
    intent: str
    domain: str
    confidence: float


class ClassificationResponse(BaseModel):
    text: str
    normalized_text: str
    domain: str
    intent: str
    confidence: float
    confidence_threshold: float
    needs_clarification: bool
    alternatives: list[ClassificationAlternative]