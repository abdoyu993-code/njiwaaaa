from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=2,
        description="Question posée par l'utilisateur",
    )

    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Nombre maximal de chunks recherchés",
    )


class AskSource(BaseModel):
    title: str | None = None
    filename: str | None = None
    relative_path: str | None = None
    domain: str | None = None
    best_similarity_score: float


class AskForm(BaseModel):
    title: str
    filename: str
    download_url: str


class AskResponse(BaseModel):
    question: str
    answer: str
    status: str
    domain: str
    intent: str
    confidence: float
    needs_clarification: bool
    generated: bool
    model: str | None = None
    finish_reason: str | None = None
    sources: list[AskSource] = Field(
        default_factory=list
    )
    forms: list[AskForm] = Field(
        default_factory=list
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict
    )
