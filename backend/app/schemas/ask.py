from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PreviousConversationContext(BaseModel):
    previous_question: str = Field(
        ...,
        min_length=2,
    )
    previous_domain: str = Field(
        ...,
        min_length=1,
    )
    previous_intent: str = Field(
        ...,
        min_length=1,
    )


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

    context: PreviousConversationContext | None = None

    session_id: str | None = Field(
        default=None,
        max_length=64,
        description=(
            "Identifiant de session du navigateur, "
            "utilisé pour les statistiques."
        ),
    )


class AskSource(BaseModel):
    title: str | None = None
    filename: str | None = None
    relative_path: str | None = None
    domain: str | None = None
    best_similarity_score: float


class AskForm(BaseModel):
    title: str
    filename: str | None = None
    download_url: str
    button_label: str | None = None
    is_external: bool = False


class AskServiceLink(BaseModel):
    label: str
    url: str


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
    service_links: list[AskServiceLink] = Field(
        default_factory=list
    )
    related_questions: list[str] = Field(
        default_factory=list,
        description=(
            "Questions proches réellement traitées "
            "par l'assistant."
        ),
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict
    )
