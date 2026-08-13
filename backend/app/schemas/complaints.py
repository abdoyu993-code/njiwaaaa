from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, EmailStr, Field, field_validator


COMPLAINT_CATEGORIES = (
    "permis",
    "vehicule",
    "visite_technique",
    "infractions",
    "accident",
    "prevention",
    "contact_reclamation",
    "autre",
)

COMPLAINT_CATEGORY_LABELS = {
    "permis": "Permis de conduire",
    "vehicule": "Véhicule et carte grise",
    "visite_technique": "Visite technique",
    "infractions": "Infractions et amendes",
    "accident": "Accident de la route",
    "prevention": "Prévention routière",
    "contact_reclamation": "Contact et information",
    "autre": "Autre",
}


class ComplaintCreateRequest(BaseModel):
    full_name: str = Field(
        ...,
        min_length=3,
        max_length=160,
    )
    email: EmailStr
    phone: str | None = Field(
        default=None,
        max_length=40,
    )
    national_id: str | None = Field(
        default=None,
        max_length=40,
    )
    city: str | None = Field(
        default=None,
        max_length=80,
    )
    category: str = Field(...)
    subject: str = Field(
        ...,
        min_length=3,
        max_length=200,
    )
    message: str = Field(
        ...,
        min_length=10,
        max_length=5000,
    )

    @field_validator("category")
    @classmethod
    def validate_category(cls, value: str) -> str:
        normalized = value.strip().lower()

        if normalized not in COMPLAINT_CATEGORIES:
            allowed = ", ".join(COMPLAINT_CATEGORIES)

            raise ValueError(
                f"Catégorie invalide. Valeurs autorisées : {allowed}."
            )

        return normalized

    @field_validator(
        "full_name",
        "subject",
        "message",
    )
    @classmethod
    def strip_text(cls, value: str) -> str:
        return " ".join(value.strip().split())


class ComplaintEventResponse(BaseModel):
    status: str
    status_label: str
    event_type: str = "status"
    note: str | None = None
    author: str
    created_at: datetime


class ComplaintResponse(BaseModel):
    reference: str
    full_name: str
    email: str
    phone: str | None = None
    city: str | None = None
    category: str
    category_label: str
    subject: str
    message: str
    status: str
    status_label: str
    priority: str = "normale"
    priority_label: str = "Normale"
    detected_intent: str | None = None
    admin_note: str | None = None
    official_response: str | None = None
    responded_at: datetime | None = None
    responded_by: str | None = None
    created_at: datetime
    updated_at: datetime
    events: list[ComplaintEventResponse] = Field(
        default_factory=list
    )


class ComplaintCreatedResponse(BaseModel):
    reference: str
    status: str
    status_label: str
    created_at: datetime
    message: str


class ComplaintTrackingRequest(BaseModel):
    reference: str = Field(
        ...,
        min_length=4,
        max_length=32,
    )
    email: EmailStr


class ComplaintSummary(BaseModel):
    reference: str
    full_name: str
    email: str
    category: str
    category_label: str
    subject: str
    status: str
    status_label: str
    priority: str = "normale"
    priority_label: str = "Normale"
    created_at: datetime
    updated_at: datetime


class ComplaintAnalysisRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=10,
        max_length=5000,
    )
    subject: str | None = Field(
        default=None,
        max_length=200,
    )


class ComplaintInstantAnswer(BaseModel):
    answer: str
    status: str
    domain: str | None = None
    intent: str | None = None
    confidence: float = 0.0
    sources: list[str] = Field(default_factory=list)


class ComplaintAnalysisResponse(BaseModel):
    category: str
    category_label: str
    category_confidence: float
    category_is_reliable: bool
    detected_intent: str | None = None
    priority: str
    priority_label: str
    priority_reasons: list[str] = Field(
        default_factory=list
    )
    instant_answer: ComplaintInstantAnswer | None = None


class ComplaintListResponse(BaseModel):
    total: int
    items: list[ComplaintSummary] = Field(
        default_factory=list
    )


class ComplaintStatusUpdateRequest(BaseModel):
    status: str
    note: str | None = Field(
        default=None,
        max_length=2000,
    )

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: str) -> str:
        from backend.app.database.models import (
            COMPLAINT_STATUSES,
        )

        normalized = value.strip().lower()

        if normalized not in COMPLAINT_STATUSES:
            allowed = ", ".join(COMPLAINT_STATUSES)

            raise ValueError(
                f"Statut invalide. Valeurs autorisées : {allowed}."
            )

        return normalized


class ComplaintResponseRequest(BaseModel):
    """Réponse officielle rédigée par un agent."""

    response: str = Field(
        ...,
        min_length=10,
        max_length=5000,
    )
    status: str = Field(default="traitee")

    @field_validator("response")
    @classmethod
    def clean_response(cls, value: str) -> str:
        return value.strip()

    @field_validator("status")
    @classmethod
    def validate_response_status(cls, value: str) -> str:
        normalized = value.strip().lower()

        if normalized not in {"traitee", "rejetee", "en_cours"}:
            raise ValueError(
                "Statut invalide pour une réponse : "
                "traitee, rejetee ou en_cours."
            )

        return normalized
