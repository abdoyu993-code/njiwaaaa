from __future__ import annotations

from typing import Any

from pydantic import BaseModel, EmailStr, Field


class AdminLoginRequest(BaseModel):
    username: str = Field(
        ...,
        min_length=2,
        max_length=80,
    )
    password: str = Field(
        ...,
        min_length=2,
        max_length=200,
    )


class AdminLoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_at: int
    username: str
    display_name: str


class AdminProfileResponse(BaseModel):
    username: str
    display_name: str


class DashboardResponse(BaseModel):
    period_days: int
    kpis: dict[str, Any]
    timeseries: list[dict[str, Any]]
    domains: list[dict[str, Any]]
    top_intents: list[dict[str, Any]]
    answer_types: list[dict[str, Any]]
    recent: list[dict[str, Any]]
    complaints: dict[str, Any]
    feedback: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Avis des citoyens sur les réponses de "
            "l'assistant."
        ),
    )


class EmailTestRequest(BaseModel):
    """Destinataire d'un email de vérification SMTP."""

    to: EmailStr
