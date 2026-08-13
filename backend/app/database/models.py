from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import (
    Mapped,
    mapped_column,
    relationship,
)

from backend.app.database.session import Base


COMPLAINT_STATUSES = (
    "recue",
    "en_cours",
    "traitee",
    "rejetee",
)

COMPLAINT_STATUS_LABELS = {
    "recue": "Reçue",
    "en_cours": "En cours de traitement",
    "traitee": "Traitée",
    "rejetee": "Rejetée",
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Complaint(Base):
    """Réclamation déposée par un citoyen."""

    __tablename__ = "complaints"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
    )
    reference: Mapped[str] = mapped_column(
        String(32),
        unique=True,
        index=True,
    )
    full_name: Mapped[str] = mapped_column(String(160))
    email: Mapped[str] = mapped_column(
        String(160),
        index=True,
    )
    phone: Mapped[str | None] = mapped_column(
        String(40),
        default=None,
    )
    national_id: Mapped[str | None] = mapped_column(
        String(40),
        default=None,
    )
    city: Mapped[str | None] = mapped_column(
        String(80),
        default=None,
    )
    category: Mapped[str] = mapped_column(
        String(60),
        index=True,
    )
    subject: Mapped[str] = mapped_column(String(200))
    message: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(
        String(20),
        default="recue",
        index=True,
    )
    priority: Mapped[str] = mapped_column(
        String(20),
        default="normale",
        index=True,
    )
    detected_intent: Mapped[str | None] = mapped_column(
        String(80),
        default=None,
    )
    admin_note: Mapped[str | None] = mapped_column(
        Text,
        default=None,
    )
    official_response: Mapped[str | None] = mapped_column(
        Text,
        default=None,
    )
    responded_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        default=None,
    )
    responded_by: Mapped[str | None] = mapped_column(
        String(80),
        default=None,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        index=True,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
    )

    events: Mapped[list["ComplaintEvent"]] = relationship(
        back_populates="complaint",
        cascade="all, delete-orphan",
        order_by="ComplaintEvent.created_at",
    )

    @property
    def status_label(self) -> str:
        return COMPLAINT_STATUS_LABELS.get(
            self.status,
            self.status,
        )


class ComplaintEvent(Base):
    """Étape du traitement d'une réclamation."""

    __tablename__ = "complaint_events"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
    )
    complaint_id: Mapped[int] = mapped_column(
        ForeignKey(
            "complaints.id",
            ondelete="CASCADE",
        ),
        index=True,
    )
    status: Mapped[str] = mapped_column(String(20))
    event_type: Mapped[str] = mapped_column(
        String(20),
        default="status",
    )
    note: Mapped[str | None] = mapped_column(
        Text,
        default=None,
    )
    author: Mapped[str] = mapped_column(
        String(80),
        default="systeme",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
    )

    complaint: Mapped[Complaint] = relationship(
        back_populates="events",
    )

    @property
    def status_label(self) -> str:
        return COMPLAINT_STATUS_LABELS.get(
            self.status,
            self.status,
        )


class Journey(Base):
    """Parcours administratif guidé suivi par un citoyen."""

    __tablename__ = "journeys"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
    )
    reference: Mapped[str] = mapped_column(
        String(32),
        unique=True,
        index=True,
    )
    session_id: Mapped[str | None] = mapped_column(
        String(64),
        default=None,
        index=True,
    )
    journey_key: Mapped[str] = mapped_column(
        String(60),
        index=True,
    )
    goal: Mapped[str] = mapped_column(String(500))
    completed_steps: Mapped[str] = mapped_column(
        Text,
        default="[]",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        index=True,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
    )


class ConversationLog(Base):
    """Trace d'un échange avec l'assistant."""

    __tablename__ = "conversation_logs"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
    )
    session_id: Mapped[str | None] = mapped_column(
        String(64),
        default=None,
        index=True,
    )
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    domain: Mapped[str] = mapped_column(
        String(60),
        index=True,
    )
    intent: Mapped[str] = mapped_column(
        String(80),
        index=True,
    )
    confidence: Mapped[float] = mapped_column(
        Float,
        default=0.0,
    )
    status: Mapped[str] = mapped_column(
        String(40),
        default="",
        index=True,
    )
    generated: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
    )
    extractive: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
    )
    needs_clarification: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
    )
    total_time_ms: Mapped[float] = mapped_column(
        Float,
        default=0.0,
    )
    language: Mapped[str] = mapped_column(
        String(12),
        default="fr",
    )
    feedback: Mapped[str | None] = mapped_column(
        String(16),
        default=None,
        index=True,
    )
    feedback_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        default=None,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        index=True,
    )
