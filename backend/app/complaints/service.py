from __future__ import annotations

import secrets
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from backend.app.database.models import (
    COMPLAINT_STATUS_LABELS,
    Complaint,
    ComplaintEvent,
    utc_now,
)
from backend.app.complaints.triage import PRIORITY_LABELS
from backend.app.schemas.complaints import (
    COMPLAINT_CATEGORY_LABELS,
    ComplaintCreateRequest,
)


REFERENCE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


def generate_reference(
    session: Session,
    now: datetime | None = None,
) -> str:
    """Produit une référence unique de type NARSA-2026-A1B2C3."""
    year = (now or datetime.now(timezone.utc)).year

    for _ in range(20):
        suffix = "".join(
            secrets.choice(REFERENCE_ALPHABET)
            for _ in range(6)
        )
        reference = f"NARSA-{year}-{suffix}"

        exists = session.scalar(
            select(Complaint.id).where(
                Complaint.reference == reference
            )
        )

        if not exists:
            return reference

    raise RuntimeError(
        "Impossible de générer une référence unique."
    )


def create_complaint(
    session: Session,
    payload: ComplaintCreateRequest,
    analysis: dict[str, Any] | None = None,
) -> Complaint:
    """Enregistre une réclamation et son premier événement."""
    triage = analysis or {}
    priority = triage.get("priority") or "normale"

    complaint = Complaint(
        reference=generate_reference(session),
        full_name=payload.full_name,
        email=str(payload.email).strip().lower(),
        phone=(payload.phone or "").strip() or None,
        national_id=(
            (payload.national_id or "").strip() or None
        ),
        city=(payload.city or "").strip() or None,
        category=payload.category,
        subject=payload.subject,
        message=payload.message,
        status="recue",
        priority=priority,
        detected_intent=triage.get("detected_intent"),
    )

    note = "Réclamation reçue et enregistrée dans la plateforme."

    if priority != "normale":
        reasons = "; ".join(
            triage.get("priority_reasons") or []
        )
        note += (
            " Priorité estimée automatiquement : "
            f"{PRIORITY_LABELS.get(priority, priority).lower()}"
        )

        if reasons:
            note += f" ({reasons.lower()})"

        note += "."

    complaint.events.append(
        ComplaintEvent(
            status="recue",
            event_type="creation",
            note=note,
            author="systeme",
        )
    )

    session.add(complaint)
    session.commit()
    session.refresh(complaint)

    return complaint


def get_complaint_by_reference(
    session: Session,
    reference: str,
    email: str | None = None,
) -> Complaint | None:
    """Retrouve une réclamation, éventuellement filtrée par email."""
    statement = (
        select(Complaint)
        .options(selectinload(Complaint.events))
        .where(
            Complaint.reference
            == reference.strip().upper()
        )
    )

    if email:
        statement = statement.where(
            func.lower(Complaint.email)
            == email.strip().lower()
        )

    return session.scalar(statement)


def list_complaints(
    session: Session,
    status: str | None = None,
    category: str | None = None,
    priority: str | None = None,
    search: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[int, list[Complaint]]:
    """Liste paginée des réclamations pour l'espace agent."""
    filters = []

    if status:
        filters.append(Complaint.status == status)

    if category:
        filters.append(Complaint.category == category)

    if priority:
        filters.append(Complaint.priority == priority)

    if search:
        pattern = f"%{search.strip().lower()}%"
        filters.append(
            func.lower(Complaint.reference).like(pattern)
            | func.lower(Complaint.full_name).like(pattern)
            | func.lower(Complaint.email).like(pattern)
            | func.lower(Complaint.subject).like(pattern)
        )

    count_statement = select(
        func.count(Complaint.id)
    )
    list_statement = select(Complaint)

    for condition in filters:
        count_statement = count_statement.where(condition)
        list_statement = list_statement.where(condition)

    total = int(
        session.scalar(count_statement) or 0
    )

    items = list(
        session.scalars(
            list_statement.order_by(
                Complaint.created_at.desc()
            )
            .limit(max(1, min(limit, 200)))
            .offset(max(0, offset))
        )
    )

    return total, items


def update_complaint_status(
    session: Session,
    complaint: Complaint,
    status: str,
    note: str | None,
    author: str,
) -> Complaint:
    """Change le statut et journalise l'étape."""
    complaint.status = status
    complaint.updated_at = utc_now()

    if note:
        complaint.admin_note = note

    session.add(
        ComplaintEvent(
            complaint_id=complaint.id,
            status=status,
            event_type="status",
            note=note,
            author=author or "agent",
        )
    )

    session.commit()
    session.refresh(complaint)

    return complaint


def add_official_response(
    session: Session,
    complaint: Complaint,
    response: str,
    author: str,
    status: str = "traitee",
) -> Complaint:
    """
    Enregistre la réponse officielle rédigée par un agent.

    La réponse complète reste dans la plateforme : l'email
    n'est qu'une notification.
    """
    cleaned = response.strip()

    if not cleaned:
        raise ValueError("La réponse ne peut pas être vide.")

    complaint.official_response = cleaned
    complaint.responded_at = utc_now()
    complaint.responded_by = author or "agent"
    complaint.status = status
    complaint.updated_at = utc_now()

    session.add(
        ComplaintEvent(
            complaint_id=complaint.id,
            status=status,
            event_type="response",
            note="Réponse officielle transmise au citoyen.",
            author=author or "agent",
        )
    )

    session.commit()
    session.refresh(complaint)

    return complaint


def serialize_complaint(
    complaint: Complaint,
) -> dict[str, Any]:
    return {
        "reference": complaint.reference,
        "full_name": complaint.full_name,
        "email": complaint.email,
        "phone": complaint.phone,
        "city": complaint.city,
        "category": complaint.category,
        "category_label": COMPLAINT_CATEGORY_LABELS.get(
            complaint.category,
            complaint.category,
        ),
        "subject": complaint.subject,
        "message": complaint.message,
        "status": complaint.status,
        "status_label": COMPLAINT_STATUS_LABELS.get(
            complaint.status,
            complaint.status,
        ),
        "priority": complaint.priority or "normale",
        "priority_label": PRIORITY_LABELS.get(
            complaint.priority or "normale",
            "Normale",
        ),
        "detected_intent": complaint.detected_intent,
        "admin_note": complaint.admin_note,
        "official_response": complaint.official_response,
        "responded_at": complaint.responded_at,
        "responded_by": complaint.responded_by,
        "created_at": complaint.created_at,
        "updated_at": complaint.updated_at,
        "events": [
            {
                "status": event.status,
                "status_label": (
                    COMPLAINT_STATUS_LABELS.get(
                        event.status,
                        event.status,
                    )
                ),
                "event_type": event.event_type or "status",
                "note": event.note,
                "author": event.author,
                "created_at": event.created_at,
            }
            for event in complaint.events
        ],
    }


def serialize_summary(
    complaint: Complaint,
) -> dict[str, Any]:
    return {
        "reference": complaint.reference,
        "full_name": complaint.full_name,
        "email": complaint.email,
        "category": complaint.category,
        "category_label": COMPLAINT_CATEGORY_LABELS.get(
            complaint.category,
            complaint.category,
        ),
        "subject": complaint.subject,
        "status": complaint.status,
        "status_label": COMPLAINT_STATUS_LABELS.get(
            complaint.status,
            complaint.status,
        ),
        "priority": complaint.priority or "normale",
        "priority_label": PRIORITY_LABELS.get(
            complaint.priority or "normale",
            "Normale",
        ),
        "created_at": complaint.created_at,
        "updated_at": complaint.updated_at,
    }
