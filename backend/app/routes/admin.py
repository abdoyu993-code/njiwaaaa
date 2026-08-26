from __future__ import annotations

import csv
import io

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from backend.app.analytics.insights import build_insights
from backend.app.analytics.service import (
    build_dashboard,
    build_recent_conversations,
)
from backend.app.complaints.service import (
    add_official_response,
    get_complaint_by_reference,
    list_complaints,
    serialize_complaint,
    serialize_summary,
    update_complaint_status,
)
from backend.app.core.config import settings
from backend.app.core.security import (
    create_admin_token,
    require_admin,
    verify_credentials,
)
from backend.app.database.session import get_session
from backend.app.notifications.email import (
    notify_response_added,
    notify_status_changed,
    send_test_email,
    sent_messages,
    smtp_configuration,
)
from backend.app.schemas.admin import (
    AdminLoginRequest,
    AdminLoginResponse,
    AdminProfileResponse,
    DashboardResponse,
    EmailTestRequest,
)
from backend.app.schemas.complaints import (
    ComplaintListResponse,
    ComplaintResponse,
    ComplaintResponseRequest,
    ComplaintStatusUpdateRequest,
)


router = APIRouter(
    prefix="/admin",
    tags=["Espace administration"],
)


def csv_response(
    filename: str,
    header: list[str],
    rows: list[list[str]],
) -> StreamingResponse:
    """Construit un fichier CSV téléchargeable (séparateur ;)."""
    buffer = io.StringIO()
    writer = csv.writer(
        buffer,
        delimiter=";",
        quoting=csv.QUOTE_MINIMAL,
    )

    writer.writerow(header)
    writer.writerows(rows)

    # BOM pour que les accents restent lisibles dans Excel.
    content = "﻿" + buffer.getvalue()

    return StreamingResponse(
        iter([content]),
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{filename}"'
            )
        },
    )


def notification_status(
    result: dict | None,
) -> dict[str, str | None] | None:
    """
    Réduit le résultat d'un envoi email à ce que l'agent doit
    voir. Ne laisse passer ni le corps du message ni aucun
    élément de configuration sensible.
    """
    if not result:
        return None

    return {
        "status": result.get("status", "inconnu"),
        "to": result.get("to", ""),
        "detail": result.get("detail"),
        "error": result.get("error"),
    }


@router.post(
    "/login",
    response_model=AdminLoginResponse,
)
def login(
    payload: AdminLoginRequest,
) -> AdminLoginResponse:
    """Authentifie un agent NARSA et retourne un jeton signé."""
    if not verify_credentials(
        username=payload.username,
        password=payload.password,
    ):
        raise HTTPException(
            status_code=401,
            detail="Identifiants incorrects.",
        )

    token, expires_at = create_admin_token(
        username=payload.username.strip()
    )

    return AdminLoginResponse(
        access_token=token,
        expires_at=expires_at,
        username=payload.username.strip(),
        display_name=settings.admin_display_name,
    )


@router.get(
    "/me",
    response_model=AdminProfileResponse,
)
def profile(
    username: str = Depends(require_admin),
) -> AdminProfileResponse:
    return AdminProfileResponse(
        username=username,
        display_name=settings.admin_display_name,
    )


@router.get(
    "/dashboard",
    response_model=DashboardResponse,
)
def dashboard(
    days: int = Query(
        default=7,
        ge=1,
        le=90,
    ),
    _: str = Depends(require_admin),
    session: Session = Depends(get_session),
) -> DashboardResponse:
    """Indicateurs agrégés du Dashboard IA."""
    return DashboardResponse(
        **build_dashboard(
            session=session,
            days=days,
        )
    )


@router.get("/insights")
def insights(
    days: int = Query(
        default=7,
        ge=1,
        le=90,
    ),
    _: str = Depends(require_admin),
    session: Session = Depends(get_session),
) -> dict:
    """Alertes qualité et angles morts documentaires."""
    return build_insights(
        session=session,
        days=days,
    )


@router.get("/complaints/export")
def export_complaints(
    status: str | None = None,
    category: str | None = None,
    priority: str | None = None,
    search: str | None = None,
    _: str = Depends(require_admin),
    session: Session = Depends(get_session),
) -> StreamingResponse:
    """Export CSV des réclamations filtrées."""
    _total, items = list_complaints(
        session=session,
        status=status,
        category=category,
        priority=priority,
        search=search,
        limit=200,
        offset=0,
    )

    columns = [
        "reference",
        "created_at",
        "full_name",
        "email",
        "category_label",
        "subject",
        "status_label",
        "priority_label",
    ]

    def rows() -> list[list[str]]:
        exported = []

        for item in items:
            summary = serialize_summary(item)
            summary["created_at"] = (
                summary["created_at"].isoformat()
            )

            exported.append(
                [str(summary.get(column, "")) for column in columns]
            )

        return exported

    return csv_response(
        filename="reclamations-narsa.csv",
        header=[
            "Référence",
            "Déposée le",
            "Citoyen",
            "Email",
            "Catégorie",
            "Objet",
            "Statut",
            "Priorité",
        ],
        rows=rows(),
    )


@router.get("/conversations/export")
def export_conversations(
    limit: int = Query(
        default=200,
        ge=1,
        le=1000,
    ),
    _: str = Depends(require_admin),
    session: Session = Depends(get_session),
) -> StreamingResponse:
    """Export CSV de l'historique des échanges."""
    rows = [
        [
            row["created_at"].isoformat(),
            row["question"],
            row["domain"],
            row["intent"],
            f"{row['confidence']:.2f}",
            f"{row['time_ms']:.0f}",
            row["status"],
        ]
        for row in build_recent_conversations(
            session=session,
            limit=limit,
        )
    ]

    return csv_response(
        filename="conversations-narsa.csv",
        header=[
            "Date",
            "Question",
            "Domaine",
            "Intention",
            "Confiance",
            "Temps (ms)",
            "Statut",
        ],
        rows=rows,
    )


@router.get("/conversations")
def conversations(
    limit: int = Query(
        default=25,
        ge=1,
        le=100,
    ),
    _: str = Depends(require_admin),
    session: Session = Depends(get_session),
) -> dict[str, list[dict]]:
    """Historique récent des échanges avec l'assistant."""
    return {
        "items": build_recent_conversations(
            session=session,
            limit=limit,
        )
    }


@router.get(
    "/complaints",
    response_model=ComplaintListResponse,
)
def complaints(
    status: str | None = None,
    category: str | None = None,
    priority: str | None = None,
    search: str | None = None,
    limit: int = Query(
        default=50,
        ge=1,
        le=200,
    ),
    offset: int = Query(
        default=0,
        ge=0,
    ),
    _: str = Depends(require_admin),
    session: Session = Depends(get_session),
) -> ComplaintListResponse:
    """Liste filtrable des réclamations."""
    total, items = list_complaints(
        session=session,
        status=status,
        category=category,
        priority=priority,
        search=search,
        limit=limit,
        offset=offset,
    )

    return ComplaintListResponse(
        total=total,
        items=[
            serialize_summary(item)
            for item in items
        ],
    )


@router.get(
    "/complaints/{reference}",
    response_model=ComplaintResponse,
)
def complaint_detail(
    reference: str,
    _: str = Depends(require_admin),
    session: Session = Depends(get_session),
) -> ComplaintResponse:
    complaint = get_complaint_by_reference(
        session=session,
        reference=reference,
    )

    if complaint is None:
        raise HTTPException(
            status_code=404,
            detail="Réclamation introuvable.",
        )

    return ComplaintResponse(
        **serialize_complaint(complaint)
    )


@router.patch(
    "/complaints/{reference}",
    response_model=ComplaintResponse,
)
def change_complaint_status(
    reference: str,
    payload: ComplaintStatusUpdateRequest,
    username: str = Depends(require_admin),
    session: Session = Depends(get_session),
) -> ComplaintResponse:
    """Fait évoluer le statut d'une réclamation."""
    complaint = get_complaint_by_reference(
        session=session,
        reference=reference,
    )

    if complaint is None:
        raise HTTPException(
            status_code=404,
            detail="Réclamation introuvable.",
        )

    updated = update_complaint_status(
        session=session,
        complaint=complaint,
        status=payload.status,
        note=payload.note,
        author=username,
    )

    # L'email n'est qu'une notification : son échec ne doit
    # jamais remettre en cause l'enregistrement du statut.
    notification = notify_status_changed(
        to=updated.email,
        reference=updated.reference,
        full_name=updated.full_name,
        status=updated.status,
    )

    refreshed = get_complaint_by_reference(
        session=session,
        reference=updated.reference,
    )

    return ComplaintResponse(
        **serialize_complaint(refreshed or updated),
        notification=notification_status(notification),
    )


@router.post(
    "/complaints/{reference}/response",
    response_model=ComplaintResponse,
)
def answer_complaint(
    reference: str,
    payload: ComplaintResponseRequest,
    username: str = Depends(require_admin),
    session: Session = Depends(get_session),
) -> ComplaintResponse:
    """
    Enregistre la réponse officielle d'un agent.

    La réponse est conservée dans la plateforme et devient
    visible dans le suivi citoyen ; un email prévient le
    citoyen qu'une réponse est disponible.
    """
    complaint = get_complaint_by_reference(
        session=session,
        reference=reference,
    )

    if complaint is None:
        raise HTTPException(
            status_code=404,
            detail="Réclamation introuvable.",
        )

    try:
        updated = add_official_response(
            session=session,
            complaint=complaint,
            response=payload.response,
            author=username,
            status=payload.status,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    # La réponse est déjà enregistrée et committée : un échec
    # d'envoi d'email ne fait perdre aucune donnée métier.
    # Son résultat est remonté à l'agent pour qu'un vrai
    # problème SMTP soit visible immédiatement.
    notification = notify_response_added(
        to=updated.email,
        reference=updated.reference,
        full_name=updated.full_name,
    )

    refreshed = get_complaint_by_reference(
        session=session,
        reference=updated.reference,
    )

    return ComplaintResponse(
        **serialize_complaint(refreshed or updated),
        notification=notification_status(notification),
    )


# ----------------------------------------------------------------------
# Diagnostic des notifications email
# ----------------------------------------------------------------------


@router.get("/email/status")
def email_status(
    _: str = Depends(require_admin),
) -> dict:
    """
    Configuration SMTP effective et dernières tentatives.

    Le mot de passe n'est jamais exposé : seule l'information
    « un mot de passe est configuré » remonte.
    """
    return {
        "configuration": smtp_configuration(),
        "recent": [
            {
                "status": message.get("status"),
                "to": message.get("to"),
                "subject": message.get("subject"),
                "detail": message.get("detail"),
                "error": message.get("error"),
            }
            for message in sent_messages()[-10:]
        ],
    }


@router.post("/email/test")
def email_test(
    payload: EmailTestRequest,
    _: str = Depends(require_admin),
) -> dict:
    """Envoie un email de test pour valider la chaîne SMTP."""
    result = send_test_email(to=str(payload.to))

    return {
        "notification": notification_status(result),
        "configuration": smtp_configuration(),
    }
