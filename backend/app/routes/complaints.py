from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.classifier.classifier_service import (
    predict_intent,
)
from backend.app.complaints.service import (
    create_complaint,
    get_complaint_by_reference,
    serialize_complaint,
)
from backend.app.complaints.triage import (
    analyze_complaint_text,
)
from backend.app.database.session import get_session
from backend.app.notifications.email import (
    notify_complaint_created,
)
from backend.app.rag.generation_service import (
    generate_answer,
)
from backend.app.schemas.complaints import (
    COMPLAINT_CATEGORY_LABELS,
    ComplaintAnalysisRequest,
    ComplaintAnalysisResponse,
    ComplaintCreatedResponse,
    ComplaintCreateRequest,
    ComplaintResponse,
    ComplaintTrackingRequest,
)


router = APIRouter(
    prefix="/complaints",
    tags=["Réclamations"],
)


@router.get("/categories")
def list_categories() -> dict[str, list[dict[str, str]]]:
    """Catégories proposées dans le formulaire de dépôt."""
    return {
        "categories": [
            {
                "value": value,
                "label": label,
            }
            for value, label in (
                COMPLAINT_CATEGORY_LABELS.items()
            )
        ]
    }


def analyze_text(
    text: str,
    with_instant_answer: bool,
) -> dict[str, Any]:
    """
    Pré-diagnostic d'une réclamation : catégorie, priorité et,
    si la demande est déjà documentée, réponse immédiate.
    """
    classification: dict[str, Any] | None = None

    try:
        classification = predict_intent(
            text=text,
            top_k=3,
        )

    except Exception as error:  # pragma: no cover
        print(
            "Classification de la réclamation impossible :",
            f"{type(error).__name__}: {error}",
        )

    analysis = analyze_complaint_text(
        text=text,
        classification=classification,
    )
    analysis["instant_answer"] = None

    if not with_instant_answer:
        return analysis

    try:
        result = generate_answer(
            question=text,
            top_k=3,
        )

    except Exception as error:  # pragma: no cover
        print(
            "Réponse immédiate indisponible :",
            f"{type(error).__name__}: {error}",
        )
        return analysis

    # Une réponse n'est proposée que si elle s'appuie
    # réellement sur des documents.
    if result.get("status") != "context_found":
        return analysis

    analysis["instant_answer"] = {
        "answer": result["answer"],
        "status": result["status"],
        "domain": result.get("domain"),
        "intent": result.get("intent"),
        "confidence": result.get("confidence", 0.0),
        "sources": [
            source.get("title")
            or source.get("filename")
            or "source"
            for source in result.get("sources", [])[:3]
        ],
    }

    return analysis


@router.post(
    "/analyze",
    response_model=ComplaintAnalysisResponse,
)
def analyze_complaint(
    payload: ComplaintAnalysisRequest,
) -> ComplaintAnalysisResponse:
    """
    Analyse la demande avant dépôt : catégorie proposée,
    priorité estimée et réponse immédiate si le sujet est
    déjà couvert par les documents officiels.
    """
    text = " ".join(
        part
        for part in [payload.subject, payload.message]
        if part
    )

    return ComplaintAnalysisResponse(
        **analyze_text(
            text=text,
            with_instant_answer=True,
        )
    )


@router.post(
    "",
    response_model=ComplaintCreatedResponse,
    status_code=201,
)
def submit_complaint(
    payload: ComplaintCreateRequest,
    session: Session = Depends(get_session),
) -> ComplaintCreatedResponse:
    """Dépose une nouvelle réclamation et retourne sa référence."""
    analysis = analyze_text(
        text=f"{payload.subject} {payload.message}",
        with_instant_answer=False,
    )

    try:
        complaint = create_complaint(
            session=session,
            payload=payload,
            analysis=analysis,
        )

    except RuntimeError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error

    # Notification d'accusé de réception : un échec d'envoi ne
    # remet pas en cause l'enregistrement de la réclamation.
    notify_complaint_created(
        to=complaint.email,
        reference=complaint.reference,
        full_name=complaint.full_name,
    )

    return ComplaintCreatedResponse(
        reference=complaint.reference,
        status=complaint.status,
        status_label=complaint.status_label,
        created_at=complaint.created_at,
        message=(
            "Votre réclamation a bien été enregistrée. "
            "Conservez la référence "
            f"{complaint.reference} pour suivre son "
            "traitement."
        ),
    )


@router.post(
    "/track",
    response_model=ComplaintResponse,
)
def track_complaint(
    payload: ComplaintTrackingRequest,
    session: Session = Depends(get_session),
) -> ComplaintResponse:
    """Consulte l'état d'une réclamation (référence + email)."""
    complaint = get_complaint_by_reference(
        session=session,
        reference=payload.reference,
        email=str(payload.email),
    )

    if complaint is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Aucune réclamation ne correspond à cette "
                "référence et à cette adresse email."
            ),
        )

    return ComplaintResponse(
        **serialize_complaint(complaint)
    )
