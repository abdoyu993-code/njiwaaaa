from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.classifier.classifier_service import (
    predict_intent,
)
from backend.app.database.session import get_session
from backend.app.journeys.service import (
    catalogue_summary,
    create_journey,
    detect_journey,
    get_by_reference,
    serialize_journey,
    toggle_step,
)
from backend.app.schemas.journeys import (
    JourneyPlanRequest,
    JourneyPlanResponse,
    JourneyResponse,
    JourneyStepUpdateRequest,
)


router = APIRouter(
    prefix="/journeys",
    tags=["Parcours guidés"],
)


@router.get("/catalogue")
def catalogue() -> dict[str, list[dict[str, Any]]]:
    """Liste des parcours administratifs disponibles."""
    return {"journeys": catalogue_summary()}


@router.post(
    "/plan",
    response_model=JourneyPlanResponse,
)
def plan_journey(
    payload: JourneyPlanRequest,
    session: Session = Depends(get_session),
) -> JourneyPlanResponse:
    """
    Construit un parcours complet à partir d'un objectif
    exprimé librement par le citoyen.
    """
    classification: dict[str, Any] | None = None

    try:
        classification = predict_intent(
            text=payload.goal,
            top_k=3,
        )

    except Exception as error:  # pragma: no cover
        print(
            "Classification de l'objectif impossible :",
            f"{type(error).__name__}: {error}",
        )

    journey_key, confidence, reason = detect_journey(
        goal=payload.goal,
        classification=classification,
    )

    if journey_key is None:
        return JourneyPlanResponse(
            detected=False,
            confidence=0.0,
            reason=reason,
            suggestions=catalogue_summary(),
            message=(
                "Je n'ai pas reconnu de démarche précise dans "
                "votre message. Choisissez un parcours ci-dessous "
                "ou posez votre question à l'assistant."
            ),
        )

    journey = create_journey(
        session=session,
        journey_key=journey_key,
        goal=payload.goal,
        session_id=payload.session_id,
    )

    return JourneyPlanResponse(
        detected=True,
        confidence=confidence,
        reason=reason,
        journey=JourneyResponse(
            **serialize_journey(journey)
        ),
        message=(
            "Voici les étapes de votre démarche. Cochez-les "
            "au fur et à mesure : votre progression est "
            "conservée avec la référence "
            f"{journey.reference}."
        ),
    )


@router.post(
    "/start/{journey_key}",
    response_model=JourneyResponse,
)
def start_journey(
    journey_key: str,
    payload: JourneyPlanRequest | None = None,
    session: Session = Depends(get_session),
) -> JourneyResponse:
    """Démarre directement un parcours choisi dans le catalogue."""
    from backend.app.journeys.catalogue import get_journey

    definition = get_journey(journey_key)

    if definition is None:
        raise HTTPException(
            status_code=404,
            detail="Parcours introuvable.",
        )

    journey = create_journey(
        session=session,
        journey_key=journey_key,
        goal=(
            payload.goal
            if payload and payload.goal
            else definition["title"]
        ),
        session_id=payload.session_id if payload else None,
    )

    return JourneyResponse(**serialize_journey(journey))


@router.get(
    "/{reference}",
    response_model=JourneyResponse,
)
def read_journey(
    reference: str,
    session: Session = Depends(get_session),
) -> JourneyResponse:
    journey = get_by_reference(
        session=session,
        reference=reference,
    )

    if journey is None:
        raise HTTPException(
            status_code=404,
            detail="Parcours introuvable.",
        )

    return JourneyResponse(**serialize_journey(journey))


@router.post(
    "/{reference}/steps/{index}",
    response_model=JourneyResponse,
)
def update_step(
    reference: str,
    index: int,
    payload: JourneyStepUpdateRequest,
    session: Session = Depends(get_session),
) -> JourneyResponse:
    """Coche ou décoche une étape du parcours."""
    journey = get_by_reference(
        session=session,
        reference=reference,
    )

    if journey is None:
        raise HTTPException(
            status_code=404,
            detail="Parcours introuvable.",
        )

    try:
        updated = toggle_step(
            session=session,
            journey=journey,
            index=index,
            done=payload.done,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    return JourneyResponse(**serialize_journey(updated))
