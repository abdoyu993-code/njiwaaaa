from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.app.analytics.service import record_feedback
from backend.app.database.session import get_session


router = APIRouter(
    prefix="/feedback",
    tags=["Assistant RAG"],
)


class FeedbackRequest(BaseModel):
    interaction_id: int = Field(..., ge=1)
    helpful: bool


class FeedbackResponse(BaseModel):
    recorded: bool
    message: str


@router.post(
    "",
    response_model=FeedbackResponse,
)
def submit_feedback(
    payload: FeedbackRequest,
    session: Session = Depends(get_session),
) -> FeedbackResponse:
    """
    Enregistre l'avis du citoyen sur une réponse.

    L'avis est rattaché à l'interaction journalisée, ce qui
    permet de l'analyser ensuite par domaine et par intention.
    """
    recorded = record_feedback(
        session=session,
        interaction_id=payload.interaction_id,
        helpful=payload.helpful,
    )

    if not recorded:
        raise HTTPException(
            status_code=404,
            detail="Interaction introuvable.",
        )

    return FeedbackResponse(
        recorded=True,
        message=(
            "Merci pour votre retour, il nous aide à améliorer "
            "l'assistant."
        ),
    )
