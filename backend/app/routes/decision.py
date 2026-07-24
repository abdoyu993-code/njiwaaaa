from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.app.decision.decision_router import (
    route_question,
)
from backend.app.schemas.decision import (
    DecisionRequest,
    DecisionResponse,
)


router = APIRouter(
    prefix="/route",
    tags=["Decision"],
)


@router.post(
    "",
    response_model=DecisionResponse,
)
def decide_route(
    request: DecisionRequest,
) -> DecisionResponse:
    try:
        result = route_question(
            text=request.text,
            top_k=request.top_k,
        )

        return DecisionResponse(**result)

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur du routeur : {error}",
        ) from error