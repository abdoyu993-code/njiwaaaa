from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.app.classifier.classifier_service import (
    predict_intent,
)
from backend.app.schemas.classification import (
    ClassificationRequest,
    ClassificationResponse,
)


router = APIRouter(
    prefix="/classify",
    tags=["Classification"],
)


@router.post(
    "",
    response_model=ClassificationResponse,
)
def classify_text(
    request: ClassificationRequest,
) -> ClassificationResponse:
    try:
        result = predict_intent(
            text=request.text,
            top_k=request.top_k,
        )

        return ClassificationResponse(**result)

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur de classification : {error}",
        ) from error