from __future__ import annotations

from fastapi import FastAPI

from backend.app.routes.ask import (
    router as ask_router,
)
from backend.app.routes.classification import (
    router as classification_router,
)
from backend.app.routes.decision import (
    router as decision_router,
)
from backend.app.routes.forms import (
    router as forms_router,
)


app = FastAPI(
    title="NARSA AI Assistant API",
    description=(
        "API intelligente NARSA avec classification "
        "d'intentions, recherche documentaire RAG, "
        "génération Gemini et téléchargement de formulaires."
    ),
    version="1.2.0",
)


app.include_router(
    classification_router
)
app.include_router(
    decision_router
)
app.include_router(
    ask_router
)
app.include_router(
    forms_router
)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "NARSA AI Assistant API",
        "status": "running",
        "version": "1.2.0",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "healthy",
    }