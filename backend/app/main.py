from __future__ import annotations

import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.core.config import settings
from backend.app.database.session import init_database
from backend.app.routes.admin import (
    router as admin_router,
)
from backend.app.routes.ask import (
    router as ask_router,
)
from backend.app.routes.classification import (
    router as classification_router,
)
from backend.app.routes.complaints import (
    router as complaints_router,
)
from backend.app.routes.decision import (
    router as decision_router,
)
from backend.app.routes.documents import (
    router as documents_router,
)
from backend.app.routes.feedback import (
    router as feedback_router,
)
from backend.app.routes.forms import (
    router as forms_router,
)
from backend.app.routes.journeys import (
    router as journeys_router,
)


def _preload_models() -> None:
    """Charge les modèles en tâche de fond au démarrage."""
    try:
        from backend.app.rag.generation_service import (
            get_generation_service,
        )

        get_generation_service()
        print("Modèles NARSA préchargés.")

    except Exception as error:  # pragma: no cover
        print(
            "Préchargement des modèles impossible :",
            f"{type(error).__name__}: {error}",
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_database()

    if settings.preload_models:
        threading.Thread(
            target=_preload_models,
            name="narsa-preload",
            daemon=True,
        ).start()

    yield


app = FastAPI(
    title=settings.app_name,
    description=(
        "API de la plateforme NARSA : assistant "
        "conversationnel (classification d'intentions, "
        "recherche documentaire RAG, génération Gemini), "
        "dépôt et suivi des réclamations, espace "
        "administration et formulaires officiels."
    ),
    version=settings.app_version,
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(classification_router)
app.include_router(decision_router)
app.include_router(ask_router)
app.include_router(feedback_router)
app.include_router(forms_router)
app.include_router(complaints_router)
app.include_router(journeys_router)
app.include_router(documents_router)
app.include_router(admin_router)


@app.get(
    "/api",
    tags=["Statut"],
)
def api_root() -> dict[str, object]:
    return {
        "message": settings.app_name,
        "status": "running",
        "version": settings.app_version,
        "generation_enabled": settings.gemini_enabled,
        "endpoints": [
            "/ask",
            "/feedback",
            "/classify",
            "/route",
            "/forms",
            "/complaints",
            "/complaints/analyze",
            "/complaints/track",
            "/journeys/plan",
            "/journeys/catalogue",
            "/documents/content",
            "/admin/login",
            "/admin/dashboard",
            "/admin/insights",
            "/docs",
        ],
    }


@app.get(
    "/stats",
    tags=["Statut"],
)
def public_stats() -> dict[str, object]:
    """Chiffres clés publics affichés sur la page d'accueil."""
    from sqlalchemy import func, select

    from backend.app.database.models import (
        Complaint,
        ConversationLog,
    )
    from backend.app.database.session import (
        SessionLocal,
        ensure_database,
    )

    try:
        ensure_database()

        with SessionLocal() as session:
            complaints = int(
                session.scalar(
                    select(func.count(Complaint.id))
                )
                or 0
            )
            treated = int(
                session.scalar(
                    select(func.count(Complaint.id)).where(
                        Complaint.status == "traitee"
                    )
                )
                or 0
            )
            conversations = int(
                session.scalar(
                    select(func.count(ConversationLog.id))
                )
                or 0
            )
            sourced = int(
                session.scalar(
                    select(
                        func.count(ConversationLog.id)
                    ).where(
                        ConversationLog.status
                        == "context_found"
                    )
                )
                or 0
            )

    except Exception as error:  # pragma: no cover
        print(
            "Statistiques publiques indisponibles :",
            f"{type(error).__name__}: {error}",
        )
        return {
            "complaints": 0,
            "complaints_treated": 0,
            "conversations": 0,
            "sourced_rate": 0,
        }

    return {
        "complaints": complaints,
        "complaints_treated": treated,
        "conversations": conversations,
        "sourced_rate": (
            round(sourced / conversations * 100)
            if conversations
            else 0
        ),
    }


@app.get(
    "/health",
    tags=["Statut"],
)
def health() -> dict[str, object]:
    return {
        "status": "healthy",
        "version": settings.app_version,
        "generation_enabled": settings.gemini_enabled,
    }


if (
    settings.serve_frontend
    and settings.frontend_dir.is_dir()
):
    # Interface web servie directement par l'API.
    # Monté en dernier pour ne pas masquer les routes.
    app.mount(
        "/",
        StaticFiles(
            directory=str(settings.frontend_dir),
            html=True,
        ),
        name="frontend",
    )
