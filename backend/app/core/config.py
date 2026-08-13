from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[3]

load_dotenv(PROJECT_ROOT / ".env")


def _get_bool(
    name: str,
    default: bool = False,
) -> bool:
    raw = os.getenv(name)

    if raw is None:
        return default

    return raw.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
        "oui",
    }


def _get_int(
    name: str,
    default: int,
) -> int:
    raw = os.getenv(name)

    if raw is None or not raw.strip():
        return default

    try:
        return int(raw.strip())
    except ValueError:
        return default


class Settings:
    """Configuration applicative lue depuis l'environnement."""

    def __init__(self) -> None:
        self.project_root = PROJECT_ROOT

        # --- API ---
        self.app_name = "NARSA AI Assistant API"
        self.app_version = "2.0.0"
        self.cors_origins = [
            origin.strip()
            for origin in os.getenv(
                "CORS_ORIGINS",
                "*",
            ).split(",")
            if origin.strip()
        ]

        # --- Gemini ---
        self.gemini_api_key = (
            os.getenv("GEMINI_API_KEY") or ""
        ).strip()
        self.gemini_model = (
            os.getenv("GEMINI_MODEL")
            or "gemini-3.5-flash"
        ).strip()

        # Modèles de repli essayés quand le quota du modèle
        # principal est épuisé (chaque modèle a son propre quota).
        self.gemini_fallback_models = [
            model.strip()
            for model in (
                os.getenv("GEMINI_FALLBACK_MODELS")
                or "gemini-3.5-flash-lite,gemini-2.5-flash"
            ).split(",")
            if model.strip()
        ]

        # Durée de mise au repos d'un modèle après un 429.
        self.gemini_quota_cooldown_seconds = float(
            _get_int("GEMINI_QUOTA_COOLDOWN_SECONDS", 900)
        )

        # --- Base de données ---
        self.database_url = (
            os.getenv("DATABASE_URL")
            or f"sqlite:///{PROJECT_ROOT / 'backend' / 'data' / 'narsa.db'}"
        ).strip()

        # --- Espace administration ---
        self.admin_username = (
            os.getenv("ADMIN_USERNAME") or "agent"
        ).strip()
        self.admin_password = (
            os.getenv("ADMIN_PASSWORD") or "narsa2026"
        ).strip()
        self.admin_display_name = (
            os.getenv("ADMIN_DISPLAY_NAME")
            or "Agent NARSA"
        ).strip()
        self.admin_token_ttl_minutes = _get_int(
            "ADMIN_TOKEN_TTL_MINUTES",
            480,
        )
        self.admin_secret = (
            os.getenv("ADMIN_SECRET")
            or "narsa-secret-local"
        ).strip()

        # --- Démarrage ---
        # Précharge les modèles au lancement pour éviter
        # une première question très lente.
        self.preload_models = _get_bool(
            "PRELOAD_MODELS",
            True,
        )

        # --- Notifications email ---
        self.email_enabled = _get_bool(
            "EMAIL_ENABLED",
            True,
        )
        self.smtp_host = (
            os.getenv("SMTP_HOST") or ""
        ).strip()
        self.smtp_port = _get_int("SMTP_PORT", 587)
        self.smtp_username = (
            os.getenv("SMTP_USERNAME") or ""
        ).strip()
        self.smtp_password = os.getenv("SMTP_PASSWORD") or ""
        self.smtp_use_tls = _get_bool("SMTP_USE_TLS", True)
        self.smtp_use_ssl = _get_bool("SMTP_USE_SSL", False)
        self.smtp_timeout = float(
            _get_int("SMTP_TIMEOUT", 10)
        )
        self.email_sender = (
            os.getenv("EMAIL_SENDER")
            or "no-reply@narsa.gov.ma"
        ).strip()
        self.portal_url = (
            os.getenv("PORTAL_URL")
            or "http://127.0.0.1:8000"
        ).strip().rstrip("/")

        # --- Cache des réponses ---
        self.answer_cache_enabled = _get_bool(
            "ANSWER_CACHE_ENABLED",
            True,
        )

        # --- Journalisation des conversations ---
        self.log_conversations = _get_bool(
            "LOG_CONVERSATIONS",
            True,
        )

        # --- Frontend statique ---
        self.serve_frontend = _get_bool(
            "SERVE_FRONTEND",
            True,
        )
        self.frontend_dir = (
            PROJECT_ROOT / "frontend" / "web"
        )

    @property
    def gemini_enabled(self) -> bool:
        return bool(self.gemini_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
