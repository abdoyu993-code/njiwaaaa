from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import (
    DeclarativeBase,
    Session,
    sessionmaker,
)

from backend.app.core.config import settings


class Base(DeclarativeBase):
    """Base déclarative commune à tous les modèles."""


def _engine_kwargs(database_url: str) -> dict:
    if database_url.startswith("sqlite"):
        return {
            "connect_args": {
                "check_same_thread": False,
            },
        }

    return {}


def _prepare_sqlite_directory(database_url: str) -> None:
    prefix = "sqlite:///"

    if not database_url.startswith(prefix):
        return

    file_path = Path(
        database_url[len(prefix):]
    )

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


_prepare_sqlite_directory(settings.database_url)

engine = create_engine(
    settings.database_url,
    future=True,
    **_engine_kwargs(settings.database_url),
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
    future=True,
)


_database_ready = False


def _apply_light_migrations() -> None:
    """
    Ajoute les colonnes apparues après la création de la base.

    Le projet n'utilise pas d'outil de migration : cette
    fonction évite de perdre les réclamations déjà déposées.
    """
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())

    # Colonnes ajoutées après la première mise en service.
    expected: dict[str, dict[str, str]] = {
        "complaints": {
            "priority": "VARCHAR(20) DEFAULT 'normale'",
            "detected_intent": "VARCHAR(80)",
            "official_response": "TEXT",
            "responded_at": "DATETIME",
            "responded_by": "VARCHAR(80)",
        },
        "complaint_events": {
            "event_type": "VARCHAR(20) DEFAULT 'status'",
        },
        "conversation_logs": {
            "language": "VARCHAR(12) DEFAULT 'fr'",
            "feedback": "VARCHAR(16)",
            "feedback_at": "DATETIME",
        },
    }

    statements = []

    for table, columns in expected.items():
        if table not in tables:
            continue

        existing = {
            column["name"]
            for column in inspector.get_columns(table)
        }

        for column, definition in columns.items():
            if column not in existing:
                statements.append(
                    f"ALTER TABLE {table} "
                    f"ADD COLUMN {column} {definition}"
                )

    if not statements:
        return

    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))


def init_database() -> None:
    """Crée les tables manquantes."""
    global _database_ready

    from backend.app.database import models  # noqa: F401

    _apply_light_migrations()
    Base.metadata.create_all(bind=engine)
    _database_ready = True


def ensure_database() -> None:
    """Initialise la base au premier accès si nécessaire."""
    if not _database_ready:
        init_database()


def get_session() -> Iterator[Session]:
    """Dépendance FastAPI fournissant une session."""
    ensure_database()

    session = SessionLocal()

    try:
        yield session
    finally:
        session.close()
