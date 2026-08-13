from __future__ import annotations

import json
import secrets
import unicodedata
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.journeys.catalogue import JOURNEYS, get_journey
from backend.app.routes.forms import ALLOWED_FORMS


REFERENCE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"

# Score minimal pour retenir un parcours à partir du texte libre.
MIN_MARKER_SCORE = 1


def normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.lower())

    without_accents = "".join(
        character
        for character in decomposed
        if unicodedata.category(character) != "Mn"
    )

    cleaned = "".join(
        character if character.isalnum() else " "
        for character in without_accents
    )

    return " ".join(cleaned.split())


def detect_journey(
    goal: str,
    classification: dict[str, Any] | None = None,
) -> tuple[str | None, float, str]:
    """
    Identifie le parcours correspondant à un objectif exprimé
    librement.

    Retourne la clé du parcours, un score de confiance et
    l'explication de la décision.
    """
    normalized = normalize(goal)
    scores: dict[str, float] = {}

    for key, journey in JOURNEYS.items():
        score = 0.0

        for marker in journey["markers"]:
            if normalize(marker) in normalized:
                # Un marqueur long est plus discriminant.
                score += 1.0 + len(marker) / 100

        if score:
            scores[key] = score

    intent = (classification or {}).get("intent")
    confidence = float(
        (classification or {}).get("confidence", 0.0)
    )

    if intent and confidence >= 0.60:
        for key, journey in JOURNEYS.items():
            if intent in journey["intents"]:
                scores[key] = scores.get(key, 0.0) + 1.5

    if not scores:
        return None, 0.0, "Aucun parcours ne correspond à cette demande."

    best_key = max(scores, key=lambda key: scores[key])
    best_score = scores[best_key]

    if best_score < MIN_MARKER_SCORE:
        return None, 0.0, "Demande trop imprécise pour proposer un parcours."

    reason_parts = []

    matched_markers = [
        marker
        for marker in JOURNEYS[best_key]["markers"]
        if normalize(marker) in normalized
    ]

    if matched_markers:
        reason_parts.append(
            "expressions reconnues : "
            + ", ".join(matched_markers[:3])
        )

    if intent and intent in JOURNEYS[best_key]["intents"]:
        reason_parts.append(
            f"intention détectée : {intent}"
        )

    return (
        best_key,
        min(round(best_score / 3, 2), 1.0),
        " ; ".join(reason_parts) or "correspondance documentaire",
    )


def generate_reference(session: Session) -> str:
    from backend.app.database.models import Journey

    for _ in range(20):
        suffix = "".join(
            secrets.choice(REFERENCE_ALPHABET)
            for _ in range(6)
        )
        reference = f"PARC-{datetime.now(timezone.utc).year}-{suffix}"

        exists = session.scalar(
            select(Journey.id).where(
                Journey.reference == reference
            )
        )

        if not exists:
            return reference

    raise RuntimeError(
        "Impossible de générer une référence de parcours."
    )


def create_journey(
    session: Session,
    journey_key: str,
    goal: str,
    session_id: str | None,
) -> Any:
    from backend.app.database.models import Journey

    journey = Journey(
        reference=generate_reference(session),
        session_id=session_id,
        journey_key=journey_key,
        goal=goal[:500],
        completed_steps="[]",
    )

    session.add(journey)
    session.commit()
    session.refresh(journey)

    return journey


def get_by_reference(
    session: Session,
    reference: str,
) -> Any:
    from backend.app.database.models import Journey

    return session.scalar(
        select(Journey).where(
            Journey.reference == reference.strip().upper()
        )
    )


def toggle_step(
    session: Session,
    journey: Any,
    index: int,
    done: bool,
) -> Any:
    from backend.app.database.models import utc_now

    definition = get_journey(journey.journey_key)

    if definition is None:
        raise ValueError("Parcours inconnu.")

    if not 0 <= index < len(definition["steps"]):
        raise ValueError("Étape inexistante dans ce parcours.")

    completed = set(json.loads(journey.completed_steps or "[]"))

    if done:
        completed.add(index)
    else:
        completed.discard(index)

    journey.completed_steps = json.dumps(sorted(completed))
    journey.updated_at = utc_now()

    session.commit()
    session.refresh(journey)

    return journey


def _form_entry(filename: str) -> dict[str, str] | None:
    metadata = ALLOWED_FORMS.get(filename)

    if not metadata:
        return None

    return {
        "title": metadata["title"],
        "filename": filename,
        "download_url": f"/forms/{filename}",
    }


def serialize_journey(journey: Any) -> dict[str, Any]:
    """Assemble la définition du parcours et son avancement."""
    definition = get_journey(journey.journey_key)

    if definition is None:
        raise ValueError("Parcours inconnu.")

    completed = set(json.loads(journey.completed_steps or "[]"))
    steps = []

    for index, step in enumerate(definition["steps"]):
        forms = [
            entry
            for entry in (
                _form_entry(filename)
                for filename in step.get("forms", [])
            )
            if entry
        ]

        steps.append(
            {
                "index": index,
                "title": step["title"],
                "detail": step["detail"],
                "duration": step.get("duration"),
                "documents": list(step.get("documents", [])),
                "links": list(step.get("links", [])),
                "forms": forms,
                "source": step.get("source"),
                "done": index in completed,
            }
        )

    total = len(steps)
    done_count = len(
        [step for step in steps if step["done"]]
    )
    next_step = next(
        (step for step in steps if not step["done"]),
        None,
    )

    return {
        "reference": journey.reference,
        "journey_key": journey.journey_key,
        "title": definition["title"],
        "summary": definition["summary"],
        "domain": definition["domain"],
        "goal": journey.goal,
        "deadline": definition.get("deadline"),
        "warning": definition.get("warning"),
        "steps": steps,
        "completed_count": done_count,
        "total_steps": total,
        "progress": (
            round(done_count / total * 100)
            if total
            else 0
        ),
        "next_step": next_step,
        "created_at": journey.created_at,
        "updated_at": journey.updated_at,
    }


def catalogue_summary() -> list[dict[str, Any]]:
    return [
        {
            "journey_key": key,
            "title": journey["title"],
            "summary": journey["summary"],
            "domain": journey["domain"],
            "total_steps": len(journey["steps"]),
        }
        for key, journey in JOURNEYS.items()
    ]
