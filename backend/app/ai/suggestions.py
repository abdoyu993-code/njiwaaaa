from __future__ import annotations

import csv
import unicodedata
from functools import lru_cache
from pathlib import Path

from backend.app.core.config import settings


TRAIN_DATASET = (
    settings.project_root
    / "backend"
    / "data"
    / "train.csv"
)

MIN_QUESTION_LENGTH = 25
MAX_QUESTION_LENGTH = 110
SUGGESTIONS_PER_ANSWER = 3


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.lower())

    return "".join(
        character
        for character in decomposed
        if unicodedata.category(character) != "Mn"
    )


def _is_readable(question: str) -> bool:
    """Écarte les formulations bruitées du jeu d'entraînement."""
    if not (
        MIN_QUESTION_LENGTH
        <= len(question)
        <= MAX_QUESTION_LENGTH
    ):
        return False

    if not question.endswith("?"):
        return False

    lowered = question.lower()

    noisy_markers = (
        "question concernant",
        "aide pour",
        "besoin d'aide",
        "info sur",
        ". et ",
        "..",
    )

    return not any(
        marker in lowered
        for marker in noisy_markers
    )


@lru_cache(maxsize=1)
def _catalogue() -> dict[str, dict[str, list[str]]]:
    """
    Construit un catalogue de questions lisibles, indexé par
    intention et par domaine, à partir du jeu d'entraînement.
    """
    by_intent: dict[str, list[str]] = {}
    by_domain: dict[str, list[str]] = {}

    if not TRAIN_DATASET.is_file():
        return {"by_intent": by_intent, "by_domain": by_domain}

    seen: set[str] = set()

    with TRAIN_DATASET.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        for row in csv.DictReader(file):
            question = (row.get("text") or "").strip()
            intent = (row.get("intent") or "").strip()
            domain = (row.get("domain") or "").strip()

            if not question or not intent:
                continue

            if not _is_readable(question):
                continue

            key = _normalize(question)

            if key in seen:
                continue

            seen.add(key)
            by_intent.setdefault(intent, []).append(question)
            by_domain.setdefault(domain, []).append(question)

    return {"by_intent": by_intent, "by_domain": by_domain}


def related_questions(
    question: str,
    intent: str | None,
    domain: str | None,
    limit: int = SUGGESTIONS_PER_ANSWER,
) -> list[str]:
    """
    Propose des questions proches réellement traitées par
    l'assistant : d'abord la même intention, puis le même
    domaine pour élargir le sujet.
    """
    catalogue = _catalogue()
    asked = _normalize(question)

    suggestions: list[str] = []
    seen: set[str] = {asked}

    pools = [
        catalogue["by_intent"].get(intent or "", []),
        catalogue["by_domain"].get(domain or "", []),
    ]

    for pool in pools:
        for candidate in pool:
            key = _normalize(candidate)

            if key in seen:
                continue

            # Évite de reproposer une reformulation quasi
            # identique à la question posée.
            if key in asked or asked in key:
                continue

            seen.add(key)
            suggestions.append(candidate)

            if len(suggestions) >= limit:
                return suggestions

    return suggestions
