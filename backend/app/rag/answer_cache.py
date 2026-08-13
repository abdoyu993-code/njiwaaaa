from __future__ import annotations

import re
import threading
import time
import unicodedata
from copy import deepcopy
from pathlib import Path
from typing import Any

from backend.app.core.config import settings


"""
Cache des réponses documentaires.

Deux citoyens qui posent la même question reçoivent la même
réponse : la recalculer (et surtout rappeler Gemini) n'apporte
rien. Le cache est invalidé dès que les documents sources
changent, afin de ne jamais servir une information périmée.

Aucune question contenant des données personnelles n'est mise en
cache (voir `_is_cacheable_question`).
"""


DOCUMENTS_DIR = (
    settings.project_root / "backend" / "data" / "documents"
)

MAX_ENTRIES = 256
DEFAULT_TTL_SECONDS = 3600.0
SIGNATURE_CHECK_INTERVAL = 30.0
MAX_QUESTION_LENGTH = 220

# Séquences pouvant identifier une personne (CIN, téléphone,
# immatriculation, numéro de dossier).
_SENSITIVE_PATTERNS = (
    re.compile(r"\d{5,}"),
    re.compile(r"\b\d{2}[\s.-]?\d{2}[\s.-]?\d{2}[\s.-]?\d{2}\b"),
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    re.compile(r"\b[A-Za-z]{1,2}\s?\d{4,}\b"),
)

_entries: dict[str, dict[str, Any]] = {}
_order: list[str] = []
_lock = threading.Lock()

_signature: str | None = None
_signature_checked_at = 0.0

_hits = 0
_misses = 0
_stored = 0


def normalize_question(question: str) -> str:
    decomposed = unicodedata.normalize("NFD", question.lower())

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


def _is_cacheable_question(question: str) -> bool:
    """Écarte les questions contenant des données personnelles."""
    if len(question) > MAX_QUESTION_LENGTH:
        return False

    return not any(
        pattern.search(question)
        for pattern in _SENSITIVE_PATTERNS
    )


def documents_signature() -> str:
    """Empreinte du corpus : taille et date des Markdown."""
    if not DOCUMENTS_DIR.is_dir():
        return "absent"

    parts = []

    for path in sorted(DOCUMENTS_DIR.rglob("*.md")):
        stat = path.stat()
        parts.append(f"{path.name}:{stat.st_size}:{int(stat.st_mtime)}")

    return "|".join(parts)


def _current_signature() -> str:
    global _signature, _signature_checked_at

    now = time.time()

    if (
        _signature is not None
        and now - _signature_checked_at < SIGNATURE_CHECK_INTERVAL
    ):
        return _signature

    _signature = documents_signature()
    _signature_checked_at = now

    return _signature


def _invalidate_if_documents_changed() -> None:
    global _signature

    signature = _current_signature()

    with _lock:
        stored = _entries.get("__signature__")

        if stored is not None and stored["value"] == signature:
            return

        _entries.clear()
        _order.clear()
        _entries["__signature__"] = {"value": signature}


def build_key(
    question: str,
    top_k: int,
    language: str = "fr",
) -> str:
    return f"{language}|{top_k}|{normalize_question(question)}"


def get(
    question: str,
    top_k: int,
    language: str = "fr",
) -> dict[str, Any] | None:
    """Retourne une réponse déjà calculée, si elle est valable."""
    global _hits, _misses

    if not settings.answer_cache_enabled:
        return None

    if not _is_cacheable_question(question):
        return None

    _invalidate_if_documents_changed()

    key = build_key(question, top_k, language)

    with _lock:
        entry = _entries.get(key)

        if entry is None:
            _misses += 1
            return None

        if time.time() - entry["stored_at"] > DEFAULT_TTL_SECONDS:
            _entries.pop(key, None)

            if key in _order:
                _order.remove(key)

            _misses += 1
            return None

        _hits += 1

        return deepcopy(entry["result"])


def store(
    question: str,
    top_k: int,
    result: dict[str, Any],
    language: str = "fr",
) -> None:
    """Mémorise une réponse documentaire réutilisable."""
    global _stored

    if not settings.answer_cache_enabled:
        return

    if not _is_cacheable_question(question):
        return

    # Une demande de clarification dépend de la formulation :
    # elle est peu coûteuse et n'a pas vocation à être servie
    # de nouveau telle quelle.
    if result.get("status") not in {
        "context_found",
        "no_context_found",
    }:
        return

    _invalidate_if_documents_changed()

    key = build_key(question, top_k, language)
    payload = deepcopy(result)

    with _lock:
        _entries[key] = {
            "result": payload,
            "stored_at": time.time(),
        }

        if key in _order:
            _order.remove(key)

        _order.append(key)
        _stored += 1

        while len(_order) > MAX_ENTRIES:
            oldest = _order.pop(0)
            _entries.pop(oldest, None)


def stats() -> dict[str, Any]:
    with _lock:
        size = max(0, len(_entries) - 1)

    total = _hits + _misses

    return {
        "hits": _hits,
        "misses": _misses,
        "stored": _stored,
        "size": size,
        "hit_rate": round(_hits / total, 3) if total else 0.0,
    }


def clear() -> None:
    """Vide le cache (tests et rechargement documentaire)."""
    global _hits, _misses, _stored, _signature

    with _lock:
        _entries.clear()
        _order.clear()
        _hits = 0
        _misses = 0
        _stored = 0
        _signature = None
