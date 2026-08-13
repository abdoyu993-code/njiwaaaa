from __future__ import annotations

import pytest

from backend.app.rag.rag_service import (
    NarsaRAGService,
    format_coverage,
)
from backend.app.routes.ask import detect_simple_message


@pytest.mark.parametrize(
    "message",
    [
        "bonjour",
        "Salam",
        "salam alaykom",
        "Ahlan",
        "coucou",
        "sbah lkhir",
    ],
)
def test_greetings_are_answered_locally(
    message: str,
) -> None:
    result = detect_simple_message(message)

    assert result is not None
    assert result[0] == "salutation"


@pytest.mark.parametrize(
    "message",
    [
        "merci",
        "Chokran",
        "chokran bzaf",
        "merci beaucoup",
    ],
)
def test_thanks_are_recognized(message: str) -> None:
    result = detect_simple_message(message)

    assert result is not None
    assert result[0] == "remerciement"


@pytest.mark.parametrize(
    "message",
    [
        "au revoir",
        "bslama",
        "bonne journee",
    ],
)
def test_goodbyes_are_recognized(message: str) -> None:
    result = detect_simple_message(message)

    assert result is not None
    assert result[0] == "au_revoir"


@pytest.mark.parametrize(
    "message",
    [
        "bonjour, comment récupérer mon permis perdu ?",
        "salam, chno khass l carte grise",
        "Quels documents pour la visite technique ?",
        "Comment payer une amende ?",
    ],
)
def test_business_questions_reach_the_rag(
    message: str,
) -> None:
    assert detect_simple_message(message) is None


def test_no_context_message_guides_the_user() -> None:
    message = NarsaRAGService._build_no_context_message(
        "permis"
    )

    assert "documents officiels" in message
    assert "duplicata" in message
    assert "réclamation" in message


def test_no_context_message_handles_unknown_domain() -> None:
    message = NarsaRAGService._build_no_context_message(
        "domaine_inconnu"
    )

    assert message
    assert "réclamation" in message


def test_coverage_listing_is_readable() -> None:
    coverage = format_coverage("infractions")

    assert coverage.startswith("- ")
    assert coverage.endswith(".")
    assert coverage.count("\n") == 2
