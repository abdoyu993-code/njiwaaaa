from __future__ import annotations

import pytest

from backend.app.ai.language import (
    LANGUAGE_ARABIC,
    LANGUAGE_ARABIZI,
    LANGUAGE_DARIJA,
    LANGUAGE_FRENCH,
    detect_language,
    response_instruction,
    to_search_query,
)
from backend.app.routes.ask import detect_simple_message


# ----------------------------- Détection ------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Comment récupérer mon permis perdu ?",
        "Quels sont les documents pour la carte grise ?",
        "Bonjour, je voudrais contester une amende.",
    ],
)
def test_french_is_detected(text: str) -> None:
    assert detect_language(text) == LANGUAGE_FRENCH


@pytest.mark.parametrize(
    "text",
    [
        "ما هي الوثائق المطلوبة لرخصة السياقة؟",
        "كيف يمكنني أداء الغرامة؟",
    ],
)
def test_standard_arabic_is_detected(text: str) -> None:
    assert detect_language(text) == LANGUAGE_ARABIC


@pytest.mark.parametrize(
    "text",
    [
        "بغيت نعرف كيفاش نجدد رخصة السياقة",
        "شنو خاصني باش ندير الفحص التقني",
    ],
)
def test_darija_in_arabic_script_is_detected(
    text: str,
) -> None:
    assert detect_language(text) == LANGUAGE_DARIJA


@pytest.mark.parametrize(
    "text",
    [
        "bghit n3ref kifach njdad permis",
        "chno ndir ila tal7at carte grise dyali",
        "wach khassni wra9 bach ndir visite technique",
    ],
)
def test_arabizi_is_detected(text: str) -> None:
    assert detect_language(text) == LANGUAGE_ARABIZI


# --------------------------- Normalisation ----------------------------


def test_french_query_is_unchanged() -> None:
    question = "Comment payer une amende ?"

    assert to_search_query(question) == question


def test_arabic_is_translated_for_search() -> None:
    translated = to_search_query(
        "ما هي الوثائق المطلوبة لرخصة السياقة؟"
    )

    assert "permis de conduire" in translated
    assert "documents" in translated
    # Plus aucun caractère arabe ne subsiste pour la recherche.
    assert not any(
        "؀" <= character <= "ۿ"
        for character in translated
    )


def test_darija_terms_are_translated() -> None:
    translated = to_search_query(
        "بغيت نعرف كيفاش نجدد رخصة السياقة"
    )

    assert "permis de conduire" in translated
    assert "comment" in translated


def test_arabizi_is_translated() -> None:
    translated = to_search_query(
        "bghit n3ref kifach njdad permis"
    )

    assert "je veux" in translated
    assert "comment" in translated
    assert "permis" in translated


def test_carte_grise_arabizi_is_understood() -> None:
    translated = to_search_query(
        "chno ndir ila tal7at carte grise dyali"
    )

    assert "que dois-je faire" in translated
    assert "perdu" in translated
    assert "carte grise" in translated


# ------------------------- Consigne de langue -------------------------


def test_prompt_instruction_matches_language() -> None:
    assert "français" in response_instruction(LANGUAGE_FRENCH)

    arabic_rule = response_instruction(LANGUAGE_ARABIC)

    assert "arabe" in arabic_rule
    assert "NARSA" in arabic_rule
    assert "adresses internet" in arabic_rule


# --------------------- Politesse dans la langue -----------------------


def test_arabic_greeting_answers_in_arabic() -> None:
    result = detect_simple_message("السلام عليكم")

    assert result is not None
    assert result[0] == "salutation"
    assert any(
        "؀" <= character <= "ۿ"
        for character in result[1]
    )


def test_arabizi_greeting_answers_in_arabizi() -> None:
    result = detect_simple_message("salam")

    assert result is not None
    assert result[0] == "salutation"


def test_arabic_thanks_are_recognized() -> None:
    result = detect_simple_message("شكرا")

    assert result is not None
    assert result[0] == "remerciement"


@pytest.mark.parametrize(
    "message",
    [
        "بغيت نعرف كيفاش نجدد رخصة السياقة",
        "chno ndir ila tal7at carte grise dyali",
    ],
)
def test_business_questions_are_not_treated_as_greetings(
    message: str,
) -> None:
    assert detect_simple_message(message) is None
