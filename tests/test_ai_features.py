from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from backend.app.ai.suggestions import related_questions
from backend.app.analytics import insights
from backend.app.complaints.triage import (
    analyze_complaint_text,
    detect_priority,
)


def classification(
    domain: str,
    intent: str,
    confidence: float,
) -> dict[str, object]:
    return {
        "domain": domain,
        "intent": intent,
        "confidence": confidence,
        "needs_clarification": confidence < 0.60,
        "alternatives": [],
    }


# --------------------------- Priorités ---------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Un accident a fait un blessé devant le centre.",
        "Situation dangereuse et urgente sur la route.",
    ],
)
def test_safety_situations_are_urgent(text: str) -> None:
    priority, reasons = detect_priority(text)

    assert priority == "urgente"
    assert reasons


@pytest.mark.parametrize(
    "text",
    [
        "Ma demande est sans reponse depuis deux mois.",
        "J'ai fait plusieurs relances sans resultat.",
        "Le delai de 30 jours arrive bientot a echeance.",
    ],
)
def test_stalled_files_are_high_priority(
    text: str,
) -> None:
    priority, reasons = detect_priority(text)

    assert priority == "haute"
    assert reasons


def test_ordinary_request_stays_normal() -> None:
    priority, _ = detect_priority(
        "Je souhaite connaitre les documents pour la carte grise."
    )

    assert priority == "normale"


def test_accident_domain_raises_priority() -> None:
    priority, _ = detect_priority(
        "Je signale un probleme de traitement.",
        domain="accident",
    )

    assert priority == "urgente"


# ------------------------- Pré-diagnostic ------------------------------


def test_reliable_classification_sets_category() -> None:
    analysis = analyze_complaint_text(
        text="Mon duplicata de permis n'est pas arrive.",
        classification=classification(
            "permis",
            "permis_duplicata",
            0.91,
        ),
    )

    assert analysis["category"] == "permis"
    assert analysis["category_label"] == "Permis de conduire"
    assert analysis["category_is_reliable"] is True
    assert analysis["detected_intent"] == "permis_duplicata"


def test_uncertain_classification_falls_back_to_other() -> None:
    analysis = analyze_complaint_text(
        text="Bonjour, j'ai une question generale.",
        classification=classification(
            "prevention",
            "conseils_casque",
            0.21,
        ),
    )

    assert analysis["category"] == "autre"
    assert analysis["category_is_reliable"] is False
    assert analysis["detected_intent"] is None


def test_analysis_without_classification_is_safe() -> None:
    analysis = analyze_complaint_text(
        text="Ma reclamation concerne un retard.",
        classification=None,
    )

    assert analysis["category"] == "autre"
    assert analysis["priority"] in {
        "normale",
        "haute",
        "urgente",
    }


# ------------------------ Questions liées ------------------------------


def test_related_questions_stay_on_topic() -> None:
    suggestions = related_questions(
        question="Comment récupérer mon permis perdu ?",
        intent="permis_perte",
        domain="permis",
    )

    assert 1 <= len(suggestions) <= 3
    assert all(
        question.endswith("?")
        for question in suggestions
    )
    assert all(
        question != "Comment récupérer mon permis perdu ?"
        for question in suggestions
    )


def test_related_questions_handle_unknown_intent() -> None:
    suggestions = related_questions(
        question="Question hors périmètre ?",
        intent="intention_inexistante",
        domain="domaine_inexistant",
    )

    assert suggestions == []


# --------------------------- Angles morts ------------------------------


class FakeLog:
    def __init__(
        self,
        question: str,
        domain: str,
        offset_minutes: int = 0,
    ) -> None:
        self.question = question
        self.domain = domain
        self.created_at = datetime.now(
            timezone.utc
        ) - timedelta(minutes=offset_minutes)


def test_blind_spots_group_similar_questions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logs = [
        FakeLog(
            "Quels documents pour la visite technique ?",
            "visite_technique",
            5,
        ),
        FakeLog(
            "Quels sont les documents demandés pour la visite technique ?",
            "visite_technique",
            3,
        ),
        FakeLog(
            "Comment obtenir une plaque provisoire ?",
            "vehicule",
            1,
        ),
    ]

    monkeypatch.setattr(
        insights,
        "_unanswered_questions",
        lambda session, since: logs,
    )
    monkeypatch.setattr(
        insights,
        "_cluster_questions",
        lambda questions: [[0, 1], [2]],
    )

    themes = insights.build_blind_spots(
        session=None,
        since=datetime.now(timezone.utc)
        - timedelta(days=7),
    )

    assert len(themes) == 2
    assert themes[0]["occurrences"] == 2
    assert themes[0]["domains"] == ["visite_technique"]
    assert themes[1]["occurrences"] == 1


def test_blind_spots_are_empty_without_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        insights,
        "_unanswered_questions",
        lambda session, since: [],
    )

    assert (
        insights.build_blind_spots(
            session=None,
            since=datetime.now(timezone.utc),
        )
        == []
    )
