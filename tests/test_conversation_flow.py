from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.routes import ask as ask_route
from backend.app.routes.ask import (
    detect_explicit_form_intent,
    has_explicit_business_topic,
    is_context_dependent_message,
)


client = TestClient(app)


# --------------------- Contexte conversationnel -----------------------


@pytest.mark.parametrize(
    "message",
    [
        "Et le formulaire ?",
        "Et où déposer le dossier ?",
        "Et les documents ?",
        "Et le délai ?",
        "Ok et ensuite ?",
        "Combien ça coûte ?",
        "Quel est le lien ?",
    ],
)
def test_followups_stay_in_context(message: str) -> None:
    assert is_context_dependent_message(message) is True


@pytest.mark.parametrize(
    "message",
    [
        "Je veux faire une mutation de véhicule.",
        "Comment récupérer mon permis perdu ?",
        "Quels documents pour la visite technique ?",
        "Comment contester une contravention ?",
    ],
)
def test_new_topic_is_not_a_followup(message: str) -> None:
    assert has_explicit_business_topic(message) is True


def test_chained_followups_keep_the_same_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    « mutation » puis « et le formulaire ? » puis « et où
    déposer le dossier ? » doivent rester sur la mutation.
    """
    received: list[str] = []

    def fake_generate_answer(
        question: str,
        top_k: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        received.append(question)

        return {
            "question": question,
            "answer": "Réponse documentaire.",
            "status": "context_found",
            "domain": "vehicule",
            "intent": "mutation_vehicule",
            "confidence": 0.93,
            "needs_clarification": False,
            "generated": False,
            "model": None,
            "finish_reason": None,
            "sources": [],
            "classification": {},
            "retrieval_query": question,
            "used_contexts_count": 1,
            "generation_attempts": 0,
            "classification_time_ms": 1.0,
            "retrieval_time_ms": 1.0,
            "generation_time_ms": 0.0,
            "total_time_ms": 2.0,
        }

    monkeypatch.setattr(
        ask_route,
        "generate_answer",
        fake_generate_answer,
    )

    first = client.post(
        "/ask",
        json={
            "question": "Je veux faire une mutation de véhicule",
        },
    ).json()

    context = {
        "previous_question": first["question"],
        "previous_domain": first["domain"],
        "previous_intent": first["intent"],
    }

    # « Et le formulaire ? » est servi localement (PDF mutation).
    forms = client.post(
        "/ask",
        json={
            "question": "Et le formulaire ?",
            "context": context,
        },
    ).json()

    assert forms["intent"] == "mutation_vehicule"
    assert [form["filename"] for form in forms["forms"]] == [
        "formulaire_carte_grise_formule_bi.pdf",
        "formulaire_carte_grise_formule_bii.pdf",
    ]

    # La question suivante reste rattachée à la mutation.
    client.post(
        "/ask",
        json={
            "question": "Et où déposer le dossier ?",
            "context": context,
        },
    )

    assert "mutation" in received[-1].lower()
    assert "déposer le dossier" in received[-1]


def test_explicit_new_topic_replaces_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[str] = []

    def fake_generate_answer(
        question: str,
        top_k: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        received.append(question)

        return {
            "question": question,
            "answer": "Réponse documentaire.",
            "status": "context_found",
            "domain": "permis",
            "intent": "permis_perte",
            "confidence": 0.9,
            "needs_clarification": False,
            "generated": False,
            "model": None,
            "finish_reason": None,
            "sources": [],
            "classification": {},
            "retrieval_query": question,
            "used_contexts_count": 1,
            "generation_attempts": 0,
            "classification_time_ms": 1.0,
            "retrieval_time_ms": 1.0,
            "generation_time_ms": 0.0,
            "total_time_ms": 2.0,
        }

    monkeypatch.setattr(
        ask_route,
        "generate_answer",
        fake_generate_answer,
    )

    client.post(
        "/ask",
        json={
            "question": "Comment se passe la visite technique ?",
            "context": {
                "previous_question": "Je veux faire une mutation",
                "previous_domain": "vehicule",
                "previous_intent": "mutation_vehicule",
            },
        },
    )

    # Le nouveau sujet ne doit pas traîner l'ancien contexte.
    assert "mutation" not in received[-1].lower()


# ------------------------- Demandes de formulaires --------------------


@pytest.mark.parametrize(
    "message,expected",
    [
        ("donne moi pdf mutation", "mutation_vehicule"),
        (
            "formulaire pour reclamé une contravention",
            "contestation_amende",
        ),
        (
            "je veux le formulaire d'immatriculation",
            "immatriculation_vehicule",
        ),
        (
            "jai perdu mon permis donne moi formulaire",
            "permis_duplicata_service",
        ),
        (
            "je veux formulaire carte grise",
            "carte_grise_demande",
        ),
    ],
)
def test_natural_form_requests_are_understood(
    message: str,
    expected: str,
) -> None:
    assert detect_explicit_form_intent(message) == expected


@pytest.mark.parametrize(
    "message",
    [
        "Comment payer une amende ?",
        "Quels sont les délais de la visite technique ?",
        "Bonjour",
    ],
)
def test_questions_without_form_request_are_ignored(
    message: str,
) -> None:
    assert detect_explicit_form_intent(message) is None


def test_generic_carte_grise_form_asks_for_clarification() -> None:
    response = client.post(
        "/ask",
        json={"question": "je veux formulaire carte grise"},
    )

    payload = response.json()

    assert payload["status"] == "clarification_required"
    assert payload["forms"] == []
    assert "immatriculation" in payload["answer"]
    assert "mutation" in payload["answer"]


def test_permit_form_request_returns_official_link() -> None:
    response = client.post(
        "/ask",
        json={
            "question": (
                "jai perdu mon permis donne moi formulaire"
            )
        },
    )

    payload = response.json()

    assert payload["status"] == "service_link_found"
    assert payload["forms"] == []
    assert payload["service_links"][0]["url"].startswith(
        "https://khadamatnarsa.ma"
    )
    # Aucun formulaire n'est inventé pour cette démarche.
    assert ".pdf" not in payload["answer"].lower()


def test_no_technical_identifier_leaks_in_answers() -> None:
    from backend.app.rag.generation_service import (
        NarsaGenerationService,
    )

    raw = (
        "Voici la procédure.\n"
        "intent: declaration_accident\n"
        "La déclaration doit être faite rapidement.\n"
        "Procedure_apres_accident\n"
        "{'domain': 'accident', 'score': 0.87}\n"
    )

    cleaned = NarsaGenerationService._sanitize_answer(raw)

    assert "intent:" not in cleaned
    assert "declaration_accident" not in cleaned
    assert "Procedure_apres_accident" not in cleaned
    assert "'score'" not in cleaned
    assert "La déclaration doit être faite" in cleaned
