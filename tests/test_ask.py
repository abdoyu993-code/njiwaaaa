from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.routes import ask as ask_route


client = TestClient(app)


def fail_if_pipeline_is_called(
    *args: Any,
    **kwargs: Any,
) -> None:
    raise AssertionError(
        "Le pipeline lourd ne doit pas être appelé."
    )


@pytest.mark.parametrize(
    ("question", "expected_intent"),
    [
        ("bonjour", "salutation"),
        ("Bonjour !", "salutation"),
        ("bonsoir", "salutation"),
        ("salut", "salutation"),
        ("salam", "salutation"),
        ("merci", "remerciement"),
        ("au revoir", "au_revoir"),
    ],
)
def test_simple_messages_are_handled_locally(
    monkeypatch: pytest.MonkeyPatch,
    question: str,
    expected_intent: str,
) -> None:
    monkeypatch.setattr(
        ask_route,
        "generate_answer",
        fail_if_pipeline_is_called,
    )

    response = client.post(
        "/ask",
        json={"question": question},
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["status"] == "local_response"
    assert payload["intent"] == expected_intent
    assert payload["generated"] is False
    assert payload["needs_clarification"] is False
    assert payload["metadata"]["handled_locally"] is True
    assert payload["metadata"]["generation_attempts"] == 0
    assert payload["metadata"]["classification_time_ms"] == 0.0
    assert payload["metadata"]["retrieval_time_ms"] == 0.0
    assert payload["metadata"]["generation_time_ms"] == 0.0
    assert payload["metadata"]["total_time_ms"] >= 0.0


def test_normal_narsa_question_uses_default_top_k_three(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: dict[str, Any] = {}

    def fake_generate_answer(
        question: str,
        top_k: int,
    ) -> dict[str, Any]:
        received["question"] = question
        received["top_k"] = top_k

        return {
            "question": question,
            "answer": "Réponse documentaire NARSA.",
            "status": "context_found",
            "domain": "permis",
            "intent": "permis_perte",
            "confidence": 0.91,
            "needs_clarification": False,
            "generated": True,
            "model": "gemini-test",
            "finish_reason": "STOP",
            "sources": [{
                "title": "Permis perdu",
                "filename": "permis_perte.md",
                "relative_path": "permis/permis_perte.md",
                "domain": "permis",
                "best_similarity_score": 0.91,
            }],
            "retrieval_query": question,
            "used_contexts_count": 3,
            "generation_attempts": 1,
            "classification": {},
            "classification_time_ms": 11.0,
            "retrieval_time_ms": 22.0,
            "generation_time_ms": 33.0,
            "total_time_ms": 66.0,
            "extractive": False,
        }

    monkeypatch.setattr(
        ask_route,
        "generate_answer",
        fake_generate_answer,
    )

    question = (
        "Quels documents faut-il pour remplacer "
        "un permis perdu ?"
    )

    response = client.post(
        "/ask",
        json={"question": question},
    )

    assert response.status_code == 200
    assert received == {
        "question": question,
        "top_k": 3,
    }
    assert response.json()["answer"] == (
        "Réponse documentaire NARSA."
    )
    assert response.json()["sources"][0]["filename"] == (
        "permis_perte.md"
    )
    assert response.json()["metadata"]["classification_time_ms"] == 11.0
    assert response.json()["metadata"]["retrieval_time_ms"] == 22.0
    assert response.json()["metadata"]["generation_time_ms"] == 33.0


@pytest.mark.parametrize(
    ("question", "expected_intent", "expected_filenames"),
    [
        (
            "Je veux le formulaire pour contester une contravention.",
            "contestation_amende",
            ["formulaire_reclamation_contravention.pdf"],
        ),
        (
            "Je cherche les formulaires PDF pour une mutation de véhicule.",
            "mutation_vehicule",
            [
                "formulaire_carte_grise_formule_bi.pdf",
                "formulaire_carte_grise_formule_bii.pdf",
            ],
        ),
        (
            "Quels formulaires faut-il pour immatriculer un véhicule ?",
            "immatriculation_vehicule",
            [
                "formulaire_immatriculation_formule_i.pdf",
                "formulaire_immatriculation_formule_ii.pdf",
            ],
        ),
    ],
)
def test_explicit_form_requests_are_handled_locally(
    monkeypatch: pytest.MonkeyPatch,
    question: str,
    expected_intent: str,
    expected_filenames: list[str],
) -> None:
    monkeypatch.setattr(
        ask_route,
        "generate_answer",
        fail_if_pipeline_is_called,
    )

    response = client.post(
        "/ask",
        json={"question": question},
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["status"] == "form_found"
    assert payload["intent"] == expected_intent
    assert payload["needs_clarification"] is False
    assert (
        "Je ne suis pas suffisamment sûr"
        not in payload["answer"]
    )
    assert [
        form["filename"]
        for form in payload["forms"]
    ] == expected_filenames
    assert payload["metadata"]["handled_locally"] is True
    assert payload["service_links"] == []
    assert payload["domain"] in {"infractions", "vehicule"}
    assert "duplicata-de-mon-permis-de-conduire" not in (
        payload["answer"]
    )


def test_generic_carte_grise_form_requests_clarification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ask_route,
        "generate_answer",
        fail_if_pipeline_is_called,
    )

    response = client.post(
        "/ask",
        json={
            "question": "donne-moi le formulaire pour la carte grise",
        },
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["status"] == "clarification_required"
    assert payload["domain"] == "vehicule"
    assert payload["intent"] == "carte_grise_demande"
    assert payload["needs_clarification"] is True
    assert payload["forms"] == []
    assert payload["service_links"] == []
    assert "immatriculation" in payload["answer"]
    assert "mutation" in payload["answer"]


def test_ambiguous_question_requests_clarification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    question = "J’ai un problème."

    def fake_generate_answer(
        question: str,
        top_k: int,
    ) -> dict[str, Any]:
        return {
            "question": question,
            "answer": (
                "Je ne suis pas suffisamment sûr de votre "
                "demande. Pouvez-vous la reformuler avec "
                "plus de détails ?"
            ),
            "status": "clarification_required",
            "domain": "vehicules",
            "intent": "mutation_vehicule",
            "confidence": 0.31,
            "needs_clarification": True,
            "generated": False,
            "model": None,
            "finish_reason": None,
            "sources": [],
            "retrieval_query": None,
            "used_contexts_count": 0,
            "generation_attempts": 0,
            "classification": {},
        }

    monkeypatch.setattr(
        ask_route,
        "generate_answer",
        fake_generate_answer,
    )

    response = client.post(
        "/ask",
        json={"question": question},
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["status"] == "clarification_required"
    assert payload["needs_clarification"] is True
    assert payload["forms"] == []
