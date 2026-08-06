from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.rag import rag_service
from backend.app.rag.rag_service import NarsaRAGService
from backend.app.routes import ask as ask_route


client = TestClient(app)

PERMIT_CONTEXT = {
    "previous_question": (
        "J’ai perdu mon permis de conduire, que dois-je faire ?"
    ),
    "previous_domain": "permis",
    "previous_intent": "permis_perte",
}


def test_b_permit_form_followup_returns_verified_service_link(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ask_route,
        "generate_answer",
        lambda **kwargs: pytest.fail(
            "Le pipeline ne doit pas être appelé."
        ),
    )

    response = client.post(
        "/ask",
        json={
            "question": "donne moi le formulaire",
            "context": PERMIT_CONTEXT,
        },
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["needs_clarification"] is False
    assert payload["status"] == "contextual_resource_found"
    assert payload["forms"] == []
    assert len(payload["service_links"]) == 1
    assert payload["service_links"][0]["label"] == (
        "Accéder au service officiel de duplicata du permis"
    )
    assert payload["service_links"][0]["url"] == (
        "https://khadamatnarsa.ma/fr/services/"
        "duplicata-de-mon-permis-de-conduire"
    )
    assert payload["service_links"][0]["url"] in payload["answer"]
    assert ".pdf" not in payload["answer"].lower()


@pytest.mark.parametrize(
    (
        "question",
        "expected_status",
        "expected_intent",
        "expected_domain",
        "expected_filenames",
        "expects_clarification",
    ),
    [
        (
            "donne-moi le formulaire pour la carte grise",
            "clarification_required",
            "carte_grise_demande",
            "vehicule",
            [],
            True,
        ),
        (
            "je veux télécharger le formulaire pour contester une contravention",
            "form_found",
            "contestation_amende",
            "infractions",
            ["formulaire_reclamation_contravention.pdf"],
            False,
        ),
        (
            "donne-moi le formulaire d’immatriculation",
            "form_found",
            "immatriculation_vehicule",
            "vehicule",
            [
                "formulaire_immatriculation_formule_i.pdf",
                "formulaire_immatriculation_formule_ii.pdf",
            ],
            False,
        ),
    ],
)
def test_explicit_form_topic_overrides_permit_context(
    monkeypatch: pytest.MonkeyPatch,
    question: str,
    expected_status: str,
    expected_intent: str,
    expected_domain: str,
    expected_filenames: list[str],
    expects_clarification: bool,
) -> None:
    monkeypatch.setattr(
        ask_route,
        "generate_answer",
        lambda **kwargs: pytest.fail(
            "Le pipeline ne doit pas être appelé."
        ),
    )

    response = client.post(
        "/ask",
        json={
            "question": question,
            "context": PERMIT_CONTEXT,
        },
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["status"] == expected_status
    assert payload["intent"] == expected_intent
    assert payload["domain"] == expected_domain
    assert payload["needs_clarification"] is expects_clarification
    assert [
        form["filename"]
        for form in payload["forms"]
    ] == expected_filenames
    assert payload["service_links"] == []
    assert "duplicata-de-mon-permis-de-conduire" not in (
        payload["answer"]
    )

    if expected_intent == "carte_grise_demande":
        assert "immatriculation" in payload["answer"]
        assert "mutation" in payload["answer"]


def test_dependent_documents_question_reuses_previous_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: dict[str, Any] = {}

    def fake_generate_answer(
        question: str,
        top_k: int,
    ) -> dict[str, Any]:
        received["question"] = question
        return _fake_answer(question, "permis", "permis_perte")

    monkeypatch.setattr(
        ask_route,
        "generate_answer",
        fake_generate_answer,
    )

    response = client.post(
        "/ask",
        json={
            "question": "et les documents ?",
            "context": PERMIT_CONTEXT,
        },
    )

    assert response.status_code == 200
    assert PERMIT_CONTEXT["previous_question"] in received["question"]
    assert "et les documents ?" in received["question"]
    assert response.json()["metadata"]["context_used"] is True


def test_f_independent_question_ignores_previous_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: dict[str, Any] = {}

    def fake_generate_answer(
        question: str,
        top_k: int,
    ) -> dict[str, Any]:
        received["question"] = question
        return _fake_answer(
            question,
            "infractions",
            "paiement_amende",
        )

    monkeypatch.setattr(
        ask_route,
        "generate_answer",
        fake_generate_answer,
    )
    question = "Comment payer une amende ?"

    response = client.post(
        "/ask",
        json={
            "question": question,
            "context": PERMIT_CONTEXT,
        },
    )

    assert response.status_code == 200
    assert received["question"] == question
    assert response.json()["metadata"]["context_used"] is False


def test_e_general_carte_grise_requests_case_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    classification = {
        "domain": "vehicule",
        "intent": "carte_grise_demande",
        "confidence": 0.91,
        "needs_clarification": False,
        "alternatives": [],
    }
    monkeypatch.setattr(
        rag_service,
        "predict_intent",
        lambda text, top_k: classification,
    )
    service = object.__new__(NarsaRAGService)
    service.clarification_threshold = 0.60
    service.minimum_similarity_score = 0.30

    result = service.search(
        "Quels documents faut-il pour obtenir une carte grise ?"
    )

    assert result["status"] == "clarification_required"
    assert "première immatriculation" in result["message"]
    assert "mutation" in result["message"]
    assert "échange" in result["message"]
    assert "duplicata" in result["message"]
    assert "khadamatnarsa.ma/fr/services/carte-grise" not in (
        result["message"]
    )


def _fake_answer(
    question: str,
    domain: str,
    intent: str,
) -> dict[str, Any]:
    return {
        "question": question,
        "answer": "Réponse complète.",
        "status": "context_found",
        "domain": domain,
        "intent": intent,
        "confidence": 0.9,
        "needs_clarification": False,
        "generated": False,
        "model": None,
        "finish_reason": None,
        "sources": [],
        "retrieval_query": question,
        "used_contexts_count": 1,
        "generation_attempts": 0,
        "classification": {},
        "classification_time_ms": 1.0,
        "retrieval_time_ms": 1.0,
        "generation_time_ms": 0.0,
        "total_time_ms": 2.0,
        "extractive": True,
    }
