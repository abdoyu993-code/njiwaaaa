from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.routes import ask as ask_route


client = TestClient(app)


def fake_answer(question: str, **kwargs: Any) -> dict[str, Any]:
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


def ask(question: str) -> dict[str, Any]:
    return client.post(
        "/ask",
        json={"question": question, "session_id": "feedback"},
    ).json()


def test_answer_exposes_its_interaction_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ask_route,
        "generate_answer",
        lambda **kwargs: fake_answer(kwargs["question"]),
    )

    payload = ask("Comment récupérer mon permis perdu ?")

    assert isinstance(
        payload["metadata"]["interaction_id"],
        int,
    )


@pytest.mark.parametrize("helpful", [True, False])
def test_feedback_is_recorded(
    monkeypatch: pytest.MonkeyPatch,
    helpful: bool,
) -> None:
    monkeypatch.setattr(
        ask_route,
        "generate_answer",
        lambda **kwargs: fake_answer(kwargs["question"]),
    )

    interaction_id = ask(
        "Comment payer une contravention ?"
    )["metadata"]["interaction_id"]

    response = client.post(
        "/feedback",
        json={
            "interaction_id": interaction_id,
            "helpful": helpful,
        },
    )

    assert response.status_code == 200
    assert response.json()["recorded"] is True

    from backend.app.database.models import ConversationLog
    from backend.app.database.session import SessionLocal

    with SessionLocal() as session:
        entry = session.get(ConversationLog, interaction_id)

        assert entry is not None
        assert entry.feedback == (
            "useful" if helpful else "not_useful"
        )
        assert entry.feedback_at is not None


def test_unknown_interaction_is_rejected() -> None:
    response = client.post(
        "/feedback",
        json={"interaction_id": 999999, "helpful": True},
    )

    assert response.status_code == 404


def test_feedback_feeds_the_dashboard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ask_route,
        "generate_answer",
        lambda **kwargs: fake_answer(kwargs["question"]),
    )

    interaction_id = ask(
        "Quels documents pour la visite technique ?"
    )["metadata"]["interaction_id"]

    client.post(
        "/feedback",
        json={
            "interaction_id": interaction_id,
            "helpful": True,
        },
    )

    token = client.post(
        "/admin/login",
        json={
            "username": "agent",
            "password": "narsa2026",
        },
    ).json()["access_token"]

    dashboard = client.get(
        "/admin/dashboard?days=7",
        headers={"Authorization": f"Bearer {token}"},
    ).json()

    feedback = dashboard["feedback"]

    assert feedback["useful"] >= 1
    assert feedback["total"] >= 1
    assert 0 <= feedback["satisfaction"] <= 100


def test_language_is_journalised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ask_route,
        "generate_answer",
        lambda **kwargs: fake_answer(kwargs["question"]),
    )

    payload = ask("bghit n3ref kifach njdad permis")
    interaction_id = payload["metadata"]["interaction_id"]

    assert payload["metadata"]["language"] == "arabizi"

    from backend.app.database.models import ConversationLog
    from backend.app.database.session import SessionLocal

    with SessionLocal() as session:
        entry = session.get(ConversationLog, interaction_id)

        assert entry is not None
        assert entry.language == "arabizi"
