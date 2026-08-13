from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(app)


def build_payload(**overrides: Any) -> dict[str, Any]:
    payload = {
        "full_name": "Youssef El Amrani",
        "email": "youssef@example.ma",
        "phone": "0600000000",
        "city": "Rabat",
        "category": "permis",
        "subject": "Retard de duplicata",
        "message": (
            "Ma demande de duplicata de permis est "
            "sans réponse depuis deux mois."
        ),
    }
    payload.update(overrides)

    return payload


def test_categories_are_exposed() -> None:
    response = client.get("/complaints/categories")

    assert response.status_code == 200

    categories = response.json()["categories"]
    values = [item["value"] for item in categories]

    assert "permis" in values
    assert "autre" in values
    assert all(item["label"] for item in categories)


def test_complaint_is_created_with_reference() -> None:
    response = client.post(
        "/complaints",
        json=build_payload(),
    )

    assert response.status_code == 201

    data = response.json()

    assert data["reference"].startswith("NARSA-")
    assert data["status"] == "recue"
    assert data["reference"] in data["message"]


def test_tracking_requires_matching_email() -> None:
    created = client.post(
        "/complaints",
        json=build_payload(email="suivi@example.ma"),
    ).json()

    reference = created["reference"]

    ok = client.post(
        "/complaints/track",
        json={
            "reference": reference,
            "email": "suivi@example.ma",
        },
    )

    assert ok.status_code == 200

    data = ok.json()

    assert data["status_label"] == "Reçue"
    assert data["category_label"] == "Permis de conduire"
    assert len(data["events"]) == 1

    wrong = client.post(
        "/complaints/track",
        json={
            "reference": reference,
            "email": "autre@example.ma",
        },
    )

    assert wrong.status_code == 404


def test_reference_lookup_is_case_insensitive() -> None:
    created = client.post(
        "/complaints",
        json=build_payload(email="casse@example.ma"),
    ).json()

    response = client.post(
        "/complaints/track",
        json={
            "reference": created["reference"].lower(),
            "email": "CASSE@example.ma",
        },
    )

    assert response.status_code == 200
    assert response.json()["reference"] == (
        created["reference"]
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"category": "categorie_inexistante"},
        {"email": "adresse-invalide"},
        {"message": "court"},
        {"full_name": "A"},
    ],
)
def test_invalid_payloads_are_rejected(
    overrides: dict[str, Any],
) -> None:
    response = client.post(
        "/complaints",
        json=build_payload(**overrides),
    )

    assert response.status_code == 422


def test_analysis_suggests_category_and_priority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.app.routes import complaints as complaints_route

    monkeypatch.setattr(
        complaints_route,
        "predict_intent",
        lambda text, top_k: {
            "domain": "permis",
            "intent": "permis_duplicata",
            "confidence": 0.93,
            "needs_clarification": False,
            "alternatives": [],
        },
    )
    monkeypatch.setattr(
        complaints_route,
        "generate_answer",
        lambda question, top_k: {
            "status": "context_found",
            "answer": "Le duplicata se demande en agence.",
            "domain": "permis",
            "intent": "permis_duplicata",
            "confidence": 0.93,
            "sources": [
                {"title": "Duplicata du permis"},
            ],
        },
    )

    response = client.post(
        "/complaints/analyze",
        json={
            "message": (
                "Mon duplicata de permis est sans reponse "
                "depuis deux mois malgre plusieurs relances."
            ),
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["category"] == "permis"
    assert data["category_is_reliable"] is True
    assert data["priority"] == "haute"
    assert data["priority_reasons"]
    assert data["instant_answer"]["answer"] == (
        "Le duplicata se demande en agence."
    )
    assert data["instant_answer"]["sources"] == [
        "Duplicata du permis"
    ]


def test_analysis_hides_answer_without_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.app.routes import complaints as complaints_route

    monkeypatch.setattr(
        complaints_route,
        "predict_intent",
        lambda text, top_k: {
            "domain": "vehicule",
            "intent": "carte_grise_demande",
            "confidence": 0.35,
            "needs_clarification": True,
            "alternatives": [],
        },
    )
    monkeypatch.setattr(
        complaints_route,
        "generate_answer",
        lambda question, top_k: {
            "status": "no_context_found",
            "answer": "Aucun document pertinent.",
            "domain": "vehicule",
            "intent": "carte_grise_demande",
            "confidence": 0.35,
            "sources": [],
        },
    )

    response = client.post(
        "/complaints/analyze",
        json={
            "message": (
                "Je souhaite signaler un probleme concernant "
                "mon dossier de vehicule."
            ),
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["instant_answer"] is None
    assert data["category"] == "autre"
    assert data["category_is_reliable"] is False


def test_priority_is_stored_at_submission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.app.routes import complaints as complaints_route

    monkeypatch.setattr(
        complaints_route,
        "predict_intent",
        lambda text, top_k: {
            "domain": "accident",
            "intent": "declaration_accident",
            "confidence": 0.88,
            "needs_clarification": False,
            "alternatives": [],
        },
    )

    created = client.post(
        "/complaints",
        json=build_payload(
            email="urgence@example.ma",
            category="accident",
            subject="Accident avec blesse",
            message=(
                "Un accident avec un blesse a eu lieu et "
                "aucun secours n'est intervenu rapidement."
            ),
        ),
    )

    assert created.status_code == 201

    tracked = client.post(
        "/complaints/track",
        json={
            "reference": created.json()["reference"],
            "email": "urgence@example.ma",
        },
    ).json()

    assert tracked["priority"] == "urgente"
    assert tracked["priority_label"] == "Urgente"
    assert tracked["detected_intent"] == (
        "declaration_accident"
    )
    assert "priorité" in tracked["events"][0]["note"].lower()
