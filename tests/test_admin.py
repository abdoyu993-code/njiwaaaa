from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from backend.app.core import security
from backend.app.main import app


client = TestClient(app)


def admin_headers() -> dict[str, str]:
    response = client.post(
        "/admin/login",
        json={
            "username": "agent",
            "password": "narsa2026",
        },
    )

    assert response.status_code == 200

    return {
        "Authorization": (
            f"Bearer {response.json()['access_token']}"
        )
    }


def create_complaint() -> str:
    response = client.post(
        "/complaints",
        json={
            "full_name": "Salma Bennani",
            "email": "salma@example.ma",
            "category": "infractions",
            "subject": "Contravention contestée",
            "message": (
                "Je conteste une contravention reçue "
                "alors que mon véhicule était vendu."
            ),
        },
    )

    assert response.status_code == 201

    return response.json()["reference"]


def test_login_rejects_wrong_password() -> None:
    response = client.post(
        "/admin/login",
        json={
            "username": "agent",
            "password": "mauvais-mot-de-passe",
        },
    )

    assert response.status_code == 401


@pytest.mark.parametrize(
    "path",
    [
        "/admin/dashboard",
        "/admin/complaints",
        "/admin/conversations",
    ],
)
def test_admin_routes_require_a_token(path: str) -> None:
    assert client.get(path).status_code == 401
    assert (
        client.get(
            path,
            headers={"Authorization": "Bearer faux-jeton"},
        ).status_code
        == 401
    )


def test_expired_token_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token, expires_at = security.create_admin_token("agent")

    class FrozenClock:
        @staticmethod
        def time() -> float:
            return float(expires_at + 60)

    monkeypatch.setattr(
        security,
        "time",
        FrozenClock,
    )

    with pytest.raises(ValueError):
        security.decode_admin_token(token)


def test_tampered_token_is_refused() -> None:
    token, _ = security.create_admin_token("agent")
    payload, _, _signature = token.partition(".")

    with pytest.raises(ValueError):
        security.decode_admin_token(
            f"{payload}.signature-falsifiee"
        )


def test_dashboard_returns_expected_sections() -> None:
    headers = admin_headers()
    response = client.get(
        "/admin/dashboard?days=7",
        headers=headers,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["period_days"] == 7
    assert len(data["timeseries"]) == 7
    assert set(data["kpis"]) == {
        "conversations",
        "interactions",
        "avg_confidence",
        "avg_time_ms",
        "clarifications",
        "low_confidence",
    }
    assert {
        item["label"] for item in data["answer_types"]
    } == {
        "Réponses générées (Gemini)",
        "Réponses extractives (RAG)",
        "Clarifications",
        "Sans contexte",
    }
    assert "by_status" in data["complaints"]


def test_agent_can_change_complaint_status() -> None:
    headers = admin_headers()
    reference = create_complaint()

    listing = client.get(
        "/admin/complaints",
        headers=headers,
    )

    assert listing.status_code == 200
    assert listing.json()["total"] >= 1

    updated = client.patch(
        f"/admin/complaints/{reference}",
        headers=headers,
        json={
            "status": "en_cours",
            "note": "Dossier transmis au service concerné.",
        },
    )

    assert updated.status_code == 200

    data = updated.json()

    assert data["status"] == "en_cours"
    assert data["admin_note"] == (
        "Dossier transmis au service concerné."
    )
    assert [event["status"] for event in data["events"]] == [
        "recue",
        "en_cours",
    ]

    citizen_view = client.post(
        "/complaints/track",
        json={
            "reference": reference,
            "email": "salma@example.ma",
        },
    ).json()

    assert citizen_view["status_label"] == (
        "En cours de traitement"
    )


def test_invalid_status_is_rejected() -> None:
    headers = admin_headers()
    reference = create_complaint()

    response = client.patch(
        f"/admin/complaints/{reference}",
        headers=headers,
        json={"status": "statut_inconnu"},
    )

    assert response.status_code == 422


def test_search_filter_matches_reference() -> None:
    headers = admin_headers()
    reference = create_complaint()

    response = client.get(
        f"/admin/complaints?search={reference}",
        headers=headers,
    )

    assert response.status_code == 200

    references = [
        item["reference"]
        for item in response.json()["items"]
    ]

    assert references == [reference]


def test_token_expiry_is_in_the_future() -> None:
    response = client.post(
        "/admin/login",
        json={
            "username": "agent",
            "password": "narsa2026",
        },
    )

    assert response.json()["expires_at"] > int(time.time())


def test_insights_expose_alerts_and_blind_spots() -> None:
    headers = admin_headers()

    response = client.get(
        "/admin/insights?days=7",
        headers=headers,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["period_days"] == 7
    assert isinstance(data["blind_spots"], list)
    assert data["alerts"]
    assert all(
        alert["level"]
        in {"critique", "attention", "ok"}
        for alert in data["alerts"]
    )


def test_insights_require_authentication() -> None:
    assert client.get("/admin/insights").status_code == 401


def test_complaints_export_returns_csv() -> None:
    headers = admin_headers()
    create_complaint()

    response = client.get(
        "/admin/complaints/export",
        headers=headers,
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith(
        "text/csv"
    )
    assert "reclamations-narsa.csv" in (
        response.headers["content-disposition"]
    )

    lines = response.text.strip().splitlines()

    assert lines[0].endswith("Priorité")
    assert len(lines) >= 2


def test_conversations_export_returns_csv() -> None:
    headers = admin_headers()

    response = client.get(
        "/admin/conversations/export?limit=10",
        headers=headers,
    )

    assert response.status_code == 200
    assert "Question" in response.text.splitlines()[0]


def test_priority_filter_narrows_the_list() -> None:
    headers = admin_headers()
    create_complaint()

    response = client.get(
        "/admin/complaints?priority=normale",
        headers=headers,
    )

    assert response.status_code == 200
    assert all(
        item["priority"] == "normale"
        for item in response.json()["items"]
    )
