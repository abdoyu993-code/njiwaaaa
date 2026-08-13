from __future__ import annotations

import smtplib
from typing import Any

import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import settings
from backend.app.main import app
from backend.app.notifications import email as email_module


client = TestClient(app)

CITIZEN_EMAIL = "citoyen.suivi@example.ma"


@pytest.fixture(autouse=True)
def clean_email_trace() -> None:
    email_module.clear_sent_messages()


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


def create_complaint(
    email: str = CITIZEN_EMAIL,
) -> str:
    response = client.post(
        "/complaints",
        json={
            "full_name": "Hicham Berrada",
            "email": email,
            "category": "visite_technique",
            "subject": "Refus injustifié au contrôle",
            "message": (
                "Mon véhicule a été refusé au contrôle "
                "technique pour un motif non conforme aux "
                "critères affichés."
            ),
        },
    )

    assert response.status_code == 201

    return response.json()["reference"]


def track(reference: str, email: str = CITIZEN_EMAIL) -> dict[str, Any]:
    response = client.post(
        "/complaints/track",
        json={"reference": reference, "email": email},
    )

    assert response.status_code == 200

    return response.json()


# ------------------------- Flux complet -------------------------------


def test_full_flow_from_submission_to_citizen_view() -> None:
    """
    Dépôt → ouverture par l'agent → réponse → visible dans le
    suivi citoyen, avec notification email à chaque étape.
    """
    reference = create_complaint()

    creation_mails = email_module.sent_messages()

    assert len(creation_mails) == 1
    assert creation_mails[0]["to"] == CITIZEN_EMAIL
    assert reference in creation_mails[0]["subject"]

    headers = admin_headers()

    # L'agent ouvre le dossier.
    detail = client.get(
        f"/admin/complaints/{reference}",
        headers=headers,
    )

    assert detail.status_code == 200
    assert detail.json()["official_response"] is None

    # L'agent rédige et valide sa réponse.
    answer = (
        "Après vérification, le refus était justifié par "
        "un défaut de freinage. Vous pouvez présenter le "
        "véhicule à une contre-visite dans un centre agréé."
    )

    updated = client.post(
        f"/admin/complaints/{reference}/response",
        headers=headers,
        json={"response": answer, "status": "traitee"},
    )

    assert updated.status_code == 200

    payload = updated.json()

    assert payload["official_response"] == answer
    assert payload["status"] == "traitee"
    assert payload["responded_by"] == "agent"
    assert payload["responded_at"] is not None

    # La réponse est visible côté citoyen.
    citizen_view = track(reference)

    assert citizen_view["official_response"] == answer
    assert citizen_view["status_label"] == "Traitée"

    # Un email a prévenu le citoyen qu'une réponse existe.
    response_mails = [
        message
        for message in email_module.sent_messages()
        if "réponse" in message["subject"].lower()
    ]

    assert response_mails
    assert reference in response_mails[-1]["subject"]


def test_history_records_every_step() -> None:
    reference = create_complaint()
    headers = admin_headers()

    client.patch(
        f"/admin/complaints/{reference}",
        headers=headers,
        json={
            "status": "en_cours",
            "note": "Dossier transmis au centre concerné.",
        },
    )
    client.post(
        f"/admin/complaints/{reference}/response",
        headers=headers,
        json={
            "response": (
                "Le centre a confirmé le motif du refus, "
                "une contre-visite est possible."
            )
        },
    )

    events = track(reference)["events"]

    assert [event["event_type"] for event in events] == [
        "creation",
        "status",
        "response",
    ]
    assert [event["status"] for event in events] == [
        "recue",
        "en_cours",
        "traitee",
    ]
    assert all(event["created_at"] for event in events)
    assert events[-1]["author"] == "agent"


def test_status_change_notifies_the_citizen() -> None:
    reference = create_complaint()
    headers = admin_headers()

    email_module.clear_sent_messages()

    client.patch(
        f"/admin/complaints/{reference}",
        headers=headers,
        json={"status": "en_cours"},
    )

    messages = email_module.sent_messages()

    assert len(messages) == 1
    assert "en cours de traitement" in messages[0]["body"]


def test_response_requires_authentication() -> None:
    reference = create_complaint()

    response = client.post(
        f"/admin/complaints/{reference}/response",
        json={"response": "Réponse sans authentification."},
    )

    assert response.status_code == 401


def test_empty_response_is_rejected() -> None:
    reference = create_complaint()
    headers = admin_headers()

    response = client.post(
        f"/admin/complaints/{reference}/response",
        headers=headers,
        json={"response": "  "},
    )

    assert response.status_code == 422
    assert track(reference)["official_response"] is None


def test_unknown_reference_returns_404() -> None:
    headers = admin_headers()

    response = client.post(
        "/admin/complaints/NARSA-2026-INCONNU/response",
        headers=headers,
        json={"response": "Réponse pour un dossier absent."},
    )

    assert response.status_code == 404


# ---------------------- Résilience des emails -------------------------


def test_email_failure_does_not_lose_the_complaint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Un serveur SMTP en panne ne fait perdre aucune donnée."""
    monkeypatch.setattr(
        settings,
        "smtp_host",
        "smtp.invalide.test",
    )

    def failing_smtp(*args: Any, **kwargs: Any) -> None:
        raise OSError("serveur SMTP injoignable")

    monkeypatch.setattr(smtplib, "SMTP", failing_smtp)

    reference = create_complaint(
        email="panne.smtp@example.ma"
    )

    messages = email_module.sent_messages()

    assert messages[-1]["status"] == "failed"

    # La réclamation est bien enregistrée malgré l'échec.
    tracked = track(
        reference,
        email="panne.smtp@example.ma",
    )

    assert tracked["reference"] == reference
    assert tracked["status"] == "recue"


def test_email_failure_does_not_lose_the_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = create_complaint(
        email="panne.reponse@example.ma"
    )
    headers = admin_headers()

    monkeypatch.setattr(
        settings,
        "smtp_host",
        "smtp.invalide.test",
    )

    def failing_smtp(*args: Any, **kwargs: Any) -> None:
        raise OSError("serveur SMTP injoignable")

    monkeypatch.setattr(smtplib, "SMTP", failing_smtp)

    answer = (
        "Votre dossier a été réexaminé et la décision est "
        "confirmée."
    )

    updated = client.post(
        f"/admin/complaints/{reference}/response",
        headers=headers,
        json={"response": answer},
    )

    assert updated.status_code == 200
    assert email_module.sent_messages()[-1]["status"] == (
        "failed"
    )

    tracked = track(
        reference,
        email="panne.reponse@example.ma",
    )

    assert tracked["official_response"] == answer


def test_emails_can_be_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "email_enabled", False)

    result = email_module.notify_complaint_created(
        to="test@example.ma",
        reference="NARSA-2026-TEST",
        full_name="Test",
    )

    assert result["status"] == "disabled"


def test_console_mode_keeps_the_full_message() -> None:
    result = email_module.notify_response_added(
        to="test@example.ma",
        reference="NARSA-2026-TEST",
        full_name="Test",
    )

    assert result["status"] == "console"
    assert "portail de suivi" in result["body"]
    assert "NARSA-2026-TEST" in result["body"]
