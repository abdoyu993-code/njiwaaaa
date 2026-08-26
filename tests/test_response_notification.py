from __future__ import annotations

import smtplib
import ssl
from typing import Any

import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import settings
from backend.app.main import app
from backend.app.notifications import email as email_module


"""
Scénario exact demandé pour la mise en production de la
réponse agent :

dépôt citoyen → référence → réponse de l'agent → persistance
→ visibilité dans le suivi → notification email réellement
remise au serveur SMTP.

Ces tests complètent `test_complaint_workflow.py` : ils
vérifient la mécanique SMTP elle-même (STARTTLS, SSL,
authentification, expéditeur, destinataire) sans jamais
ouvrir de connexion réseau.
"""


client = TestClient(app)

TEST_EMAIL = "test.reponse.agent@example.ma"

OFFICIAL_ANSWER = (
    "Bonjour,\n"
    "\n"
    "Nous avons bien étudié votre réclamation.\n"
    "Votre demande a été prise en charge.\n"
    "\n"
    "Cordialement,\n"
    "NARSA"
)


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


def create_test_complaint(
    email: str = TEST_EMAIL,
) -> str:
    """Test 1 — création avec le jeu de données demandé."""
    response = client.post(
        "/complaints",
        json={
            "full_name": "Test Email",
            "email": email,
            "category": "autre",
            "subject": "Test réponse agent",
            "message": "Ceci est une réclamation de test.",
        },
    )

    assert response.status_code == 201

    return response.json()["reference"]


# ----------------------------------------------------------------------
# Faux serveur SMTP : capture ce qui serait réellement envoyé
# ----------------------------------------------------------------------


class FakeSMTP:
    """
    Remplace `smtplib.SMTP` / `smtplib.SMTP_SSL`.

    Enregistre la séquence d'appels pour vérifier que STARTTLS,
    l'authentification et l'envoi ont bien lieu dans cet ordre.
    """

    instances: list["FakeSMTP"] = []

    def __init__(
        self,
        host: str,
        port: int,
        *,
        context: ssl.SSLContext | None = None,
        timeout: float | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.ssl_direct = context is not None
        self.calls: list[str] = []
        self.login_arguments: tuple[str, str] | None = None
        self.message: Any = None

        FakeSMTP.instances.append(self)

    def __enter__(self) -> "FakeSMTP":
        return self

    def __exit__(self, *_: object) -> None:
        self.calls.append("quit")

    def ehlo(self) -> None:
        self.calls.append("ehlo")

    def starttls(
        self,
        context: ssl.SSLContext | None = None,
    ) -> None:
        assert context is not None, (
            "STARTTLS doit être négocié avec un contexte SSL."
        )
        self.calls.append("starttls")

    def login(self, username: str, password: str) -> None:
        self.calls.append("login")
        self.login_arguments = (username, password)

    def send_message(self, message: Any) -> dict:
        self.calls.append("send_message")
        self.message = message

        # Aucun destinataire refusé.
        return {}


@pytest.fixture
def smtp_server(
    monkeypatch: pytest.MonkeyPatch,
) -> type[FakeSMTP]:
    """Configure un SMTP « réel » simulé en STARTTLS."""
    FakeSMTP.instances = []

    monkeypatch.setattr(settings, "email_enabled", True)
    monkeypatch.setattr(settings, "smtp_host", "smtp.test.local")
    monkeypatch.setattr(settings, "smtp_port", 587)
    monkeypatch.setattr(settings, "smtp_use_tls", True)
    monkeypatch.setattr(settings, "smtp_use_ssl", False)
    monkeypatch.setattr(settings, "smtp_username", "agent@test.local")
    monkeypatch.setattr(settings, "smtp_password", "mot-de-passe-test")
    monkeypatch.setattr(
        settings,
        "email_sender",
        "no-reply@test.local",
    )
    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    monkeypatch.setattr(smtplib, "SMTP_SSL", FakeSMTP)

    return FakeSMTP


# ----------------------------------------------------------------------
# Test 1 — création
# ----------------------------------------------------------------------


def test_creation_produces_a_unique_reference() -> None:
    first = create_test_complaint()
    second = create_test_complaint()

    assert first.startswith("NARSA-")
    assert first != second


# ----------------------------------------------------------------------
# Test 2 — réponse de l'agent
# ----------------------------------------------------------------------


def test_agent_response_is_persisted_exactly() -> None:
    reference = create_test_complaint()
    headers = admin_headers()

    saved = client.post(
        f"/admin/complaints/{reference}/response",
        headers=headers,
        json={
            "response": OFFICIAL_ANSWER,
            "status": "traitee",
        },
    )

    assert saved.status_code == 200

    payload = saved.json()

    # La réponse est stockée telle quelle : les sauts de ligne
    # de l'agent ne sont pas écrasés.
    assert payload["official_response"] == OFFICIAL_ANSWER
    assert payload["responded_at"] is not None
    assert payload["responded_by"] == "agent"
    assert payload["status"] == "traitee"

    response_events = [
        event
        for event in payload["events"]
        if event["event_type"] == "response"
    ]

    assert len(response_events) == 1
    assert response_events[0]["author"] == "agent"


def test_agent_can_correct_an_existing_response() -> None:
    """L'agent doit pouvoir corriger sa réponse sans doublon."""
    reference = create_test_complaint()
    headers = admin_headers()

    client.post(
        f"/admin/complaints/{reference}/response",
        headers=headers,
        json={"response": OFFICIAL_ANSWER},
    )

    corrected = "Bonjour,\n\nAprès réexamen, voici la décision définitive."

    updated = client.post(
        f"/admin/complaints/{reference}/response",
        headers=headers,
        json={"response": corrected},
    )

    assert updated.status_code == 200
    assert updated.json()["official_response"] == corrected

    detail = client.get(
        f"/admin/complaints/{reference}",
        headers=headers,
    )

    assert detail.json()["official_response"] == corrected


# ----------------------------------------------------------------------
# Test 3 — suivi citoyen
# ----------------------------------------------------------------------


def test_tracking_returns_the_complete_response() -> None:
    reference = create_test_complaint()
    headers = admin_headers()

    client.post(
        f"/admin/complaints/{reference}/response",
        headers=headers,
        json={"response": OFFICIAL_ANSWER},
    )

    tracked = client.post(
        "/complaints/track",
        json={
            "reference": reference,
            "email": TEST_EMAIL,
        },
    )

    assert tracked.status_code == 200

    payload = tracked.json()

    assert payload["official_response"] == OFFICIAL_ANSWER
    assert payload["reference"] == reference
    assert payload["status_label"] == "Traitée"
    assert payload["responded_at"] is not None

    # Le citoyen ne voit jamais l'état de la notification
    # interne, réservé à l'espace agent.
    assert payload["notification"] is None


def test_tracking_requires_the_email_used_at_submission() -> None:
    reference = create_test_complaint()

    wrong = client.post(
        "/complaints/track",
        json={
            "reference": reference,
            "email": "quelqu.un.dautre@example.ma",
        },
    )

    assert wrong.status_code == 404


# ----------------------------------------------------------------------
# Test 4 — email réellement remis au serveur SMTP
# ----------------------------------------------------------------------


def test_response_email_reaches_the_smtp_server(
    smtp_server: type[FakeSMTP],
) -> None:
    """
    Vérifie la mécanique SMTP complète : connexion, STARTTLS,
    authentification, expéditeur, destinataire et contenu.
    """
    reference = create_test_complaint()
    headers = admin_headers()

    email_module.clear_sent_messages()
    smtp_server.instances.clear()

    saved = client.post(
        f"/admin/complaints/{reference}/response",
        headers=headers,
        json={"response": OFFICIAL_ANSWER},
    )

    assert saved.status_code == 200

    # Le statut de l'envoi remonte à l'agent.
    notification = saved.json()["notification"]

    assert notification["status"] == "sent"
    assert notification["to"] == TEST_EMAIL
    assert notification["error"] is None

    assert len(smtp_server.instances) == 1

    server = smtp_server.instances[0]

    assert server.host == "smtp.test.local"
    assert server.port == 587
    assert server.calls == [
        "ehlo",
        "starttls",
        "ehlo",
        "login",
        "send_message",
        "quit",
    ]
    assert server.login_arguments == (
        "agent@test.local",
        "mot-de-passe-test",
    )

    message = server.message

    assert message["To"] == TEST_EMAIL
    assert "no-reply@test.local" in message["From"]
    assert reference in message["Subject"]
    assert "Une réponse a été ajoutée" in message["Subject"]

    body = message.get_content()

    assert "Test Email" in body
    assert reference in body
    assert f"{settings.portal_url}/#/suivi" in body

    # L'email n'est qu'une notification : la réponse complète
    # reste sur le portail.
    assert OFFICIAL_ANSWER not in body


def test_ssl_mode_uses_a_direct_tls_connection(
    smtp_server: type[FakeSMTP],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "smtp_use_ssl", True)
    monkeypatch.setattr(settings, "smtp_use_tls", False)
    monkeypatch.setattr(settings, "smtp_port", 465)

    smtp_server.instances.clear()

    result = email_module.notify_response_added(
        to=TEST_EMAIL,
        reference="NARSA-2026-SSL",
        full_name="Test Email",
    )

    assert result["status"] == "sent"

    server = smtp_server.instances[0]

    assert server.ssl_direct is True
    assert server.port == 465

    # En SSL direct, aucun STARTTLS ne doit être négocié.
    assert "starttls" not in server.calls
    assert server.calls == ["login", "send_message", "quit"]


def test_anonymous_smtp_does_not_authenticate(
    smtp_server: type[FakeSMTP],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Un relais interne sans compte ne doit pas appeler login."""
    monkeypatch.setattr(settings, "smtp_username", "")

    smtp_server.instances.clear()

    email_module.notify_response_added(
        to=TEST_EMAIL,
        reference="NARSA-2026-ANON",
        full_name="Test Email",
    )

    assert "login" not in smtp_server.instances[0].calls


def test_authentication_failure_is_diagnosed(
    smtp_server: type[FakeSMTP],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Un refus d'authentification doit produire un diagnostic
    exploitable, sans faire perdre la réponse enregistrée.
    """
    reference = create_test_complaint(
        email="auth.refusee@example.ma"
    )
    headers = admin_headers()

    def refuse(self: FakeSMTP, username: str, password: str) -> None:
        raise smtplib.SMTPAuthenticationError(
            535,
            b"5.7.8 Username and Password not accepted",
        )

    monkeypatch.setattr(FakeSMTP, "login", refuse)

    saved = client.post(
        f"/admin/complaints/{reference}/response",
        headers=headers,
        json={"response": OFFICIAL_ANSWER},
    )

    # La réponse reste enregistrée malgré l'échec SMTP.
    assert saved.status_code == 200
    assert saved.json()["official_response"] == OFFICIAL_ANSWER

    notification = saved.json()["notification"]

    assert notification["status"] == "failed"
    assert "SMTPAuthenticationError" in notification["error"]
    assert "mot de passe d'application" in notification["detail"]

    tracked = client.post(
        "/complaints/track",
        json={
            "reference": reference,
            "email": "auth.refusee@example.ma",
        },
    )

    assert tracked.json()["official_response"] == OFFICIAL_ANSWER


def test_refused_recipient_is_not_reported_as_sent(
    smtp_server: type[FakeSMTP],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Un destinataire refusé ne doit jamais compter comme envoyé."""

    def refuse_recipient(self: FakeSMTP, message: Any) -> dict:
        self.calls.append("send_message")

        return {TEST_EMAIL: (550, b"No such user")}

    monkeypatch.setattr(
        FakeSMTP,
        "send_message",
        refuse_recipient,
    )

    result = email_module.notify_response_added(
        to=TEST_EMAIL,
        reference="NARSA-2026-REFUS",
        full_name="Test Email",
    )

    assert result["status"] == "failed"
    assert "refusé le destinataire" in result["detail"]


# ----------------------------------------------------------------------
# Diagnostic SMTP côté agent
# ----------------------------------------------------------------------


def test_email_status_never_exposes_the_password(
    smtp_server: type[FakeSMTP],
) -> None:
    headers = admin_headers()

    response = client.get("/admin/email/status", headers=headers)

    assert response.status_code == 200

    configuration = response.json()["configuration"]

    assert configuration["mode"] == "smtp"
    assert configuration["smtp_host"] == "smtp.test.local"
    assert configuration["smtp_password_configured"] is True

    # Ni le mot de passe ni l'identifiant complet ne sortent.
    serialized = response.text

    assert "mot-de-passe-test" not in serialized
    assert "agent@test.local" not in serialized
    assert configuration["smtp_username"] == "ag***@test.local"


def test_email_status_requires_authentication() -> None:
    assert client.get("/admin/email/status").status_code == 401
    assert (
        client.post(
            "/admin/email/test",
            json={"to": TEST_EMAIL},
        ).status_code
        == 401
    )


def test_email_test_endpoint_sends_a_single_message(
    smtp_server: type[FakeSMTP],
) -> None:
    """Permet de valider SMTP sans polluer un dossier réel."""
    headers = admin_headers()

    smtp_server.instances.clear()

    response = client.post(
        "/admin/email/test",
        headers=headers,
        json={"to": TEST_EMAIL},
    )

    assert response.status_code == 200
    assert response.json()["notification"]["status"] == "sent"

    assert len(smtp_server.instances) == 1
    assert (
        smtp_server.instances[0].message["To"] == TEST_EMAIL
    )


def test_console_mode_is_reported_as_not_sent() -> None:
    """
    Sans SMTP_HOST, l'agent doit voir que rien n'est parti :
    ne jamais laisser croire qu'un email a été envoyé.
    """
    reference = create_test_complaint(
        email="mode.console@example.ma"
    )
    headers = admin_headers()

    saved = client.post(
        f"/admin/complaints/{reference}/response",
        headers=headers,
        json={"response": OFFICIAL_ANSWER},
    )

    notification = saved.json()["notification"]

    assert notification["status"] == "console"
    assert notification["status"] != "sent"
    assert "SMTP_HOST" in notification["detail"]


def test_masking_keeps_secrets_out_of_logs() -> None:
    assert (
        email_module.mask_identifier("agent.narsa@gmail.com")
        == "ag***@gmail.com"
    )
    assert email_module.mask_identifier("") == ""
