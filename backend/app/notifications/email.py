from __future__ import annotations

import smtplib
import ssl
from email.message import EmailMessage
from typing import Any

from backend.app.core.config import settings


"""
Notifications par email.

Le mail sert uniquement à prévenir le citoyen : la réponse
officielle complète reste consultable sur le portail de suivi.

Aucune erreur d'envoi ne remonte à l'appelant : une réclamation
enregistrée ne doit jamais être perdue parce que le serveur SMTP
est indisponible.
"""


# Trace en mémoire des envois, utile aux tests et au diagnostic.
_sent_messages: list[dict[str, Any]] = []
MAX_TRACE = 50


def sent_messages() -> list[dict[str, Any]]:
    return list(_sent_messages)


def clear_sent_messages() -> None:
    _sent_messages.clear()


def _trace(entry: dict[str, Any]) -> None:
    _sent_messages.append(entry)

    while len(_sent_messages) > MAX_TRACE:
        _sent_messages.pop(0)


def send_email(
    to: str,
    subject: str,
    body: str,
) -> dict[str, Any]:
    """
    Envoie un email et retourne le résultat de la tentative.

    Retourne toujours un dictionnaire ; ne lève jamais.
    Statuts possibles : `sent`, `console`, `disabled`, `failed`.
    """
    if not settings.email_enabled:
        result = {
            "status": "disabled",
            "to": to,
            "subject": subject,
        }
        _trace(result)

        return result

    if not settings.smtp_host:
        # Sans serveur SMTP configuré, le message est affiché
        # dans les journaux : le développement reste possible
        # sans compte email.
        print("=" * 60)
        print(f"[EMAIL SIMULÉ] À : {to}")
        print(f"[EMAIL SIMULÉ] Objet : {subject}")
        print(body)
        print("=" * 60)

        result = {
            "status": "console",
            "to": to,
            "subject": subject,
            "body": body,
        }
        _trace(result)

        return result

    message = EmailMessage()
    message["From"] = settings.email_sender
    message["To"] = to
    message["Subject"] = subject
    message.set_content(body)

    try:
        if settings.smtp_use_ssl:
            context = ssl.create_default_context()

            with smtplib.SMTP_SSL(
                settings.smtp_host,
                settings.smtp_port,
                context=context,
                timeout=settings.smtp_timeout,
            ) as server:
                if settings.smtp_username:
                    server.login(
                        settings.smtp_username,
                        settings.smtp_password,
                    )

                server.send_message(message)

        else:
            with smtplib.SMTP(
                settings.smtp_host,
                settings.smtp_port,
                timeout=settings.smtp_timeout,
            ) as server:
                if settings.smtp_use_tls:
                    server.starttls(
                        context=ssl.create_default_context()
                    )

                if settings.smtp_username:
                    server.login(
                        settings.smtp_username,
                        settings.smtp_password,
                    )

                server.send_message(message)

        result = {
            "status": "sent",
            "to": to,
            "subject": subject,
        }

    except Exception as error:  # pragma: no cover - dépend du réseau
        print(
            "Envoi d'email impossible :",
            f"{type(error).__name__}: {error}",
        )

        result = {
            "status": "failed",
            "to": to,
            "subject": subject,
            "error": f"{type(error).__name__}: {error}",
        }

    _trace(result)

    return result


# ----------------------------------------------------------------------
# Messages métier
# ----------------------------------------------------------------------


def _portal_line() -> str:
    return (
        "Consultez le portail de suivi pour voir le détail :\n"
        f"{settings.portal_url}/#/suivi"
    )


def notify_complaint_created(
    to: str,
    reference: str,
    full_name: str,
) -> dict[str, Any]:
    return send_email(
        to=to,
        subject=(
            f"Votre réclamation {reference} a bien été "
            "enregistrée"
        ),
        body=(
            f"Bonjour {full_name},\n\n"
            f"Votre réclamation {reference} a bien été "
            "enregistrée par la NARSA.\n\n"
            "Conservez cette référence : elle vous permet de "
            "suivre l'avancement de votre demande.\n\n"
            f"{_portal_line()}\n\n"
            "Ceci est un message automatique, merci de ne pas "
            "y répondre."
        ),
    )


def notify_response_added(
    to: str,
    reference: str,
    full_name: str,
) -> dict[str, Any]:
    return send_email(
        to=to,
        subject=(
            f"Une réponse a été ajoutée à votre réclamation "
            f"{reference}"
        ),
        body=(
            f"Bonjour {full_name},\n\n"
            f"Une réponse a été ajoutée à votre réclamation "
            f"{reference}.\n\n"
            "La réponse officielle complète est consultable "
            "sur le portail de suivi.\n\n"
            f"{_portal_line()}\n\n"
            "Ceci est un message automatique, merci de ne pas "
            "y répondre."
        ),
    )


STATUS_MESSAGES = {
    "en_cours": (
        "Votre réclamation est maintenant en cours de "
        "traitement."
    ),
    "traitee": "Votre réclamation a été traitée.",
    "rejetee": (
        "Votre réclamation a été examinée et n'a pas pu être "
        "retenue. Le motif est indiqué sur le portail."
    ),
}


def notify_status_changed(
    to: str,
    reference: str,
    full_name: str,
    status: str,
) -> dict[str, Any] | None:
    """Notifie les changements de statut significatifs."""
    sentence = STATUS_MESSAGES.get(status)

    if sentence is None:
        return None

    return send_email(
        to=to,
        subject=f"Réclamation {reference} — mise à jour",
        body=(
            f"Bonjour {full_name},\n\n"
            f"{sentence}\n\n"
            f"Référence : {reference}\n\n"
            f"{_portal_line()}\n\n"
            "Ceci est un message automatique, merci de ne pas "
            "y répondre."
        ),
    )
