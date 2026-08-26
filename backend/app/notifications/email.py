from __future__ import annotations

import logging
import smtplib
import ssl
from email.message import EmailMessage
from email.utils import formataddr, make_msgid
from typing import Any

from backend.app.core.config import settings


"""
Notifications par email.

Le mail sert uniquement à prévenir le citoyen : la réponse
officielle complète reste consultable sur le portail de suivi.

Aucune erreur d'envoi ne remonte à l'appelant : une réclamation
enregistrée ne doit jamais être perdue parce que le serveur SMTP
est indisponible. En contrepartie, chaque tentative est tracée
(journal applicatif + trace mémoire) pour permettre le
diagnostic d'un vrai problème SMTP.
"""


logger = logging.getLogger("narsa.email")


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


# ----------------------------------------------------------------------
# Diagnostic de configuration
# ----------------------------------------------------------------------


def mask_identifier(value: str) -> str:
    """
    Masque un identifiant pour les journaux.

    `agent.narsa@gmail.com` devient `ag***@gmail.com` :
    assez pour vérifier le compte utilisé, jamais assez
    pour le réutiliser.
    """
    if not value:
        return ""

    local, separator, domain = value.partition("@")
    visible = local[:2]

    return f"{visible}***{separator}{domain}"


def smtp_configuration() -> dict[str, Any]:
    """
    Décrit la configuration SMTP courante.

    Ne contient JAMAIS le mot de passe : uniquement le fait
    qu'un mot de passe est renseigné ou non.
    """
    if not settings.email_enabled:
        mode = "disabled"
    elif not settings.smtp_host:
        mode = "console"
    else:
        mode = "smtp"

    return {
        "mode": mode,
        "email_enabled": settings.email_enabled,
        "smtp_host": settings.smtp_host,
        "smtp_port": settings.smtp_port,
        "smtp_use_tls": settings.smtp_use_tls,
        "smtp_use_ssl": settings.smtp_use_ssl,
        "smtp_timeout": settings.smtp_timeout,
        "smtp_username": mask_identifier(
            settings.smtp_username
        ),
        "smtp_password_configured": bool(
            settings.smtp_password
        ),
        "email_sender": settings.email_sender,
        "portal_url": settings.portal_url,
    }


def _diagnose(error: Exception) -> str:
    """Traduit une erreur SMTP en piste de correction concrète."""
    if isinstance(error, smtplib.SMTPAuthenticationError):
        return (
            "Le serveur a refusé l'authentification "
            "(SMTP_USERNAME / SMTP_PASSWORD). Sur Gmail, un "
            "mot de passe d'application est obligatoire : le "
            "mot de passe principal du compte est toujours "
            "rejeté."
        )

    if isinstance(error, smtplib.SMTPSenderRefused):
        return (
            "L'expéditeur EMAIL_SENDER a été refusé. Il doit "
            "correspondre au compte authentifié par "
            "SMTP_USERNAME ou à un alias autorisé."
        )

    if isinstance(error, smtplib.SMTPRecipientsRefused):
        return (
            "Le destinataire a été refusé par le serveur : "
            "vérifiez l'adresse email saisie par le citoyen."
        )

    if isinstance(error, smtplib.SMTPNotSupportedError):
        return (
            "Le serveur ne supporte pas l'option demandée. "
            "Vérifiez la cohérence de SMTP_USE_TLS et "
            "SMTP_USE_SSL avec le port utilisé (587 pour "
            "STARTTLS, 465 pour SSL direct)."
        )

    if isinstance(error, ssl.SSLError):
        return (
            "Échec de la négociation TLS. Un port 465 exige "
            "SMTP_USE_SSL=true ; un port 587 exige "
            "SMTP_USE_TLS=true."
        )

    if isinstance(error, (TimeoutError, OSError)):
        return (
            "Le serveur SMTP est injoignable (hôte, port ou "
            "réseau). Vérifiez SMTP_HOST et SMTP_PORT, et que "
            "le port sortant n'est pas bloqué."
        )

    return (
        "Échec d'envoi SMTP. Consultez le journal du serveur "
        "pour le détail de l'exception."
    )


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
        logger.info(
            "Email non envoyé (EMAIL_ENABLED=false) : %s",
            subject,
        )

        result = {
            "status": "disabled",
            "to": to,
            "subject": subject,
            "detail": (
                "Les notifications email sont désactivées "
                "(EMAIL_ENABLED=false)."
            ),
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

        logger.warning(
            "SMTP_HOST absent : email simulé en console "
            "pour %s.",
            mask_identifier(to),
        )

        result = {
            "status": "console",
            "to": to,
            "subject": subject,
            "body": body,
            "detail": (
                "Aucun SMTP_HOST configuré : le message a été "
                "affiché dans la console au lieu d'être envoyé."
            ),
        }
        _trace(result)

        return result

    message = EmailMessage()
    message["From"] = formataddr(
        ("NARSA", settings.email_sender)
    )
    message["To"] = to
    message["Subject"] = subject
    message["Message-ID"] = make_msgid(domain="narsa.gov.ma")
    message.set_content(body)

    logger.info(
        "Envoi SMTP vers %s via %s:%s (tls=%s, ssl=%s, "
        "compte=%s)",
        mask_identifier(to),
        settings.smtp_host,
        settings.smtp_port,
        settings.smtp_use_tls,
        settings.smtp_use_ssl,
        mask_identifier(settings.smtp_username) or "anonyme",
    )

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

                refused = server.send_message(message)

        else:
            with smtplib.SMTP(
                settings.smtp_host,
                settings.smtp_port,
                timeout=settings.smtp_timeout,
            ) as server:
                server.ehlo()

                if settings.smtp_use_tls:
                    server.starttls(
                        context=ssl.create_default_context()
                    )
                    server.ehlo()

                if settings.smtp_username:
                    server.login(
                        settings.smtp_username,
                        settings.smtp_password,
                    )

                refused = server.send_message(message)

        # `send_message` retourne les destinataires refusés
        # alors que d'autres ont été acceptés : ce n'est pas
        # une exception, mais ce n'est pas un succès non plus.
        if refused:
            logger.error(
                "Destinataire refusé par %s : %s",
                settings.smtp_host,
                sorted(
                    mask_identifier(address)
                    for address in refused
                ),
            )

            result = {
                "status": "failed",
                "to": to,
                "subject": subject,
                "error": "SMTPRecipientsRefused",
                "detail": (
                    "Le serveur SMTP a refusé le destinataire."
                ),
            }

        else:
            logger.info(
                "Email accepté par %s pour %s.",
                settings.smtp_host,
                mask_identifier(to),
            )

            result = {
                "status": "sent",
                "to": to,
                "subject": subject,
                "detail": (
                    "Message accepté par le serveur "
                    f"{settings.smtp_host}."
                ),
            }

    except Exception as error:  # pragma: no cover - dépend du réseau
        diagnosis = _diagnose(error)

        # Le mot de passe n'apparaît jamais : seuls l'hôte,
        # le port et le compte masqué sont journalisés.
        logger.error(
            "Envoi d'email impossible vers %s via %s:%s — "
            "%s: %s | %s",
            mask_identifier(to),
            settings.smtp_host,
            settings.smtp_port,
            type(error).__name__,
            error,
            diagnosis,
            exc_info=True,
        )

        print(
            "Envoi d'email impossible :",
            f"{type(error).__name__}: {error}",
        )

        result = {
            "status": "failed",
            "to": to,
            "subject": subject,
            "error": f"{type(error).__name__}: {error}",
            "detail": diagnosis,
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


def send_test_email(to: str) -> dict[str, Any]:
    """
    Vérifie la chaîne SMTP de bout en bout sans toucher à une
    réclamation réelle. Réservé à l'espace agent.
    """
    return send_email(
        to=to,
        subject="Test de configuration SMTP — portail NARSA",
        body=(
            "Bonjour,\n\n"
            "Ce message confirme que la configuration SMTP du "
            "portail NARSA est opérationnelle.\n\n"
            f"Portail de suivi : {settings.portal_url}/#/suivi\n\n"
            "Ceci est un message automatique de test."
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
