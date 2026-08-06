from __future__ import annotations

from typing import Any


def sources_for_display(
    response_data: dict[str, Any],
    technical_details_enabled: bool,
) -> list[dict[str, Any]]:
    """Expose les sources uniquement dans le mode technique."""
    if not technical_details_enabled:
        return []

    return list(
        response_data.get("sources", [])
    )


def local_forms_for_display(
    forms: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Ne conserve que les vrais PDF locaux."""
    return [
        form
        for form in forms
        if form.get("filename")
        and not form.get("is_external", False)
    ]


def service_links_for_display(
    response_data: dict[str, Any],
) -> list[dict[str, str]]:
    """Retourne les actions vers des services officiels vérifiés."""
    return [
        link
        for link in response_data.get("service_links", [])
        if link.get("label") and link.get("url")
    ]


def previous_context_from_messages(
    messages: list[dict[str, Any]],
) -> dict[str, str] | None:
    """Construit le contexte du dernier échange métier réussi."""
    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue

        metadata = message.get("metadata")

        if not isinstance(metadata, dict):
            continue

        domain = metadata.get("domain")
        intent = metadata.get("intent")
        question = metadata.get("question")

        if domain and intent and question:
            return {
                "previous_question": str(question),
                "previous_domain": str(domain),
                "previous_intent": str(intent),
            }

    return None
