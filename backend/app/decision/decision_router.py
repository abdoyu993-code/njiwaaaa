from __future__ import annotations

from typing import Any

from backend.app.classifier.classifier_service import (
    predict_intent,
)


MCP_INTENTS = {
    "prise_rendez_vous",
    "visite_technique_rdv",
    "suivi_demande",
    "paiement_amende",
}

DIRECT_RESPONSE_INTENTS = {
    "appel_urgence",
}

DEFAULT_CONFIDENCE_THRESHOLD = 0.60


def select_route(
    classification: dict[str, Any],
) -> tuple[str, str]:
    """
    Sélectionne le traitement adapté à la question.
    """
    intent = classification["intent"]
    confidence = classification["confidence"]
    needs_clarification = classification[
        "needs_clarification"
    ]

    if (
        needs_clarification
        or confidence < DEFAULT_CONFIDENCE_THRESHOLD
    ):
        return (
            "clarification",
            "La confiance du classifieur est insuffisante.",
        )

    if intent in MCP_INTENTS:
        return (
            "mcp",
            "Cette intention nécessite une action ou une simulation.",
        )

    if intent in DIRECT_RESPONSE_INTENTS:
        return (
            "direct_response",
            "Cette intention nécessite une réponse immédiate.",
        )

    return (
        "rag",
        "Cette intention nécessite une recherche documentaire.",
    )


def route_question(
    text: str,
    top_k: int = 3,
) -> dict[str, Any]:
    """
    Classifie une question puis sélectionne son traitement.
    """
    classification = predict_intent(
        text=text,
        top_k=top_k,
    )

    route, reason = select_route(classification)

    return {
        "text": text,
        "route": route,
        "reason": reason,
        "classification": classification,
    }