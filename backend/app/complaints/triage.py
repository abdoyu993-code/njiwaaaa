from __future__ import annotations

import unicodedata
from typing import Any

from backend.app.schemas.complaints import (
    COMPLAINT_CATEGORIES,
    COMPLAINT_CATEGORY_LABELS,
)


PRIORITY_LABELS = {
    "urgente": "Urgente",
    "haute": "Haute",
    "normale": "Normale",
}

PRIORITIES = tuple(PRIORITY_LABELS)

# Situations mettant en jeu la sécurité ou une échéance légale.
URGENT_MARKERS = (
    "accident",
    "blesse",
    "blessee",
    "hopital",
    "urgence",
    "urgent",
    "danger",
    "dangereux",
    "deces",
    "mort",
    "gravement",
    "risque immediat",
)

# Signaux d'un dossier qui traine ou d'un délai réglementaire.
HIGH_PRIORITY_MARKERS = (
    "aucune reponse",
    "sans reponse",
    "toujours pas",
    "depuis des mois",
    "depuis deux mois",
    "depuis trois mois",
    "plusieurs relances",
    "relance",
    "deuxieme fois",
    "troisieme fois",
    "delai depasse",
    "hors delai",
    "penalite",
    "majoration",
    "saisie",
    "retard important",
    "bloque",
    "bloquee",
)

DEADLINE_MARKERS = (
    "30 jours",
    "delai de 30",
    "mise en demeure",
    "convocation",
    "tribunal",
)


def normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.lower())

    without_accents = "".join(
        character
        for character in decomposed
        if unicodedata.category(character) != "Mn"
    )

    return " ".join(without_accents.split())


def detect_priority(
    text: str,
    domain: str | None = None,
) -> tuple[str, list[str]]:
    """
    Estime la priorité de traitement et explique la décision.

    Le résultat est une aide au tri pour l'agent : il reste
    modifiable et n'a aucune valeur réglementaire.
    """
    normalized = normalize(text)
    reasons: list[str] = []

    urgent_hits = [
        marker
        for marker in URGENT_MARKERS
        if marker in normalized
    ]

    if urgent_hits or domain == "accident":
        if urgent_hits:
            reasons.append(
                "Situation de sécurité mentionnée : "
                + ", ".join(sorted(set(urgent_hits))[:3])
            )
        else:
            reasons.append(
                "Réclamation rattachée au domaine des accidents"
            )

        return "urgente", reasons

    high_hits = [
        marker
        for marker in HIGH_PRIORITY_MARKERS
        if marker in normalized
    ]
    deadline_hits = [
        marker
        for marker in DEADLINE_MARKERS
        if marker in normalized
    ]

    if high_hits:
        reasons.append(
            "Dossier en attente ou relancé : "
            + ", ".join(sorted(set(high_hits))[:3])
        )

    if deadline_hits:
        reasons.append(
            "Échéance réglementaire évoquée : "
            + ", ".join(sorted(set(deadline_hits))[:3])
        )

    if high_hits or deadline_hits:
        return "haute", reasons

    return "normale", ["Aucun signal d'urgence détecté"]


def category_from_domain(domain: str | None) -> str:
    """Convertit un domaine métier en catégorie de réclamation."""
    if domain and domain in COMPLAINT_CATEGORIES:
        return domain

    return "autre"


def analyze_complaint_text(
    text: str,
    classification: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Assemble le pré-diagnostic d'une réclamation : catégorie
    proposée, priorité estimée et intention détectée.
    """
    domain = (
        (classification or {}).get("domain")
        if classification
        else None
    )
    confidence = float(
        (classification or {}).get("confidence", 0.0)
    )

    # Une classification incertaine ne doit pas imposer une
    # catégorie : l'agent et le citoyen gardent la main.
    reliable = confidence >= 0.60
    category = (
        category_from_domain(domain)
        if reliable
        else "autre"
    )

    priority, reasons = detect_priority(
        text=text,
        domain=domain if reliable else None,
    )

    return {
        "category": category,
        "category_label": COMPLAINT_CATEGORY_LABELS[category],
        "category_confidence": round(confidence, 2),
        "category_is_reliable": reliable,
        "detected_intent": (
            (classification or {}).get("intent")
            if reliable
            else None
        ),
        "priority": priority,
        "priority_label": PRIORITY_LABELS[priority],
        "priority_reasons": reasons,
    }
