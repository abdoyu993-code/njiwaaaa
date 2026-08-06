from __future__ import annotations

import unicodedata
from time import perf_counter
from typing import Any

from fastapi import APIRouter, HTTPException

from backend.app.rag.generation_service import (
    generate_answer,
)
from backend.app.schemas.ask import (
    AskRequest,
    AskResponse,
)


router = APIRouter(
    prefix="/ask",
    tags=["Assistant RAG"],
)


FORM_MAPPING: dict[str, list[dict[str, Any]]] = {
    "contestation_amende": [
        {
            "title": (
                "Formulaire de réclamation "
                "sur avis de contravention"
            ),
            "filename": (
                "formulaire_reclamation_contravention.pdf"
            ),
            "download_url": (
                "/forms/"
                "formulaire_reclamation_contravention.pdf"
            ),
        }
    ],

    "immatriculation_vehicule": [
        {
            "title": (
                "Dossier d’immatriculation - Formule I"
            ),
            "filename": (
                "formulaire_immatriculation_formule_i.pdf"
            ),
            "download_url": (
                "/forms/"
                "formulaire_immatriculation_formule_i.pdf"
            ),
        },
        {
            "title": (
                "Dossier d’immatriculation - Formule II"
            ),
            "filename": (
                "formulaire_immatriculation_formule_ii.pdf"
            ),
            "download_url": (
                "/forms/"
                "formulaire_immatriculation_formule_ii.pdf"
            ),
        },
    ],

    "mutation_vehicule": [
        {
            "title": (
                "Dossier carte grise - Formule BI"
            ),
            "filename": (
                "formulaire_carte_grise_formule_bi.pdf"
            ),
            "download_url": (
                "/forms/"
                "formulaire_carte_grise_formule_bi.pdf"
            ),
        },
        {
            "title": (
                "Dossier carte grise - Formule BII"
            ),
            "filename": (
                "formulaire_carte_grise_formule_bii.pdf"
            ),
            "download_url": (
                "/forms/"
                "formulaire_carte_grise_formule_bii.pdf"
            ),
        },
    ],
}


FORM_ANSWERS: dict[str, str] = {
    "contestation_amende": (
        "Voici le formulaire de réclamation sur avis "
        "de contravention.\n\n"
        "Vous devez le remplir, le signer et joindre :\n\n"
        "- une copie de l’avis de contravention ;\n"
        "- les justificatifs correspondant au motif "
        "de votre contestation.\n\n"
        "La réclamation doit être déposée dans un délai "
        "de 30 jours à compter de la date de notification "
        "de l’avis de contravention.\n\n"
        "Vous pouvez télécharger le formulaire ci-dessous."
    ),

    "immatriculation_vehicule": (
        "Voici les dossiers d’immatriculation disponibles.\n\n"
        "Les documents proposés sont les Formules I et II. "
        "Téléchargez les deux documents ci-dessous et vérifiez "
        "auprès du centre d’immatriculation la formule applicable "
        "à votre situation."
    ),

    "mutation_vehicule": (
        "Voici les dossiers disponibles pour une opération "
        "de mutation ou de carte grise.\n\n"
        "Les documents proposés sont les Formules BI et BII. "
        "Vous pouvez télécharger les deux PDF ci-dessous."
    ),
}

PERMIT_DUPLICATE_SERVICE: dict[str, str] = {
    "label": "Accéder au service officiel de duplicata du permis",
    "url": (
        "https://khadamatnarsa.ma/fr/services/"
        "duplicata-de-mon-permis-de-conduire"
    ),
}

PERMIT_DUPLICATE_INTENTS = {
    "permis_perte",
    "permis_vol",
    "permis_duplicata",
}

FORM_INTENT_ALIASES = {
    "reclamation_infraction": "contestation_amende",
    "changement_proprietaire": "mutation_vehicule",
    "vehicule_vendu": "mutation_vehicule",
}

GENERIC_CARTE_GRISE_INTENT = "carte_grise_demande"

EXPLICIT_BUSINESS_TOPIC_MARKERS = (
    "permis",
    "carte grise",
    "immatriculation",
    "immatriculer",
    "mutation",
    "changement proprietaire",
    "vente vehicule",
    "vehicule vendu",
    "contravention",
    "amende",
    "infraction",
    "visite technique",
    "controle technique",
    "accident",
    "assurance",
    "ceinture",
    "casque",
    "telephone au volant",
    "fatigue",
)

SPECIFIC_CARTE_GRISE_MARKERS = (
    "mutation",
    "changement proprietaire",
    "vente vehicule",
    "vehicule vendu",
    "immatriculation",
    "immatriculer",
    "premiere mise en circulation",
    "duplicata",
    "perte",
    "vol",
    "modification",
)

DEPENDENT_MESSAGE_MARKERS = (
    "le formulaire",
    "donne moi",
    "donne le moi",
    "et les documents",
    "ou je le depose",
    "ou la deposer",
    "combien ca coute",
    "quel est le lien",
    "quelle est le lien",
    "et le lien",
)

FORM_FOLLOWUP_MARKERS = (
    "formulaire",
    "donne le moi",
)

SIMPLE_MESSAGE_RESPONSES: dict[str, tuple[str, str]] = {
    "bonjour": (
        "salutation",
        "Bonjour 👋 Comment puis-je vous aider concernant les services de la NARSA ?",
    ),
    "bonsoir": (
        "salutation",
        "Bonsoir 👋 Comment puis-je vous aider concernant les services de la NARSA ?",
    ),
    "salut": (
        "salutation",
        "Salut 👋 Comment puis-je vous aider concernant les services de la NARSA ?",
    ),
    "salam": (
        "salutation",
        "Salam 👋 Comment puis-je vous aider concernant les services de la NARSA ?",
    ),
    "merci": (
        "remerciement",
        "Avec plaisir. N’hésitez pas si vous avez une autre question sur les services de la NARSA.",
    ),
    "au revoir": (
        "au_revoir",
        "Au revoir 👋",
    ),
}

FORM_DOMAINS: dict[str, str] = {
    "contestation_amende": "infractions",
    "mutation_vehicule": "vehicule",
    "immatriculation_vehicule": "vehicule",
}


def normalize_text(
    text: str,
) -> str:
    normalized = unicodedata.normalize(
        "NFD",
        text.lower(),
    )

    without_accents = "".join(
        character
        for character in normalized
        if unicodedata.category(character) != "Mn"
    )

    without_punctuation = "".join(
        character
        if (
            character.isalnum()
            or character.isspace()
        )
        else " "
        for character in without_accents
    )

    return " ".join(
        without_punctuation.split()
    )


def detect_simple_message(
    question: str,
) -> tuple[str, str] | None:
    return SIMPLE_MESSAGE_RESPONSES.get(
        normalize_text(question)
    )


def is_context_dependent_message(
    question: str,
) -> bool:
    normalized = normalize_text(question)

    if len(normalized) > 100:
        return False

    return any(
        marker in normalized
        for marker in DEPENDENT_MESSAGE_MARKERS
    )


def has_explicit_business_topic(
    question: str,
) -> bool:
    """Indique si la question introduit une nouvelle démarche métier."""
    normalized = normalize_text(question)

    return any(
        marker in normalized
        for marker in EXPLICIT_BUSINESS_TOPIC_MARKERS
    )


def is_form_followup(
    question: str,
) -> bool:
    normalized = normalize_text(question)
    return any(
        marker in normalized
        for marker in FORM_FOLLOWUP_MARKERS
    )


def contextual_forms_for_intent(
    intent: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, str]],
    str,
] | None:
    if intent in PERMIT_DUPLICATE_INTENTS:
        service_url = PERMIT_DUPLICATE_SERVICE["url"]
        return (
            [],
            [PERMIT_DUPLICATE_SERVICE],
            (
                "Aucun formulaire PDF local n’est disponible pour "
                "cette démarche.\n\n"
                "Lien officiel du service de duplicata du permis :\n"
                f"{service_url}"
            ),
        )

    canonical_intent = FORM_INTENT_ALIASES.get(
        intent,
        intent,
    )
    forms = FORM_MAPPING.get(canonical_intent)

    if not forms:
        return None

    return (
        forms,
        [],
        FORM_ANSWERS.get(
            canonical_intent,
            "Voici la ressource disponible pour votre démarche.",
        ),
    )


def detect_explicit_form_intent(
    question: str,
) -> str | None:
    """
    Détecte une demande explicite de formulaire ou de PDF.
    """
    text = normalize_text(question)

    form_markers = (
        "telecharger",
        "formulaire",
        "pdf",
        "document a remplir",
        "documents a remplir",
        "dossier a remplir",
        "dossier d'immatriculation",
        "dossier immatriculation",
    )

    requests_form = any(
        marker in text
        for marker in form_markers
    )

    if not requests_form:
        return None

    if (
        "contravention" in text
        or "contester une amende" in text
        or "reclamation amende" in text
        or "reclamation contravention" in text
        or "avis de contravention" in text
    ):
        return "contestation_amende"

    if (
        "mutation" in text
        or "changement proprietaire" in text
        or "vente vehicule" in text
        or "vehicule vendu" in text
    ):
        return "mutation_vehicule"

    if (
        "immatriculation" in text
        or "immatriculer" in text
        or "premiere mise en circulation" in text
    ):
        return "immatriculation_vehicule"

    if (
        "carte grise" in text
        and not any(
            marker in text
            for marker in SPECIFIC_CARTE_GRISE_MARKERS
        )
    ):
        return GENERIC_CARTE_GRISE_INTENT

    return None


def build_carte_grise_form_clarification(
    question: str,
    request_started_at: float,
) -> AskResponse:
    """Demande le cas précis avant de proposer un PDF carte grise."""
    return AskResponse(
        question=question,
        answer=(
            "Pour vous proposer le bon formulaire de carte grise, "
            "précisez votre démarche :\n\n"
            "- immatriculation d’un véhicule ;\n"
            "- mutation ou changement de propriétaire.\n\n"
            "Des formulaires PDF locaux sont disponibles pour ces "
            "deux cas."
        ),
        status="clarification_required",
        domain="vehicule",
        intent=GENERIC_CARTE_GRISE_INTENT,
        confidence=1.0,
        needs_clarification=True,
        generated=False,
        model=None,
        finish_reason=None,
        sources=[],
        forms=[],
        service_links=[],
        metadata={
            "handled_locally": True,
            "explicit_form_intent": (
                GENERIC_CARTE_GRISE_INTENT
            ),
            "generation_attempts": 0,
            "used_contexts_count": 0,
            "classification_time_ms": 0.0,
            "retrieval_time_ms": 0.0,
            "generation_time_ms": 0.0,
            "total_time_ms": round(
                (perf_counter() - request_started_at) * 1000,
                2,
            ),
        },
    )


@router.post(
    "",
    response_model=AskResponse,
)
def ask_question(
    request: AskRequest,
) -> AskResponse:
    request_started_at = perf_counter()

    try:
        simple_message = detect_simple_message(
            request.question
        )

        if simple_message:
            intent, answer = simple_message

            return AskResponse(
                question=request.question,
                answer=answer,
                status="local_response",
                domain="conversation",
                intent=intent,
                confidence=1.0,
                needs_clarification=False,
                generated=False,
                model=None,
                finish_reason=None,
                sources=[],
                forms=[],
                metadata={
                    "handled_locally": True,
                    "generation_attempts": 0,
                    "used_contexts_count": 0,
                    "classification_time_ms": 0.0,
                    "retrieval_time_ms": 0.0,
                    "generation_time_ms": 0.0,
                    "total_time_ms": round(
                        (
                            perf_counter()
                            - request_started_at
                        )
                        * 1000,
                        2,
                    ),
                },
            )

        explicit_form_intent = detect_explicit_form_intent(
            request.question
        )
        has_new_business_topic = has_explicit_business_topic(
            request.question
        )

        if explicit_form_intent == GENERIC_CARTE_GRISE_INTENT:
            return build_carte_grise_form_clarification(
                question=request.question,
                request_started_at=request_started_at,
            )

        if explicit_form_intent:
            return AskResponse(
                question=request.question,
                answer=FORM_ANSWERS[
                    explicit_form_intent
                ],
                status="form_found",
                domain=FORM_DOMAINS[
                    explicit_form_intent
                ],
                intent=explicit_form_intent,
                confidence=1.0,
                needs_clarification=False,
                generated=False,
                model=None,
                finish_reason=None,
                sources=[],
                forms=FORM_MAPPING[
                    explicit_form_intent
                ],
                service_links=[],
                metadata={
                    "handled_locally": True,
                    "explicit_form_intent": (
                        explicit_form_intent
                    ),
                    "original_classifier_intent": None,
                    "classification": {},
                    "retrieval_query": None,
                    "generation_attempts": 0,
                    "used_contexts_count": 0,
                    "classification_time_ms": 0.0,
                    "retrieval_time_ms": 0.0,
                    "generation_time_ms": 0.0,
                    "total_time_ms": round(
                        (
                            perf_counter()
                            - request_started_at
                        )
                        * 1000,
                        2,
                    ),
                },
            )

        previous_context = request.context
        depends_on_context = bool(
            previous_context
            and not has_new_business_topic
            and is_context_dependent_message(
                request.question
            )
        )

        if (
            depends_on_context
            and previous_context is not None
            and is_form_followup(request.question)
        ):
            contextual_form_result = contextual_forms_for_intent(
                previous_context.previous_intent
            )

            if contextual_form_result:
                forms, service_links, answer = contextual_form_result

                return AskResponse(
                    question=request.question,
                    answer=answer,
                    status="contextual_resource_found",
                    domain=previous_context.previous_domain,
                    intent=previous_context.previous_intent,
                    confidence=1.0,
                    needs_clarification=False,
                    generated=False,
                    model=None,
                    finish_reason=None,
                    sources=[],
                    forms=forms,
                    service_links=service_links,
                    metadata={
                        "handled_locally": True,
                        "context_used": True,
                        "previous_question": (
                            previous_context.previous_question
                        ),
                        "generation_attempts": 0,
                        "used_contexts_count": 0,
                        "classification_time_ms": 0.0,
                        "retrieval_time_ms": 0.0,
                        "generation_time_ms": 0.0,
                        "total_time_ms": round(
                            (
                                perf_counter()
                                - request_started_at
                            )
                            * 1000,
                            2,
                        ),
                    },
                )

        effective_question = request.question

        if depends_on_context and previous_context is not None:
            effective_question = (
                f"{previous_context.previous_question}\n"
                "Question complémentaire : "
                f"{request.question}"
            )

        result = generate_answer(
            question=effective_question,
            top_k=request.top_k,
        )

        classifier_intent = result["intent"]

        selected_form_intent = (
            explicit_form_intent
            or classifier_intent
        )

        available_forms = FORM_MAPPING.get(
            selected_form_intent,
            [],
        )

        answer = result["answer"]
        status = result["status"]
        needs_clarification = result[
            "needs_clarification"
        ]
        generated = result["generated"]

        # Ne pas afficher de PDF lorsque la classification
        # reste incertaine.
        if needs_clarification:
            available_forms = []

        response_data = {
            "question": request.question,
            "answer": answer,
            "status": status,
            "domain": result["domain"],
            "intent": (
                classifier_intent
            ),
            "confidence": result["confidence"],
            "needs_clarification": (
                needs_clarification
            ),
            "generated": generated,
            "model": (
                result.get("model")
                if generated
                else None
            ),
            "finish_reason": (
                result.get("finish_reason")
                if generated
                else None
            ),
            "sources": (
                result.get("sources", [])
            ),
            "forms": available_forms,
            "service_links": [],
            "metadata": {
                "retrieval_query": result.get(
                    "retrieval_query"
                ),
                "used_contexts_count": result.get(
                    "used_contexts_count",
                    0,
                ),
                "generation_attempts": result.get(
                    "generation_attempts",
                    0,
                ),
                "explicit_form_intent": (
                    None
                ),
                "original_classifier_intent": (
                    classifier_intent
                ),
                "classification": result.get(
                    "classification",
                    {},
                ),
                "classification_time_ms": result.get(
                    "classification_time_ms",
                    0.0,
                ),
                "retrieval_time_ms": result.get(
                    "retrieval_time_ms",
                    0.0,
                ),
                "generation_time_ms": result.get(
                    "generation_time_ms",
                    0.0,
                ),
                "total_time_ms": round(
                    (
                        perf_counter()
                        - request_started_at
                    )
                    * 1000,
                    2,
                ),
                "extractive": result.get(
                    "extractive",
                    False,
                ),
                "context_used": depends_on_context,
                "resolved_question": (
                    effective_question
                    if depends_on_context
                    else None
                ),
            },
        }

        if result.get("generation_error"):
            response_data["metadata"][
                "generation_error"
            ] = result["generation_error"]

        return AskResponse(
            **response_data
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Une ressource nécessaire au RAG "
                f"est introuvable : {error}"
            ),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Erreur pendant le traitement "
                f"de la demande : {error}"
            ),
        ) from error
