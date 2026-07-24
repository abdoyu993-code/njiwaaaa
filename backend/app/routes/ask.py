from __future__ import annotations

import unicodedata

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


FORM_MAPPING: dict[str, list[dict[str, str]]] = {
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


def normalize_text(
    text: str,
) -> str:
    normalized = unicodedata.normalize(
        "NFD",
        text.lower(),
    )

    return "".join(
        character
        for character in normalized
        if unicodedata.category(character) != "Mn"
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
        or "carte grise" in text
    ):
        return "mutation_vehicule"

    if (
        "immatriculation" in text
        or "immatriculer" in text
        or "premiere mise en circulation" in text
    ):
        return "immatriculation_vehicule"

    return None


@router.post(
    "",
    response_model=AskResponse,
)
def ask_question(
    request: AskRequest,
) -> AskResponse:
    try:
        explicit_form_intent = (
            detect_explicit_form_intent(
                request.question
            )
        )

        result = generate_answer(
            question=request.question,
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

        # Une demande explicite de formulaire est considérée
        # comme comprise même si le classificateur général
        # avait une confiance inférieure au seuil.
        if explicit_form_intent:
            answer = FORM_ANSWERS.get(
                explicit_form_intent,
                (
                    "Voici les formulaires correspondant "
                    "à votre demande."
                ),
            )

            status = "form_found"
            needs_clarification = False
            generated = False

        # Sans détection explicite, ne pas afficher de PDF
        # lorsque la classification reste incertaine.
        elif needs_clarification:
            available_forms = []

        response_data = {
            "question": result["question"],
            "answer": answer,
            "status": status,
            "domain": result["domain"],
            "intent": (
                explicit_form_intent
                or classifier_intent
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
                if explicit_form_intent is None
                else []
            ),
            "forms": available_forms,
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
                    explicit_form_intent
                ),
                "original_classifier_intent": (
                    classifier_intent
                ),
                "classification": result.get(
                    "classification",
                    {},
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