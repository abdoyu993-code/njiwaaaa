from __future__ import annotations

import re
import unicodedata
from time import perf_counter
from typing import Any

from fastapi import APIRouter, HTTPException

from backend.app.ai.language import (
    detect_language,
    to_search_query,
)
from backend.app.ai.suggestions import related_questions
from backend.app.analytics.service import (
    log_conversation,
)
from backend.app.core.config import settings
from backend.app.database.session import (
    SessionLocal,
    ensure_database,
)
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


GREETING_MARKERS = (
    "bonjour",
    "bonsoir",
    "salut",
    "salam",
    "assalam",
    "salamo",
    "ahlan",
    "hello",
    "hi",
    "coucou",
    "labas",
    "sbah lkhir",
    "msa lkhir",
    # Arabe et darija en alphabet arabe
    "السلام عليكم",
    "السلام",
    "سلام",
    "صباح الخير",
    "مساء الخير",
    "مرحبا",
    "أهلا",
    "اهلا",
    "تحية",
)

THANKS_MARKERS = (
    "merci",
    "chokran",
    "choukran",
    "shokran",
    "thanks",
    "شكرا",
    "شكرًا",
    "بارك الله فيك",
    "الله يعطيك الصحة",
)

GOODBYE_MARKERS = (
    "au revoir",
    "bslama",
    "b slama",
    "bye",
    "a bientot",
    "bonne journee",
    "bonne soiree",
    "بسلامة",
    "مع السلامة",
    "الى اللقاء",
)

# Réponses de politesse dans la langue du citoyen.
CONVERSATION_ANSWERS: dict[str, dict[str, str]] = {
    "salutation": {
        "fr": (
            "Bonjour 👋 Je suis l’assistant de la NARSA. "
            "Comment puis-je vous aider ? Vous pouvez me "
            "poser une question sur le permis, la carte "
            "grise, la visite technique, les infractions "
            "ou la prévention routière."
        ),
        "ar": (
            "السلام عليكم 👋 أنا المساعد الذكي للوكالة الوطنية "
            "للسلامة الطرقية. كيف يمكنني مساعدتك؟ يمكنك أن "
            "تسألني عن رخصة السياقة، البطاقة الرمادية، الفحص "
            "التقني، المخالفات أو السلامة الطرقية."
        ),
        "darija": (
            "السلام 👋 أنا المساعد الذكي ديال NARSA. كيفاش "
            "نقدر نعاونك؟ تقدر تسولني على رخصة السياقة، "
            "البطاقة الرمادية، الفحص التقني ولا المخالفات."
        ),
        "arabizi": (
            "Salam 👋 Ana l'assistant dyal NARSA. Kifach "
            "n3awnek ? Tqder tsewelni 3la permis, carte "
            "grise, visite technique wla les amendes."
        ),
    },
    "remerciement": {
        "fr": (
            "Avec plaisir. N’hésitez pas si vous avez une "
            "autre question sur les services de la NARSA."
        ),
        "ar": (
            "العفو. لا تتردد في طرح أي سؤال آخر حول خدمات "
            "الوكالة الوطنية للسلامة الطرقية."
        ),
        "darija": (
            "بلا جميل. إلا كان عندك شي سؤال آخر على خدمات "
            "NARSA أنا هنا."
        ),
        "arabizi": (
            "Bla jmil. Ila kan 3andek chi sou2al akhor 3la "
            "khadamat NARSA, ana hna."
        ),
    },
    "au_revoir": {
        "fr": "Au revoir 👋 Bonne route et prudence.",
        "ar": "إلى اللقاء 👋 طريق آمنة وكن حذرا.",
        "darija": "بسلامة 👋 طريق آمنة وردّ بالك.",
        "arabizi": "Bslama 👋 Triq salama w rd balek.",
    },
}

CONVERSATION_PATTERNS: tuple[
    tuple[tuple[str, ...], str],
    ...,
] = (
    (GREETING_MARKERS, "salutation"),
    (THANKS_MARKERS, "remerciement"),
    (GOODBYE_MARKERS, "au_revoir"),
)

MAX_CONVERSATION_WORDS = 6


def conversation_answer(
    intent: str,
    language: str,
) -> str:
    """Réponse de politesse dans la langue détectée."""
    answers = CONVERSATION_ANSWERS[intent]

    return answers.get(language, answers["fr"])


def detect_simple_message(
    question: str,
) -> tuple[str, str] | None:
    """
    Reconnaît les messages de politesse en français, en arabe,
    en darija et en arabizi, et répond dans la même langue.
    """
    normalized = normalize_text(question)
    language = detect_language(question)

    exact_match = SIMPLE_MESSAGE_RESPONSES.get(normalized)

    if exact_match:
        return (
            exact_match[0],
            conversation_answer(exact_match[0], language),
        )

    if not normalized:
        return None

    words = normalized.split()

    if len(words) > MAX_CONVERSATION_WORDS:
        return None

    # Une question métier reste traitée par le pipeline RAG,
    # même lorsqu'elle commence par une salutation.
    if has_explicit_business_topic(normalized):
        return None

    if "?" in question and len(words) > 2:
        return None

    for markers, intent in CONVERSATION_PATTERNS:
        if any(
            normalized == normalize_text(marker)
            or normalized.startswith(
                f"{normalize_text(marker)} "
            )
            or f" {normalize_text(marker)} "
            in f" {normalized} "
            for marker in markers
        ):
            return (
                intent,
                conversation_answer(intent, language),
            )

    return None


# Amorces de suivi : « et le formulaire ? », « ok et après ? »
FOLLOWUP_STARTERS = (
    "et ",
    "ok ",
    "d accord",
    "daccord",
    "puis ",
    "ensuite",
    "aussi ",
    "et alors",
)

# Sujets qui, sans nouveau thème métier, prolongent la démarche
# en cours : documents, dépôt, délais, coût, lieu…
FOLLOWUP_TOPICS = (
    "formulaire",
    "document",
    "documents",
    "piece",
    "pieces",
    "dossier",
    "delai",
    "delais",
    "prix",
    "cout",
    "tarif",
    "combien",
    "ou deposer",
    "ou le deposer",
    "ou la deposer",
    "deposer",
    "depot",
    "agence",
    "lien",
    "site",
    "adresse",
    "apres",
    "ensuite",
    "etape",
    "etapes",
    "duree",
    "rendez vous",
)


def searchable_text(question: str) -> str:
    """Texte normalisé et ramené au français pour l'analyse."""
    return normalize_text(to_search_query(question))


def is_context_dependent_message(
    question: str,
) -> bool:
    """
    Détecte un message qui prolonge l'échange précédent.

    « Et le formulaire ? » puis « Et où déposer le dossier ? »
    restent rattachés à la démarche en cours.
    """
    normalized = searchable_text(question)

    if not normalized:
        return False

    words = normalized.split()

    if len(normalized) > 120 or len(words) > 14:
        return False

    if any(
        marker in normalized
        for marker in DEPENDENT_MESSAGE_MARKERS
    ):
        return True

    starts_as_followup = any(
        normalized.startswith(starter)
        for starter in FOLLOWUP_STARTERS
    )

    mentions_followup_topic = any(
        topic in normalized
        for topic in FOLLOWUP_TOPICS
    )

    # Une question courte sans sujet propre : « et après ? »
    if starts_as_followup and len(words) <= 8:
        return True

    return starts_as_followup and mentions_followup_topic or (
        mentions_followup_topic and len(words) <= 6
    )


def has_explicit_business_topic(
    question: str,
) -> bool:
    """Indique si la question introduit une nouvelle démarche métier."""
    normalized = searchable_text(question)

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


PERMIT_FORM_INTENT = "permis_duplicata_service"

# Racines tolérantes aux fautes : « formulaire », « formulair »,
# « fomulaire », « imprimé », « télécharger »…
FORM_REQUEST_PATTERN = re.compile(
    r"\b("
    r"formul\w*|fomul\w*|furmul\w*"
    r"|pdf"
    r"|telecharg\w*|telechrg\w*"
    r"|imprim\w*"
    r"|document[s]? a remplir"
    r"|dossier a remplir"
    r"|dossier d? ?immatriculation"
    r"|مطبوع|استمارة|الاستمارة"
    r")\b"
)

PERMIT_LOSS_MARKERS = (
    "permis perdu",
    "perdu mon permis",
    "perdu le permis",
    "perte de permis",
    "perte du permis",
    "permis vole",
    "vol de permis",
    "vol du permis",
    "duplicata de permis",
    "duplicata du permis",
    "duplicata permis",
    "refaire mon permis",
    "refaire le permis",
)


def detect_explicit_form_intent(
    question: str,
) -> str | None:
    """
    Détecte une demande explicite de formulaire ou de PDF,
    y compris formulée naturellement ou avec des fautes.
    """
    text = searchable_text(question)

    if not FORM_REQUEST_PATTERN.search(text):
        return None

    # Le duplicata de permis n'a pas de PDF local : on oriente
    # vers le service officiel plutôt que d'inventer un document.
    if any(
        marker in text
        for marker in PERMIT_LOSS_MARKERS
    ) or ("permis" in text and "duplicata" in text):
        return PERMIT_FORM_INTENT

    if (
        "contravention" in text
        or "amende" in text
        or "infraction" in text
        or "reclam" in text
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
        or "vehicule neuf" in text
        or "vehicule importe" in text
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


def log_interaction(
    request: AskRequest,
    response: AskResponse,
) -> int | None:
    """
    Journalise l'échange pour le Dashboard IA.

    Retourne l'identifiant de l'interaction, qui permet au
    citoyen de donner son avis sur la réponse.
    """
    if not settings.log_conversations:
        return None

    try:
        ensure_database()

        with SessionLocal() as session:
            return log_conversation(
                session=session,
                question=request.question,
                answer=response.answer,
                domain=response.domain,
                intent=response.intent,
                confidence=response.confidence,
                status=response.status,
                generated=response.generated,
                extractive=bool(
                    response.metadata.get("extractive")
                ),
                needs_clarification=(
                    response.needs_clarification
                ),
                total_time_ms=float(
                    response.metadata.get(
                        "total_time_ms",
                        0.0,
                    )
                ),
                session_id=request.session_id,
                language=str(
                    response.metadata.get("language", "fr")
                ),
            )

    except Exception as error:  # pragma: no cover
        print(
            "Journalisation de la conversation impossible :",
            f"{type(error).__name__}: {error}",
        )

    return None


@router.post(
    "",
    response_model=AskResponse,
)
def ask_question(
    request: AskRequest,
) -> AskResponse:
    response = build_ask_response(request)

    interaction_id = log_interaction(
        request=request,
        response=response,
    )

    if interaction_id is not None:
        response.metadata["interaction_id"] = interaction_id

    return response


def build_ask_response(
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

        if explicit_form_intent == PERMIT_FORM_INTENT:
            # Aucun PDF local n'existe pour le duplicata du
            # permis : on renvoie le service officiel.
            service_url = PERMIT_DUPLICATE_SERVICE["url"]

            return AskResponse(
                question=request.question,
                answer=(
                    "Aucun formulaire PDF local n’est disponible "
                    "pour le duplicata du permis de conduire.\n\n"
                    "La demande se dépose dans une agence Al Barid "
                    "Bank ou Barid Cash offrant les services "
                    "NARSA, avec le formulaire remis sur place.\n\n"
                    "Service officiel en ligne :\n"
                    f"{service_url}"
                ),
                status="service_link_found",
                domain="permis",
                intent="permis_duplicata",
                confidence=1.0,
                needs_clarification=False,
                generated=False,
                model=None,
                finish_reason=None,
                sources=[],
                forms=[],
                service_links=[PERMIT_DUPLICATE_SERVICE],
                metadata={
                    "handled_locally": True,
                    "explicit_form_intent": (
                        PERMIT_FORM_INTENT
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

        # Français, arabe, darija ou arabizi : la recherche
        # documentaire se fait toujours en français, la réponse
        # dans la langue du citoyen.
        language = detect_language(request.question)
        search_question = to_search_query(
            text=effective_question,
            language=language,
        )

        result = generate_answer(
            question=effective_question,
            top_k=request.top_k,
            language=language,
            search_question=search_question,
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
        # reste incertaine, y compris sur une réponse obtenue
        # par rattrapage documentaire.
        if needs_clarification or result.get(
            "low_confidence_rescue"
        ):
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
            "related_questions": related_questions(
                question=request.question,
                intent=classifier_intent,
                domain=result["domain"],
            ),
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
                "cached": result.get("cached", False),
                "language": language,
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
        # Le détail technique reste dans les journaux du serveur :
        # il n'a pas à être exposé au citoyen.
        print(
            "Ressource RAG introuvable :",
            f"{type(error).__name__}: {error}",
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "Le service documentaire est momentanément "
                "indisponible. Réessayez dans quelques instants."
            ),
        ) from error

    except Exception as error:
        print(
            "Erreur pendant le traitement de la demande :",
            f"{type(error).__name__}: {error}",
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Une erreur est survenue pendant le traitement "
                "de votre demande. Réessayez dans quelques "
                "instants."
            ),
        ) from error
