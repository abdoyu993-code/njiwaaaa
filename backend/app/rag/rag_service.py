from __future__ import annotations

from time import perf_counter
from typing import Any

from backend.app.classifier.classifier_service import (
    predict_intent,
)
from backend.app.rag.retriever import NarsaRetriever


DEFAULT_CLARIFICATION_THRESHOLD = 0.60
DEFAULT_MINIMUM_SIMILARITY_SCORE = 0.30
DEFAULT_TOP_K = 3

# Seuil appliqué au rattrapage des questions dont la
# classification est incertaine : seule une correspondance
# documentaire forte permet de répondre.
RESCUE_MINIMUM_SIMILARITY_SCORE = 0.55

CARTE_GRISE_CASE_MARKERS = (
    "premiere immatriculation",
    "vehicule neuf",
    "importe",
    "importation",
    "mutation",
    "occasion",
    "vente",
    "echange",
    "duplicata",
    "perte",
    "vol",
    "modification",
)


INTENT_QUERY_HINTS: dict[str, str] = {
    "permis_perte": (
        " perte permis de conduire duplicata déclaration de perte "
        "documents nécessaires pour permis perdu remplacement permis"
    ),
    "permis_vol": (
        " vol permis de conduire duplicata déclaration de vol "
        "documents nécessaires pour permis volé remplacement permis"
    ),
    "permis_duplicata": (
        " duplicata permis de conduire documents nécessaires "
        "remplacement du permis perte ou vol"
    ),
    "permis_renouvellement": (
        " renouvellement permis de conduire expiration "
        "documents nécessaires dossier"
    ),
    "permis_expire": (
        " permis expiré renouvellement durée de validité "
        "documents nécessaires"
    ),
    "permis_deteriore": (
        " permis détérioré remplacement renouvellement "
        "documents nécessaires"
    ),
    "paiement_amende": (
        " paiement amende montant délai mode de paiement "
        "amende transactionnelle forfaitaire"
    ),
    "infraction_vitesse": (
        " excès de vitesse amende retrait de points "
        "délais et modes de paiement"
    ),
    "solde_points": (
        " consultation solde de points permis de conduire "
        "portail infractions routières"
    ),
    "reclamation_infraction": (
        " réclamation avis de contravention délai trente jours "
        "formulaire justificatifs"
    ),
    "accident_blesse": (
        " accident avec blessé protéger alerter secourir "
        "services de secours premiers gestes"
    ),
    "procedure_apres_accident": (
        " procédure après accident protéger alerter secourir "
        "constat assurance témoins"
    ),
    "visite_technique_rdv": (
        " prise de rendez-vous visite technique "
        "choisir centre date heure"
    ),
    "visite_technique_documents": (
        " documents nécessaires visite technique "
        "pièces à présenter contrôle technique"
    ),
    "mutation_vehicule": (
        " mutation véhicule changement propriétaire "
        "documents nécessaires carte grise délai trente jours"
    ),
    "changement_proprietaire": (
        " changement propriétaire véhicule mutation "
        "déclaration de vente dossier carte grise"
    ),
    "conseils_telephone": (
        " téléphone au volant distraction dangers "
        "conseils de prévention"
    ),
    "conseils_ceinture": (
        " ceinture de sécurité port obligatoire "
        "utilisation correcte prévention"
    ),
    "conseils_casque": (
        " casque moto deux roues protection "
        "choix et port du casque"
    ),
    "conseils_fatigue": (
        " fatigue somnolence au volant signes "
        "pauses prévention"
    ),
    "contact_narsa": (
        " coordonnées NARSA adresse téléphone email contact"
    ),
    "reclamation_narsa": (
        " réclamation administrative NARSA portail Chikaya "
        "suivi réclamation"
    ),
    "suivi_demande": (
        " suivi dossier NARSA numéro de dossier "
        "statut de la demande"
    ),
}


NARSA_CONTACT_HINT = (
    "Vous pouvez aussi contacter la NARSA ou déposer une "
    "réclamation depuis la plateforme pour obtenir une "
    "réponse d’un agent."
)

# Sujets réellement couverts par le corpus documentaire.
# Ils servent à orienter l'utilisateur quand la recherche
# ne trouve pas d'extrait fiable.
DOMAIN_COVERAGE: dict[str, tuple[str, ...]] = {
    "permis": (
        "duplicata après perte ou vol du permis",
        "échange ou renouvellement du permis",
        "plateforme e-Rokhsati",
    ),
    "vehicule": (
        "immatriculation d’un véhicule neuf ou importé",
        "mutation et changement de propriétaire",
        "carte grise et formulaires associés",
    ),
    "visite_technique": (
        "prise de rendez-vous pour le contrôle technique",
        "procès-verbal de contrôle technique exigé "
        "lors d’une mutation",
    ),
    "infractions": (
        "paiement d’une amende",
        "réclamation sur un avis de contravention",
        "constatation et traitement des infractions",
    ),
    "accident": (
        "conduite à tenir en cas d’accident ou de panne",
        "prise en charge d’un blessé",
        "équipements de sécurité obligatoires",
    ),
    "prevention": (
        "ceinture de sécurité et port du casque",
        "téléphone au volant, fatigue et somnolence",
        "conseils aux conducteurs débutants",
    ),
    "contact_reclamation": (
        "coordonnées et formulaire de contact NARSA",
        "portail Chikaya de réclamation",
        "suivi d’un dossier",
    ),
}


def format_coverage(domain: str) -> str:
    """Liste lisible des sujets couverts pour un domaine."""
    subjects = DOMAIN_COVERAGE.get(domain)

    if not subjects:
        return ""

    lines = "\n".join(
        f"- {subject} ;" for subject in subjects[:-1]
    )

    if lines:
        lines += f"\n- {subjects[-1]}."
    else:
        lines = f"- {subjects[-1]}."

    return lines


class NarsaRAGService:
    """
    Service combinant la classification et la recherche documentaire.
    """

    def __init__(
        self,
        clarification_threshold: float = (
            DEFAULT_CLARIFICATION_THRESHOLD
        ),
        minimum_similarity_score: float = (
            DEFAULT_MINIMUM_SIMILARITY_SCORE
        ),
    ) -> None:
        self.clarification_threshold = clarification_threshold
        self.minimum_similarity_score = minimum_similarity_score

        print("Chargement du retriever RAG...")
        self.retriever = NarsaRetriever()
        print("Service RAG chargé.")

    def search(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> dict[str, Any]:
        question = self._validate_question(question)
        top_k = max(1, top_k)

        classification_started_at = perf_counter()
        classification = predict_intent(
            text=question,
            top_k=3,
        )
        classification_time_ms = self._elapsed_ms(
            classification_started_at
        )

        domain = classification["domain"]
        intent = classification["intent"]
        confidence = float(
            classification["confidence"]
        )

        needs_clarification = (
            classification["needs_clarification"]
            or confidence < self.clarification_threshold
        )

        if needs_clarification:
            # Le classifieur peut sous-estimer une question
            # pourtant claire. On tente une recherche
            # documentaire directe avant de demander une
            # reformulation, avec une exigence de score élevée.
            retrieval_started_at = perf_counter()
            rescued = self._rescue_low_confidence_question(
                question=question,
                classification=classification,
                top_k=top_k,
            )
            retrieval_time_ms = self._elapsed_ms(
                retrieval_started_at
            )

            if rescued is not None:
                return {
                    **rescued,
                    "classification_time_ms": (
                        classification_time_ms
                    ),
                    "retrieval_time_ms": retrieval_time_ms,
                }

            result = self._build_clarification_response(
                question=question,
                classification=classification,
            )

            return {
                **result,
                "classification_time_ms": (
                    classification_time_ms
                ),
                "retrieval_time_ms": retrieval_time_ms,
            }

        needs_case_selection = (
            intent == "carte_grise_demande"
            and not self._contains_case_marker(question)
        )

        if needs_case_selection:
            result = self._build_clarification_response(
                question=question,
                classification=classification,
            )
            result["message"] = (
                "La démarche de carte grise dépend de votre cas :\n\n"
                "- première immatriculation ;\n"
                "- mutation d’un véhicule d’occasion ;\n"
                "- échange de la carte grise ;\n"
                "- duplicata après perte ou vol.\n\n"
                "Précisez la situation qui vous concerne."
            )

            return {
                **result,
                "classification_time_ms": classification_time_ms,
                "retrieval_time_ms": 0.0,
            }

        retrieval_query = self._build_retrieval_query(
            question=question,
            intent=intent,
        )

        retrieval_started_at = perf_counter()
        retrieved_chunks = self.retriever.search(
            query=retrieval_query,
            domain=domain,
            intent=intent,
            top_k=top_k,
            original_query=question,
        )

        retrieval_time_ms = self._elapsed_ms(
            retrieval_started_at
        )

        intent_matched_chunks = [
            chunk
            for chunk in retrieved_chunks
            if chunk.get("intent_match")
        ]

        if intent_matched_chunks:
            retrieved_chunks = intent_matched_chunks

        relevant_chunks = self._filter_chunks(
            chunks=retrieved_chunks
        )

        if not relevant_chunks:
            result = self._build_no_context_response(
                question=question,
                classification=classification,
                retrieval_query=retrieval_query,
                retrieved_chunks=retrieved_chunks,
            )

            return {
                **result,
                "classification_time_ms": (
                    classification_time_ms
                ),
                "retrieval_time_ms": retrieval_time_ms,
            }

        sources = self._build_sources(
            chunks=relevant_chunks
        )

        return {
            "question": question,
            "retrieval_query": retrieval_query,
            "status": "context_found",
            "domain": domain,
            "intent": intent,
            "confidence": confidence,
            "needs_clarification": False,
            "message": (
                f"{len(relevant_chunks)} extrait(s) "
                "pertinent(s) ont été trouvés."
            ),
            "classification": classification,
            "contexts": relevant_chunks,
            "sources": sources,
            "classification_time_ms": (
                classification_time_ms
            ),
            "retrieval_time_ms": retrieval_time_ms,
        }

    @staticmethod
    def _elapsed_ms(
        started_at: float,
    ) -> float:
        return round(
            (perf_counter() - started_at) * 1000,
            2,
        )

    @staticmethod
    def _contains_case_marker(
        question: str,
    ) -> bool:
        normalized = NarsaRetriever.normalize_for_matching(
            question
        )
        return any(
            marker in normalized
            for marker in CARTE_GRISE_CASE_MARKERS
        )

    @staticmethod
    def _validate_question(question: str) -> str:
        if not isinstance(question, str):
            raise TypeError(
                "La question doit être une chaîne de caractères."
            )

        normalized_question = " ".join(
            question.strip().split()
        )

        if not normalized_question:
            raise ValueError(
                "La question ne peut pas être vide."
            )

        return normalized_question

    @staticmethod
    def _build_retrieval_query(
        question: str,
        intent: str,
    ) -> str:
        hint = INTENT_QUERY_HINTS.get(
            intent,
            f" intention {intent}",
        )

        return f"{question}{hint}"

    def _filter_chunks(
        self,
        chunks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        relevant_chunks: list[dict[str, Any]] = []

        for chunk in chunks:
            similarity_score = float(
                chunk.get(
                    "similarity_score",
                    0.0,
                )
            )

            if (
                similarity_score
                >= self.minimum_similarity_score
            ):
                relevant_chunks.append(chunk)

        return relevant_chunks

    def _build_clarification_response(
        self,
        question: str,
        classification: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "question": question,
            "retrieval_query": None,
            "status": "clarification_required",
            "domain": classification["domain"],
            "intent": classification["intent"],
            "confidence": classification["confidence"],
            "needs_clarification": True,
            "message": (
                "Je ne suis pas suffisamment sûr de votre "
                "demande. Pouvez-vous la reformuler en "
                "précisant votre démarche ?\n\n"
                "Je peux notamment vous renseigner sur :\n\n"
                "- le permis de conduire (perte, vol, "
                "duplicata, renouvellement) ;\n"
                "- le véhicule et la carte grise "
                "(immatriculation, mutation) ;\n"
                "- la visite technique ;\n"
                "- les infractions et le paiement des "
                "amendes ;\n"
                "- les accidents et la prévention routière.\n\n"
                f"{NARSA_CONTACT_HINT}"
            ),
            "classification": classification,
            "contexts": [],
            "sources": [],
        }

    def _build_no_context_response(
        self,
        question: str,
        classification: dict[str, Any],
        retrieval_query: str,
        retrieved_chunks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "question": question,
            "retrieval_query": retrieval_query,
            "status": "no_context_found",
            "domain": classification["domain"],
            "intent": classification["intent"],
            "confidence": classification["confidence"],
            "needs_clarification": False,
            "message": self._build_no_context_message(
                classification["domain"]
            ),
            "classification": classification,
            "contexts": [],
            "sources": [],
            "retrieved_chunks_before_filter": (
                retrieved_chunks
            ),
        }

    def _rescue_low_confidence_question(
        self,
        question: str,
        classification: dict[str, Any],
        top_k: int,
    ) -> dict[str, Any] | None:
        """
        Répond malgré une confiance faible lorsque la recherche
        documentaire trouve un extrait très pertinent.

        Le seuil est volontairement plus élevé que le seuil
        habituel : sans intention fiable, seule une
        correspondance forte autorise une réponse.
        """
        try:
            chunks = self.retriever.search(
                query=question,
                domain=None,
                intent=None,
                top_k=top_k,
                original_query=question,
            )

        except Exception as error:  # pragma: no cover
            print(
                "Recherche de secours impossible :",
                f"{type(error).__name__}: {error}",
            )
            return None

        relevant_chunks = [
            chunk
            for chunk in chunks
            if float(chunk.get("similarity_score", 0.0))
            >= RESCUE_MINIMUM_SIMILARITY_SCORE
        ]

        if not relevant_chunks:
            return None

        best_chunk = relevant_chunks[0]
        domain = (
            best_chunk.get("domain")
            or classification["domain"]
        )

        return {
            "question": question,
            "retrieval_query": question,
            "status": "context_found",
            "domain": domain,
            "intent": classification["intent"],
            "confidence": classification["confidence"],
            "needs_clarification": False,
            "low_confidence_rescue": True,
            "message": (
                f"{len(relevant_chunks)} extrait(s) très "
                "pertinent(s) ont été trouvés malgré une "
                "intention incertaine."
            ),
            "classification": classification,
            "contexts": relevant_chunks,
            "sources": self._build_sources(
                chunks=relevant_chunks
            ),
        }

    @staticmethod
    def _build_no_context_message(
        domain: str,
    ) -> str:
        """Explique l'absence de réponse et oriente l'utilisateur."""
        coverage = format_coverage(domain)

        message = (
            "Votre demande a bien été comprise, mais les "
            "documents officiels dont je dispose ne "
            "couvrent pas ce cas précis. Je préfère ne "
            "rien affirmer plutôt que de vous donner une "
            "information inexacte."
        )

        if coverage:
            message += (
                "\n\nSur ce sujet, je peux vous renseigner "
                f"sur :\n\n{coverage}"
            )

        return f"{message}\n\n{NARSA_CONTACT_HINT}"

    @staticmethod
    def _build_sources(
        chunks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        unique_sources: dict[
            str,
            dict[str, Any],
        ] = {}

        for chunk in chunks:
            source_key = (
                chunk.get("relative_path")
                or chunk.get("filename")
                or f"source-{chunk.get('rank', 0)}"
            )

            similarity_score = float(
                chunk.get(
                    "similarity_score",
                    0.0,
                )
            )

            if source_key not in unique_sources:
                unique_sources[source_key] = {
                    "title": chunk.get("title"),
                    "filename": chunk.get(
                        "filename"
                    ),
                    "relative_path": chunk.get(
                        "relative_path"
                    ),
                    "domain": chunk.get("domain"),
                    "best_similarity_score": (
                        similarity_score
                    ),
                }

            else:
                current_score = float(
                    unique_sources[source_key][
                        "best_similarity_score"
                    ]
                )

                if similarity_score > current_score:
                    unique_sources[source_key][
                        "best_similarity_score"
                    ] = similarity_score

        sources = list(
            unique_sources.values()
        )

        sources.sort(
            key=lambda source: source[
                "best_similarity_score"
            ],
            reverse=True,
        )

        return sources


_rag_service: NarsaRAGService | None = None


def get_rag_service() -> NarsaRAGService:
    global _rag_service

    if _rag_service is None:
        _rag_service = NarsaRAGService()

    return _rag_service


def search_contexts(
    question: str,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    service = get_rag_service()

    return service.search(
        question=question,
        top_k=top_k,
    )


if __name__ == "__main__":
    test_questions = [
        (
            "Quels documents faut-il pour remplacer "
            "un permis de conduire perdu ?"
        ),
        (
            "Comment payer une amende pour excès "
            "de vitesse ?"
        ),
        (
            "Que dois-je faire si je suis témoin "
            "d’un accident avec un blessé ?"
        ),
        (
            "Comment prendre rendez-vous pour une "
            "visite technique ?"
        ),
        "day3 lia permis chno ndir",
    ]

    rag_service = NarsaRAGService()

    for test_question in test_questions:
        print("\n" + "=" * 90)
        print(f"QUESTION : {test_question}")

        try:
            result = rag_service.search(
                question=test_question,
                top_k=3,
            )

            print(
                f"STATUT : {result['status']}"
            )
            print(
                f"DOMAINE : {result['domain']}"
            )
            print(
                f"INTENTION : {result['intent']}"
            )
            print(
                f"CONFIANCE : {result['confidence']}"
            )
            print(
                f"MESSAGE : {result['message']}"
            )

            if result.get("retrieval_query"):
                print(
                    "REQUÊTE DE RECHERCHE : "
                    f"{result['retrieval_query']}"
                )

            if result["contexts"]:
                print("\nCONTEXTES TROUVÉS :")

                for context in result["contexts"]:
                    print("\n" + "-" * 70)
                    print(
                        f"Rang : {context['rank']}"
                    )
                    print(
                        f"Fichier : "
                        f"{context['filename']}"
                    )
                    print(
                        f"Intention correspondante : "
                        f"{context.get('intent_match')}"
                    )
                    print(
                        f"Score sémantique : "
                        f"{context.get('semantic_score')}"
                    )
                    print(
                        f"Bonus intention : "
                        f"{context.get('intent_boost')}"
                    )
                    print(
                        f"Score final : "
                        f"{context['similarity_score']}"
                    )
                    print(
                        f"Titre : "
                        f"{context['title']}"
                    )

            if result["sources"]:
                print("\nSOURCES :")

                for source in result["sources"]:
                    print(
                        f"- {source['filename']} "
                        f"({source['best_similarity_score']})"
                    )

        except Exception as error:
            print(
                f"ERREUR : "
                f"{type(error).__name__}: {error}"
            )
