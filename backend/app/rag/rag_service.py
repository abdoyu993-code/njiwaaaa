from __future__ import annotations

from typing import Any

from backend.app.classifier.classifier_service import (
    predict_intent,
)
from backend.app.rag.retriever import NarsaRetriever


DEFAULT_CLARIFICATION_THRESHOLD = 0.60
DEFAULT_MINIMUM_SIMILARITY_SCORE = 0.30
DEFAULT_TOP_K = 5


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

        classification = predict_intent(
            text=question,
            top_k=3,
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
            return self._build_clarification_response(
                question=question,
                classification=classification,
            )

        retrieval_query = self._build_retrieval_query(
            question=question,
            intent=intent,
        )

        retrieved_chunks = self.retriever.search(
            query=retrieval_query,
            domain=domain,
            intent=intent,
            top_k=top_k,
        )

        relevant_chunks = self._filter_chunks(
            chunks=retrieved_chunks
        )

        if not relevant_chunks:
            return self._build_no_context_response(
                question=question,
                classification=classification,
                retrieval_query=retrieval_query,
                retrieved_chunks=retrieved_chunks,
            )

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
        }

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
                "demande. Pouvez-vous la reformuler avec "
                "plus de détails ?"
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
            "message": (
                "La demande a été comprise, mais aucun "
                "document suffisamment pertinent n’a été trouvé."
            ),
            "classification": classification,
            "contexts": [],
            "sources": [],
            "retrieved_chunks_before_filter": (
                retrieved_chunks
            ),
        }

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