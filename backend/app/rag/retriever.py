from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[3]

CHROMA_DIR = PROJECT_ROOT / "backend" / "data" / "chroma_db"

COLLECTION_NAME = "narsa_documents"

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

DEFAULT_TOP_K = 3
DEFAULT_INTENT_BOOST = 0.18
DEFAULT_CONTEXT_FACET_BOOST = 0.12

MUTATION_CONTEXT_INTENTS = {
    "mutation_vehicule",
    "changement_proprietaire",
    "vehicule_vendu",
}

MUTATION_QUERY_MARKERS = (
    "mutation",
    "vente",
    "vendu",
    "vendre",
    "cession",
    "changement proprietaire",
    "changer proprietaire",
    "nouveau proprietaire",
    "vendeur",
    "acheteur",
)

VISIT_TECHNICAL_QUERY_MARKERS = (
    "visite technique",
    "controle technique",
)

DOCUMENT_QUERY_MARKERS = (
    "document",
    "documents",
    "piece",
    "pieces",
    "proces verbal",
    "pv",
)

INTENT_DOMAINS = {
    "mutation_vehicule": "vehicule",
    "visite_technique_documents": "visite_technique",
}


@dataclass(frozen=True)
class QueryProfile:
    intents: frozenset[str]
    domains: frozenset[str]
    has_mutation_context: bool
    asks_visit_documents: bool


class NarsaRetriever:
    """
    Recherche les chunks pertinents dans ChromaDB.

    Le classement repose sur :

    1. la similarité sémantique ;
    2. le domaine détecté ;
    3. la correspondance avec l'intention détectée.
    """

    def __init__(
        self,
        intent_boost: float = DEFAULT_INTENT_BOOST,
    ) -> None:
        if not CHROMA_DIR.exists():
            raise FileNotFoundError(
                f"Base ChromaDB introuvable : {CHROMA_DIR}"
            )

        self.intent_boost = intent_boost

        print(
            "Chargement du modèle d'embedding "
            "du retriever..."
        )

        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )

        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR)
        )

        self.collection = self.client.get_collection(
            name=COLLECTION_NAME
        )

        print(
            "Retriever chargé. "
            f"Nombre de chunks : {self.collection.count()}"
        )

    @staticmethod
    def enhance_query(
        query: str,
        domain: str | None = None,
        intent: str | None = None,
    ) -> str:
        """
        Enrichit légèrement la question pour la recherche.

        L'intention est ajoutée sous forme lisible :
        permis_perte devient permis perte.
        """
        enhanced_query = query.strip()

        if intent:
            readable_intent = intent.replace("_", " ")

            enhanced_query += (
                f"\nIntention recherchée : {readable_intent}"
            )

        if domain:
            readable_domain = domain.replace("_", " ")

            enhanced_query += (
                f"\nDomaine recherché : {readable_domain}"
            )

        return enhanced_query

    @staticmethod
    def parse_intents(
        metadata: dict[str, Any],
    ) -> list[str]:
        """
        Transforme la chaîne stockée dans ChromaDB
        en liste d'intentions.

        Exemple :
        'permis_perte, permis_vol'
        devient :
        ['permis_perte', 'permis_vol']
        """
        raw_intents = metadata.get("intents", "")

        if not isinstance(raw_intents, str):
            return []

        return [
            intent.strip()
            for intent in raw_intents.split(",")
            if intent.strip()
        ]

    @staticmethod
    def normalize_for_matching(
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

        return " ".join(
            without_accents.split()
        )

    @classmethod
    def build_query_profile(
        cls,
        query: str,
        domain: str | None,
        intent: str | None,
    ) -> QueryProfile:
        normalized_query = cls.normalize_for_matching(
            query
        )

        has_mutation_context = any(
            marker in normalized_query
            for marker in MUTATION_QUERY_MARKERS
        )

        mentions_visit_technical = any(
            marker in normalized_query
            for marker in VISIT_TECHNICAL_QUERY_MARKERS
        )

        mentions_documents = any(
            marker in normalized_query
            for marker in DOCUMENT_QUERY_MARKERS
        )

        asks_visit_documents = (
            mentions_visit_technical
            and mentions_documents
        )

        query_intents = {
            value
            for value in [intent]
            if value
        }

        query_domains = {
            value
            for value in [domain]
            if value
        }

        if asks_visit_documents:
            query_intents.add(
                "visite_technique_documents"
            )
            query_domains.add(
                INTENT_DOMAINS[
                    "visite_technique_documents"
                ]
            )

        if has_mutation_context:
            query_intents.add(
                "mutation_vehicule"
            )
            query_domains.add(
                INTENT_DOMAINS[
                    "mutation_vehicule"
                ]
            )

        return QueryProfile(
            intents=frozenset(query_intents),
            domains=frozenset(query_domains),
            has_mutation_context=has_mutation_context,
            asks_visit_documents=asks_visit_documents,
        )

    @staticmethod
    def is_candidate_compatible(
        profile: QueryProfile,
        chunk_intents: list[str],
    ) -> bool:
        """
        Vérifie que le chunk couvre le contexte réel de la question.
        """
        if not profile.asks_visit_documents:
            return True

        chunk_intent_set = set(chunk_intents)

        has_mutation_metadata = bool(
            MUTATION_CONTEXT_INTENTS
            .intersection(chunk_intent_set)
        )

        if not profile.has_mutation_context:
            return (
                "visite_technique_documents"
                in chunk_intent_set
                and not has_mutation_metadata
            )

        return bool(
            profile.intents.intersection(
                chunk_intent_set
            )
        )

    def search(
        self,
        query: str,
        domain: str | None = None,
        intent: str | None = None,
        top_k: int = DEFAULT_TOP_K,
        original_query: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Recherche les chunks puis les reclasse selon :

        - leur similarité sémantique ;
        - leur correspondance avec l'intention.
        """
        query = query.strip()

        if not query:
            return []

        top_k = max(1, top_k)

        profile = self.build_query_profile(
            query=(
                original_query
                if original_query is not None
                else query
            ),
            domain=domain,
            intent=intent,
        )

        enhanced_query = self.enhance_query(
            query=query,
            domain=domain,
            intent=intent,
        )

        query_embedding = self.embedding_model.encode(
            enhanced_query,
            normalize_embeddings=True,
        ).tolist()

        where_filter = None

        if len(profile.domains) == 1:
            where_filter = {
                "domain": next(
                    iter(profile.domains)
                )
            }

        elif len(profile.domains) > 1:
            where_filter = {
                "domain": {
                    "$in": sorted(profile.domains)
                }
            }

        # On récupère plus de candidats que nécessaire,
        # puis on les reclasse avec l'intention.
        candidate_count = max(
            top_k * 4,
            12,
        )

        candidate_count = min(
            candidate_count,
            self.collection.count(),
        )

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=candidate_count,
            where=where_filter,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        documents = results.get(
            "documents",
            [[]],
        )[0]

        metadatas = results.get(
            "metadatas",
            [[]],
        )[0]

        distances = results.get(
            "distances",
            [[]],
        )[0]

        candidates: list[dict[str, Any]] = []

        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances,
        ):
            distance_value = float(distance)

            semantic_score = 1.0 - distance_value

            semantic_score = max(
                0.0,
                min(
                    1.0,
                    semantic_score,
                ),
            )

            chunk_intents = self.parse_intents(
                metadata
            )

            if not self.is_candidate_compatible(
                profile=profile,
                chunk_intents=chunk_intents,
            ):
                continue

            matched_query_intents = sorted(
                profile.intents.intersection(
                    chunk_intents
                )
            )

            context_coverage = (
                len(matched_query_intents)
                / len(profile.intents)
                if profile.intents
                else 0.0
            )

            intent_match = bool(
                intent
                and intent in chunk_intents
            )

            final_score = semantic_score

            if intent_match:
                final_score += self.intent_boost

            context_facet_boost = (
                DEFAULT_CONTEXT_FACET_BOOST
                * max(
                    0,
                    len(matched_query_intents) - 1,
                )
            )

            final_score += context_facet_boost

            final_score = max(
                0.0,
                min(
                    1.0,
                    final_score,
                ),
            )

            candidates.append(
                {
                    "content": document,
                    "domain": metadata.get(
                        "domain"
                    ),
                    "title": metadata.get(
                        "title"
                    ),
                    "filename": metadata.get(
                        "filename"
                    ),
                    "relative_path": metadata.get(
                        "relative_path"
                    ),
                    "chunk_index": metadata.get(
                        "chunk_index"
                    ),
                    "intents": chunk_intents,
                    "intent_match": intent_match,
                    "query_intents": sorted(
                        profile.intents
                    ),
                    "query_domains": sorted(
                        profile.domains
                    ),
                    "matched_query_intents": (
                        matched_query_intents
                    ),
                    "context_coverage": round(
                        context_coverage,
                        4,
                    ),
                    "distance": round(
                        distance_value,
                        4,
                    ),
                    "semantic_score": round(
                        semantic_score,
                        4,
                    ),
                    "intent_boost": (
                        self.intent_boost
                        if intent_match
                        else 0.0
                    ),
                    "context_facet_boost": round(
                        context_facet_boost,
                        4,
                    ),
                    "similarity_score": round(
                        final_score,
                        4,
                    ),
                }
            )

        candidates.sort(
            key=lambda item: (
                item["context_coverage"],
                item["intent_match"],
                item["similarity_score"],
                item["semantic_score"],
            ),
            reverse=True,
        )

        selected_chunks = candidates[:top_k]

        for rank, chunk in enumerate(
            selected_chunks,
            start=1,
        ):
            chunk["rank"] = rank

        return selected_chunks


if __name__ == "__main__":
    retriever = NarsaRetriever()

    test_question = (
        "Quels documents faut-il pour remplacer "
        "un permis de conduire perdu ?"
    )

    results = retriever.search(
        query=test_question,
        domain="permis",
        intent="permis_perte",
        top_k=3,
    )

    print(f"\nQuestion : {test_question}")
    print("Domaine : permis")
    print("Intention : permis_perte")

    for result in results:
        print("\n" + "=" * 70)
        print(f"Rang : {result['rank']}")
        print(f"Domaine : {result['domain']}")
        print(f"Titre : {result['title']}")
        print(f"Fichier : {result['filename']}")
        print(f"Intentions : {result['intents']}")
        print(
            "Correspondance intention : "
            f"{result['intent_match']}"
        )
        print(
            "Score sémantique : "
            f"{result['semantic_score']}"
        )
        print(
            "Bonus intention : "
            f"{result['intent_boost']}"
        )
        print(
            "Score final : "
            f"{result['similarity_score']}"
        )
        print("\nContenu :")
        print(result["content"])
