from __future__ import annotations

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

DEFAULT_TOP_K = 5
DEFAULT_INTENT_BOOST = 0.18


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

    def search(
        self,
        query: str,
        domain: str | None = None,
        intent: str | None = None,
        top_k: int = DEFAULT_TOP_K,
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

        if domain:
            where_filter = {
                "domain": domain
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

            intent_match = bool(
                intent
                and intent in chunk_intents
            )

            final_score = semantic_score

            if intent_match:
                final_score += self.intent_boost

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
                    "similarity_score": round(
                        final_score,
                        4,
                    ),
                }
            )

        candidates.sort(
            key=lambda item: (
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