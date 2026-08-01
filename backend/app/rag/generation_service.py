from __future__ import annotations

import os
import time
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from backend.app.rag.rag_service import (
    NarsaRAGService,
    get_rag_service,
)


load_dotenv()


DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
DEFAULT_MAX_CONTEXTS = 3
DEFAULT_TOP_K = 3

DEFAULT_MAX_RETRIES = 1
DEFAULT_RETRY_BASE_DELAY = 1.0
DEFAULT_RETRY_MAX_DELAY = 1.0

RETRYABLE_STATUS_CODES = {
    429,
    500,
    502,
    503,
    504,
}


class NarsaGenerationService:
    """
    Génération finale des réponses de l'assistant NARSA.

    Pipeline :
    1. classification ;
    2. recherche documentaire ;
    3. construction du contexte ;
    4. génération Gemini ;
    5. nouvelle tentative en cas d'erreur temporaire ;
    6. retour de la réponse et des sources.
    """

    def __init__(
        self,
        rag_service: NarsaRAGService | None = None,
        model_name: str | None = None,
        max_contexts: int = DEFAULT_MAX_CONTEXTS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
        retry_max_delay: float = DEFAULT_RETRY_MAX_DELAY,
    ) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY est absente du fichier .env"
            )

        self.model_name = (
            model_name
            or os.getenv(
                "GEMINI_MODEL",
                DEFAULT_GEMINI_MODEL,
            )
        )

        self.max_contexts = min(
            DEFAULT_MAX_CONTEXTS,
            max(
                1,
                max_contexts,
            ),
        )

        self.max_retries = min(
            DEFAULT_MAX_RETRIES,
            max(
                0,
                max_retries,
            ),
        )

        self.retry_base_delay = max(
            0.1,
            retry_base_delay,
        )

        self.retry_max_delay = max(
            self.retry_base_delay,
            retry_max_delay,
        )

        self.rag_service = (
            rag_service
            or get_rag_service()
        )

        self.client = genai.Client(
            api_key=self.api_key
        )

        print("Service de génération Gemini chargé.")
        print(f"Modèle utilisé : {self.model_name}")
        print(
            "Nombre maximal de nouvelles tentatives : "
            f"{self.max_retries}"
        )

    def answer(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> dict[str, Any]:
        """
        Produit une réponse finale à partir des documents.
        """
        cleaned_question = question.strip()

        if len(cleaned_question) < 2:
            raise ValueError(
                "La question doit contenir au moins 2 caractères."
            )

        rag_result = self.rag_service.search(
            question=cleaned_question,
            top_k=top_k,
        )

        status = rag_result["status"]

        if status == "clarification_required":
            return {
                **rag_result,
                "answer": rag_result["message"],
                "generated": False,
                "model": None,
                "finish_reason": None,
                "generation_attempts": 0,
            }

        if status == "no_context_found":
            return {
                **rag_result,
                "answer": (
                    "Je n’ai pas trouvé suffisamment "
                    "d’informations fiables dans les documents "
                    "disponibles pour répondre à cette question."
                ),
                "generated": False,
                "model": None,
                "finish_reason": None,
                "generation_attempts": 0,
            }

        contexts = rag_result["contexts"][
            : self.max_contexts
        ]

        context_text = self._build_context(
            contexts=contexts
        )

        prompt = self._build_prompt(
            question=cleaned_question,
            domain=rag_result["domain"],
            intent=rag_result["intent"],
            context_text=context_text,
        )

        try:
            response, attempts = (
                self._generate_with_retry(
                    prompt=prompt
                )
            )

        except Exception as error:
            return {
                **rag_result,
                "answer": self._build_extractive_fallback(
                    contexts=contexts
                ),
                "generated": False,
                "model": self.model_name,
                "finish_reason": None,
                "generation_attempts": (
                    self.max_retries + 1
                ),
                "used_contexts_count": len(contexts),
                "generation_error": (
                    f"{type(error).__name__}: {error}"
                ),
            }

        finish_reason = self._extract_finish_reason(
            response=response
        )

        print(
            "Raison de fin Gemini :",
            finish_reason,
        )

        answer_text = self._extract_response_text(
            response=response
        )

        if not answer_text:
            return {
                **rag_result,
                "answer": (
                    "Gemini n’a retourné aucune "
                    "réponse exploitable."
                ),
                "generated": False,
                "model": self.model_name,
                "finish_reason": finish_reason,
                "generation_attempts": attempts,
                "used_contexts_count": len(contexts),
            }

        return {
            **rag_result,
            "answer": answer_text,
            "generated": True,
            "model": self.model_name,
            "finish_reason": finish_reason,
            "generation_attempts": attempts,
            "used_contexts_count": len(contexts),
        }

    def _generate_with_retry(
        self,
        prompt: str,
    ) -> tuple[Any, int]:
        """
        Appelle Gemini une première fois, puis effectue au plus
        une nouvelle tentative après une courte attente.
        """
        total_attempts = (
            self.max_retries + 1
        )

        last_error: Exception | None = None

        for attempt in range(
            1,
            total_attempts + 1,
        ):
            try:
                print(
                    "Appel Gemini "
                    f"{attempt}/{total_attempts}..."
                )

                response = (
                    self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=1.0,
                            max_output_tokens=1200,
                            thinking_config=(
                                types.ThinkingConfig(
                                    thinking_level="low",
                                )
                            ),
                        ),
                    )
                )

                return response, attempt

            except Exception as error:
                last_error = error

                status_code = (
                    self._get_status_code(
                        error=error
                    )
                )

                retryable = (
                    self._is_retryable_error(
                        error=error,
                        status_code=status_code,
                    )
                )

                print(
                    "Erreur Gemini : "
                    f"{type(error).__name__}"
                )
                print(
                    "Code HTTP détecté : "
                    f"{status_code}"
                )

                if (
                    not retryable
                    or attempt >= total_attempts
                ):
                    raise

                delay = self._calculate_retry_delay(
                    attempt=attempt
                )

                print(
                    "Erreur temporaire. "
                    f"Nouvelle tentative dans "
                    f"{delay:.1f} seconde(s)..."
                )

                time.sleep(delay)

        if last_error is not None:
            raise last_error

        raise RuntimeError(
            "Échec de génération sans erreur identifiable."
        )

    def _calculate_retry_delay(
        self,
        attempt: int,
    ) -> float:
        """
        Calcule une attente courte avant l'unique retry.
        """
        exponential_delay = (
            self.retry_base_delay
            * (2 ** (attempt - 1))
        )

        limited_delay = min(
            exponential_delay,
            self.retry_max_delay,
        )

        return limited_delay

    @staticmethod
    def _get_status_code(
        error: Exception,
    ) -> int | None:
        """
        Extrait le code HTTP d'une erreur google-genai.
        """
        status_code = getattr(
            error,
            "code",
            None,
        )

        if status_code is None:
            status_code = getattr(
                error,
                "status_code",
                None,
            )

        if isinstance(
            status_code,
            int,
        ):
            return status_code

        try:
            if status_code is not None:
                return int(status_code)
        except (
            TypeError,
            ValueError,
        ):
            pass

        return None

    @staticmethod
    def _is_retryable_error(
        error: Exception,
        status_code: int | None,
    ) -> bool:
        """
        Détermine si l'erreur est temporaire.
        """
        if (
            status_code
            in RETRYABLE_STATUS_CODES
        ):
            return True

        error_text = str(error).upper()

        retryable_markers = (
            "RESOURCE_EXHAUSTED",
            "UNAVAILABLE",
            "INTERNAL",
            "DEADLINE_EXCEEDED",
            "TOO MANY REQUESTS",
            "HIGH DEMAND",
            "TIMED OUT",
            "TIMEOUT",
        )

        if any(
            marker in error_text
            for marker in retryable_markers
        ):
            return True

        if isinstance(
            error,
            errors.ServerError,
        ):
            return True

        return False

    @staticmethod
    def _extract_finish_reason(
        response: Any,
    ) -> str | None:
        candidates = getattr(
            response,
            "candidates",
            None,
        )

        if not candidates:
            return None

        finish_reason = getattr(
            candidates[0],
            "finish_reason",
            None,
        )

        if finish_reason is None:
            return None

        return str(finish_reason)

    @staticmethod
    def _extract_response_text(
        response: Any,
    ) -> str:
        response_text = getattr(
            response,
            "text",
            None,
        )

        if not response_text:
            return ""

        return response_text.strip()

    @staticmethod
    def _build_context(
        contexts: list[dict[str, Any]],
    ) -> str:
        """
        Formate les chunks transmis à Gemini.
        """
        formatted_contexts: list[str] = []

        for index, context in enumerate(
            contexts,
            start=1,
        ):
            title = (
                context.get("title")
                or "Document sans titre"
            )

            filename = (
                context.get("filename")
                or "source inconnue"
            )

            relative_path = (
                context.get("relative_path")
                or filename
            )

            content = (
                context.get("content")
                or ""
            ).strip()

            formatted_contexts.append(
                (
                    f"[SOURCE {index}]\n"
                    f"Titre : {title}\n"
                    f"Fichier : {filename}\n"
                    f"Chemin : {relative_path}\n"
                    f"Contenu :\n{content}"
                )
            )

        return "\n\n".join(
            formatted_contexts
        )

    @staticmethod
    def _build_extractive_fallback(
        contexts: list[dict[str, Any]],
    ) -> str:
        """
        Retourne uniquement les extraits récupérés, sans génération.
        """
        extracted_sections: list[str] = []

        for context in contexts:
            content = (
                context.get("content")
                or ""
            ).strip()

            if not content:
                continue

            title = (
                context.get("title")
                or context.get("filename")
                or "Document NARSA"
            )

            extracted_sections.append(
                f"### {title}\n\n{content}"
            )

        if not extracted_sections:
            return (
                "Le service de génération est temporairement "
                "indisponible et aucun extrait documentaire "
                "exploitable n’est disponible."
            )

        return (
            "Le service de génération est temporairement "
            "indisponible. Voici uniquement les extraits "
            "documentaires récupérés, sans ajout d’information :"
            "\n\n"
            + "\n\n".join(extracted_sections)
        )

    @staticmethod
    def _build_prompt(
        question: str,
        domain: str,
        intent: str,
        context_text: str,
    ) -> str:
        """
        Construit le prompt RAG envoyé à Gemini.
        """
        return f"""
Tu es un assistant d'information spécialisé dans les services de la NARSA au Maroc.

Ta réponse doit être strictement fondée sur le contexte documentaire fourni.

Règles obligatoires :

1. Réponds en français clair, naturel et simple.
2. Réponds directement à la question.
3. N'invente aucune information.
4. N'utilise pas tes connaissances générales pour compléter le contexte.
5. Si une information demandée n'apparaît pas dans le contexte, indique qu'elle n'est pas disponible dans les documents fournis.
6. Présente les démarches dans leur ordre logique.
7. Utilise une liste numérotée pour une procédure.
8. Utilise une liste à puces pour les documents nécessaires.
9. Mentionne les délais, montants et conditions uniquement lorsqu'ils apparaissent dans le contexte.
10. Évite les introductions longues.
11. Ne mentionne jamais ChromaDB, les embeddings, les scores, le classificateur ou le fonctionnement technique du RAG.
12. Ne dis pas que tu es l'assistant officiel de la NARSA.
13. Ne copie pas les métadonnées techniques.
14. Termine par une section intitulée exactement "Sources consultées".
15. Cite uniquement les titres ou fichiers présents dans le contexte.
16. Ne coupe pas la réponse au milieu d'une phrase.
17. Produis une réponse complète mais concise.

Domaine détecté :
{domain}

Intention détectée :
{intent}

Question de l'utilisateur :
{question}

Contexte documentaire :
{context_text}

Réponse finale :
""".strip()


_generation_service: NarsaGenerationService | None = None


def get_generation_service() -> NarsaGenerationService:
    """
    Retourne une instance unique du service.
    """
    global _generation_service

    if _generation_service is None:
        _generation_service = (
            NarsaGenerationService()
        )

    return _generation_service


def generate_answer(
    question: str,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """
    Fonction utilisée par FastAPI.
    """
    service = get_generation_service()

    return service.answer(
        question=question,
        top_k=top_k,
    )


if __name__ == "__main__":
    test_questions = [
        (
            "Comment payer une amende "
            "pour excès de vitesse ?"
        ),
        (
            "Comment prendre rendez-vous "
            "pour une visite technique ?"
        ),
        "day3 lia permis chno ndir",
    ]

    generation_service = (
        NarsaGenerationService()
    )

    for test_question in test_questions:
        print("\n" + "=" * 90)
        print(
            f"QUESTION : {test_question}"
        )

        result = generation_service.answer(
            question=test_question,
            top_k=4,
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
            "GÉNÉRÉ PAR GEMINI : "
            f"{result['generated']}"
        )
        print(
            "TENTATIVES GEMINI : "
            f"{result.get('generation_attempts', 0)}"
        )

        if result.get("finish_reason"):
            print(
                "RAISON DE FIN : "
                f"{result['finish_reason']}"
            )

        print("\nRÉPONSE :")
        print(
            result["answer"]
        )

        if result.get(
            "generation_error"
        ):
            print(
                "\nERREUR DE GÉNÉRATION :"
            )
            print(
                result["generation_error"]
            )
