from __future__ import annotations

import os
import re
import time
import unicodedata
from time import perf_counter
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
DEFAULT_MAX_CONTEXTS = 2
DEFAULT_TOP_K = 3

DEFAULT_EXTRACTIVE_SCORE_THRESHOLD = 0.88
DEFAULT_EXTRACTIVE_MAX_CONTENT_CHARS = 2200
DEFAULT_EXTRACTIVE_MAX_POINTS = 7

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


class IncompleteGenerationError(RuntimeError):
    """Gemini a retourné une réponse vide ou tronquée."""


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
        extractive_score_threshold: float = (
            DEFAULT_EXTRACTIVE_SCORE_THRESHOLD
        ),
        extractive_max_content_chars: int = (
            DEFAULT_EXTRACTIVE_MAX_CONTENT_CHARS
        ),
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

        self.extractive_score_threshold = max(
            0.0,
            min(
                1.0,
                extractive_score_threshold,
            ),
        )

        self.extractive_max_content_chars = max(
            200,
            extractive_max_content_chars,
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
        total_started_at = perf_counter()
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
                "generation_time_ms": 0.0,
                "total_time_ms": self._elapsed_ms(
                    total_started_at
                ),
                "extractive": False,
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
                "generation_time_ms": 0.0,
                "total_time_ms": self._elapsed_ms(
                    total_started_at
                ),
                "extractive": False,
            }

        contexts = self._select_distinct_contexts(
            contexts=rag_result["contexts"],
            limit=self.max_contexts,
        )

        print(
            "Sources RAG sélectionnées :",
            [
                context.get("filename")
                or context.get("title")
                or "source inconnue"
                for context in contexts
            ],
        )

        if self._can_answer_extractively(
            rag_result=rag_result,
            contexts=contexts,
        ):
            extractive_answer = self._build_concise_extractive_answer(
                question=cleaned_question,
                content=self._content_for_extraction(
                    contexts[0]
                ),
            )

            if (
                extractive_answer
                and self._is_complete_answer(
                    extractive_answer,
                    finish_reason=None,
                )
            ):
                return {
                    **rag_result,
                    "answer": self._sanitize_answer(
                        extractive_answer
                    ),
                    "generated": False,
                    "model": None,
                    "finish_reason": None,
                    "generation_attempts": 0,
                    "used_contexts_count": 1,
                    "generation_time_ms": 0.0,
                    "total_time_ms": self._elapsed_ms(
                        total_started_at
                    ),
                    "extractive": True,
                }

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
            generation_started_at = perf_counter()
            response, attempts = (
                self._generate_with_retry(
                    prompt=prompt
                )
            )
            generation_time_ms = self._elapsed_ms(
                generation_started_at
            )

        except Exception as error:
            generation_time_ms = self._elapsed_ms(
                generation_started_at
            )

            return {
                **rag_result,
                "answer": self._sanitize_answer(
                    self._build_extractive_fallback(
                        contexts=contexts,
                        question=cleaned_question,
                    )
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
                "generation_time_ms": generation_time_ms,
                "total_time_ms": self._elapsed_ms(
                    total_started_at
                ),
                "extractive": False,
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
        answer_text = self._sanitize_answer(
            answer_text
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
                "generation_time_ms": generation_time_ms,
                "total_time_ms": self._elapsed_ms(
                    total_started_at
                ),
                "extractive": False,
            }

        return {
            **rag_result,
            "answer": answer_text,
            "generated": True,
            "model": self.model_name,
            "finish_reason": finish_reason,
            "generation_attempts": attempts,
            "used_contexts_count": len(contexts),
            "generation_time_ms": generation_time_ms,
            "total_time_ms": self._elapsed_ms(
                total_started_at
            ),
            "extractive": False,
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
                        contents=(
                            prompt
                            if attempt == 1
                            else (
                                f"{prompt}\n\n"
                                "La réponse précédente était incomplète. "
                                "Réponds immédiatement en 3 à 7 phrases "
                                "ou étapes complètes, sans introduction."
                            )
                        ),
                        config=types.GenerateContentConfig(
                            temperature=0.3,
                            max_output_tokens=600,
                            thinking_config=(
                                types.ThinkingConfig(
                                    thinking_level="low",
                                )
                            ),
                        ),
                    )
                )

                finish_reason = self._extract_finish_reason(
                    response
                )
                response_text = self._sanitize_answer(
                    self._extract_response_text(response)
                )

                if not self._is_complete_answer(
                    response_text,
                    finish_reason=finish_reason,
                ):
                    raise IncompleteGenerationError(
                        "Réponse Gemini incomplète "
                        f"({finish_reason or 'raison inconnue'})."
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
                    or isinstance(
                        error,
                        IncompleteGenerationError,
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
    def _elapsed_ms(
        started_at: float,
    ) -> float:
        return round(
            (perf_counter() - started_at) * 1000,
            2,
        )

    @staticmethod
    def _source_key(
        context: dict[str, Any],
    ) -> str:
        return str(
            context.get("relative_path")
            or context.get("filename")
            or context.get("title")
            or f"chunk-{context.get('rank', 0)}"
        )

    @classmethod
    def _select_distinct_contexts(
        cls,
        contexts: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        selected_by_source: dict[str, dict[str, Any]] = {}

        for context in contexts:
            source_key = cls._source_key(context)

            if source_key in selected_by_source:
                selected_context = selected_by_source[source_key]
                existing_content = str(
                    selected_context.get("content") or ""
                ).strip()
                additional_content = str(
                    context.get("content") or ""
                ).strip()

                if (
                    additional_content
                    and additional_content not in existing_content
                ):
                    selected_context["content"] = (
                        f"{existing_content}\n\n{additional_content}"
                        if existing_content
                        else additional_content
                    )
                    selected_context["merged_chunks_count"] = (
                        int(
                            selected_context.get(
                                "merged_chunks_count",
                                1,
                            )
                        )
                        + 1
                    )
                continue

            if len(selected) >= limit:
                continue

            selected_context = dict(context)
            selected_context["merged_chunks_count"] = 1
            selected_by_source[source_key] = selected_context
            selected.append(selected_context)

        return selected

    def _can_answer_extractively(
        self,
        rag_result: dict[str, Any],
        contexts: list[dict[str, Any]],
    ) -> bool:
        if len(contexts) != 1:
            return False

        if len(rag_result.get("sources", [])) != 1:
            return False

        best_context = contexts[0]
        score = float(
            best_context.get(
                "similarity_score",
                0.0,
            )
        )

        if score < self.extractive_score_threshold:
            return False

        if not best_context.get("intent_match", False):
            return False

        context_coverage = float(
            best_context.get(
                "context_coverage",
                1.0,
            )
        )

        if context_coverage < 0.8:
            return False

        content = str(
            best_context.get("content")
            or ""
        ).strip()

        return bool(
            content
            and len(content)
            <= self.extractive_max_content_chars
        )

    @staticmethod
    def _content_for_extraction(
        context: dict[str, Any],
    ) -> str:
        return str(
            context.get("document_content")
            or context.get("content")
            or ""
        ).strip()

    @classmethod
    def _build_concise_extractive_answer(
        cls,
        question: str,
        content: str,
    ) -> str:
        if cls._asks_for_procedure(question):
            procedure_answer = (
                cls._build_document_procedure(
                    question=question,
                    content=content,
                )
            )

            if procedure_answer:
                return procedure_answer

        relevant_content = cls._select_relevant_section(
            question=question,
            content=content,
        )
        units = cls._extract_content_units(
            relevant_content
        )

        if not units:
            return ""

        asks_for_procedure = cls._asks_for_procedure(
            question
        )

        question_terms = cls._meaningful_terms(
            question
        )
        scored_units: list[tuple[int, int, str]] = []

        for index, unit in enumerate(units):
            overlap = len(
                question_terms.intersection(
                    cls._meaningful_terms(unit)
                )
            )
            scored_units.append(
                (overlap, index, unit)
            )

        best_overlap, anchor_index, _ = max(
            scored_units,
            key=lambda item: (
                item[0],
                -item[1],
            ),
        )

        if best_overlap == 0:
            return ""

        is_numbered_procedure = bool(
            re.search(
                r"(?m)^\s*\d+[.)]\s+",
                relevant_content,
            )
        )

        if is_numbered_procedure:
            selected_units = units[
                :DEFAULT_EXTRACTIVE_MAX_POINTS
            ]
        else:
            selected_units = []

            for unit in units[
                anchor_index:
                anchor_index + DEFAULT_EXTRACTIVE_MAX_POINTS
            ]:
                normalized_unit = cls._normalize_text(
                    unit
                )

                if any(
                    normalized_unit
                    == cls._normalize_text(existing)
                    for existing in selected_units
                ):
                    continue

                selected_units.append(unit)

        if not selected_units:
            return ""

        if asks_for_procedure and len(selected_units) < 3:
            return ""

        verified_urls = cls._extract_urls(content)

        if len(selected_units) == 1:
            answer = selected_units[0]

        elif is_numbered_procedure:
            answer = "\n".join(
                f"{index}. {unit}"
                for index, unit in enumerate(
                    selected_units,
                    start=1,
                )
            )

        else:
            answer = "\n".join(
                f"- {unit}"
                for unit in selected_units
            )

        if verified_urls:
            answer += (
                "\n\nLien indiqué dans le document : "
                f"{verified_urls[0]}"
            )

        return answer

    @classmethod
    def _build_document_procedure(
        cls,
        question: str,
        content: str,
    ) -> str:
        """Construit une procédure ordonnée à partir des sections du document."""
        sections = cls._split_markdown_sections(content)

        if not sections:
            return ""

        normalized_question = cls._normalize_text(question)
        normalized_content = cls._normalize_text(content)
        intro = ""

        for heading, body in sections:
            normalized_heading = cls._normalize_text(heading)

            if normalized_heading in {"principe", "description"}:
                intro_units = cls._extract_content_units(body)

                if intro_units:
                    intro = intro_units[0]
                    break

        steps: list[str] = []
        has_documents_section = (
            "documents necessaires" in normalized_content
        )

        if has_documents_section:
            preparation = "Préparez les documents nécessaires"
            loss_or_theft_context = any(
                marker in normalized_question
                for marker in ("perdu", "perte", "vol")
            )

            if (
                loss_or_theft_context
                and "declaration de perte ou de vol"
                in normalized_content
            ):
                preparation += (
                    " et la déclaration de perte ou de vol"
                )

            steps.append(preparation + ".")

        deposit_step = ""

        for heading, body in sections:
            normalized_heading = cls._normalize_text(heading)

            if "lieu de depot" in normalized_heading:
                deposit_units = cls._extract_content_units(body)

                if deposit_units:
                    deposit_step = deposit_units[0]
                    break

        if not deposit_step:
            for _, body in sections:
                for unit in cls._extract_content_units(body):
                    normalized_unit = cls._normalize_text(unit)

                    if (
                        "depose" in normalized_unit
                        or "depot" in normalized_unit
                    ):
                        deposit_step = unit
                        break

                if deposit_step:
                    break

        if deposit_step:
            steps.append(deposit_step)

        process_candidates: list[tuple[int, list[str]]] = []

        for heading, body in sections:
            normalized_heading = cls._normalize_text(heading)

            if not any(
                marker in normalized_heading
                for marker in (
                    "demarche",
                    "etapes",
                    "processus",
                    "traitement",
                )
            ):
                continue

            raw_numbered_steps = re.findall(
                r"(?m)^\s*\d+[.)]\s+(.+?)\s*$",
                body,
            )
            numbered_steps: list[str] = []

            for raw_step in raw_numbered_steps:
                extracted = cls._extract_content_units(raw_step)

                if extracted:
                    numbered_steps.append(extracted[0])

            if numbered_steps:
                process_candidates.append(
                    (len(numbered_steps), numbered_steps)
                )

        process_steps = (
            max(process_candidates, key=lambda item: item[0])[1]
            if process_candidates
            else []
        )

        for process_step in process_steps:
            normalized_step = cls._normalize_text(process_step)

            if (
                deposit_step
                and "reception" in normalized_step
                and any(
                    marker in normalized_step
                    for marker in ("demande", "dossier")
                )
            ):
                continue

            if not any(
                cls._normalize_text(existing)
                == normalized_step
                for existing in steps
            ):
                steps.append(process_step)

        if len(steps) > DEFAULT_EXTRACTIVE_MAX_POINTS:
            steps = (
                steps[:DEFAULT_EXTRACTIVE_MAX_POINTS - 1]
                + [steps[-1]]
            )

        if len(steps) < 3:
            return ""

        answer_parts: list[str] = []

        if intro:
            answer_parts.append(intro)

        answer_parts.append(
            "\n".join(
                f"{index}. {step}"
                for index, step in enumerate(steps, start=1)
            )
        )

        verified_urls = cls._extract_urls(content)

        if verified_urls:
            url = verified_urls[0]
            link_label = (
                "Suivi du dossier"
                if "suivi" in cls._normalize_text(url)
                else "Lien officiel"
            )
            answer_parts.append(f"{link_label} : {url}")

        return "\n\n".join(answer_parts)

    @staticmethod
    def _split_markdown_sections(
        content: str,
    ) -> list[tuple[str, str]]:
        heading_pattern = re.compile(
            r"(?m)^#{2,6}\s+(.+?)\s*$"
        )
        matches = list(heading_pattern.finditer(content))
        sections: list[tuple[str, str]] = []

        for index, match in enumerate(matches):
            section_end = (
                matches[index + 1].start()
                if index + 1 < len(matches)
                else len(content)
            )
            body = content[match.end():section_end].strip()

            if body:
                sections.append((match.group(1).strip(), body))

        return sections

    @classmethod
    def _select_relevant_section(
        cls,
        question: str,
        content: str,
    ) -> str:
        heading_pattern = re.compile(
            r"(?m)^#{2,6}\s+(.+?)\s*$"
        )
        matches = list(
            heading_pattern.finditer(content)
        )

        if not matches:
            return content

        question_terms = cls._meaningful_terms(
            question
        )
        normalized_question = cls._normalize_text(
            question
        )
        asks_for_procedure = cls._asks_for_procedure(
            normalized_question
        )
        sections: list[tuple[int, str]] = []

        for index, match in enumerate(matches):
            section_end = (
                matches[index + 1].start()
                if index + 1 < len(matches)
                else len(content)
            )
            heading = match.group(1).strip()
            body = content[
                match.end():section_end
            ].strip()

            if not body:
                continue

            section_terms = cls._meaningful_terms(
                f"{heading} {body[:500]}"
            )
            score = len(
                question_terms.intersection(
                    section_terms
                )
            )
            normalized_heading = cls._normalize_text(
                heading
            )

            if (
                asks_for_procedure
                and any(
                    marker in normalized_heading
                    for marker in (
                        "demarche",
                        "etapes",
                        "processus",
                        "traitement",
                        "lieu de depot",
                    )
                )
            ):
                score += 3

            sections.append(
                (score, body)
            )

        if not sections:
            return content

        return max(
            sections,
            key=lambda item: item[0],
        )[1]

    @classmethod
    def _asks_for_procedure(
        cls,
        question: str,
    ) -> bool:
        normalized_question = cls._normalize_text(question)
        return any(
            marker in normalized_question
            for marker in (
                "comment",
                "que dois je faire",
                "que dois-je faire",
                "demarche",
                "procedure",
                "rendez vous",
            )
        )

    @staticmethod
    def _extract_urls(
        content: str,
    ) -> list[str]:
        return list(
            dict.fromkeys(
                re.findall(
                    r"https?://[^\s)\]>]+",
                    content,
                )
            )
        )

    @classmethod
    def _extract_content_units(
        cls,
        content: str,
    ) -> list[str]:
        raw_units = re.split(
            r"(?:\r?\n)+|(?<=[.!?])\s+",
            content,
        )
        units: list[str] = []
        ignored_prefixes = (
            "domaine :",
            "intentions concernees",
            "organisme :",
            "source :",
            "langue :",
            "pays :",
            "url :",
            "references",
            "metadonnees",
        )

        for raw_unit in raw_units:
            raw_unit = raw_unit.strip()

            if raw_unit.endswith(":"):
                continue

            unit = re.sub(
                r"^\s*(?:#{1,6}\s*|[-*]\s*|\d+[.)]\s*)",
                "",
                raw_unit,
            ).strip(" \t\r\n;:-")

            if not (20 <= len(unit) <= 360):
                continue

            normalized = cls._normalize_text(
                unit
            )

            if normalized.startswith(ignored_prefixes):
                continue

            if ".md" in normalized:
                continue

            if not unit.endswith((".", "!", "?")):
                unit += "."

            unit = (
                unit[0].upper()
                + unit[1:]
            )

            units.append(unit)

        return units

    @staticmethod
    def _normalize_text(
        text: str,
    ) -> str:
        normalized = unicodedata.normalize(
            "NFD",
            text.lower(),
        )
        return " ".join(
            "".join(
                character
                for character in normalized
                if unicodedata.category(character) != "Mn"
            ).split()
        )

    @classmethod
    def _meaningful_terms(
        cls,
        text: str,
    ) -> set[str]:
        stop_words = {
            "avec", "comment", "dans", "des", "doit", "dois",
            "elle", "elles", "est", "faut", "les", "pour",
            "que", "quel", "quelle", "quels", "quelles",
            "sont", "sur", "une", "vous", "votre",
        }
        normalized = cls._normalize_text(
            text
        )
        return {
            word
            for word in re.findall(
                r"[a-z0-9]+",
                normalized,
            )
            if len(word) >= 3 and word not in stop_words
        }

    @staticmethod
    def _sanitize_answer(
        answer: str,
    ) -> str:
        if not answer:
            return ""

        without_sources = re.split(
            r"(?im)^\s*#{0,6}\s*Sources consultées\s*:?.*$",
            answer,
            maxsplit=1,
        )[0]

        visible_lines = [
            line
            for line in without_sources.splitlines()
            if ".md" not in line.lower()
            and "documents de référence" not in line.lower()
        ]

        sanitized = "\n".join(
            visible_lines
        ).strip()

        sanitized = re.sub(
            r"(?is)^\s*(?:"
            r"informations?\s+(?:compl[eè]tes?\s+)?"
            r"extraites?\s+des?\s+documents?(?:\s+disponibles?)?"
            r"|r[eé]ponse\s+de\s+(?:secours|fallback)"
            r")\s*[.:;-]*\s*",
            "",
            sanitized,
            count=1,
        )

        return re.sub(
            r"\n{3,}",
            "\n\n",
            sanitized,
        )

    @classmethod
    def _is_complete_answer(
        cls,
        answer: str,
        finish_reason: str | None,
    ) -> bool:
        if not answer or len(answer.strip()) < 20:
            return False

        if (
            finish_reason
            and "MAX_TOKENS" in finish_reason.upper()
        ):
            return False

        normalized = cls._normalize_text(
            answer.strip()
        )
        incomplete_endings = (
            "voici",
            "voici la",
            "voici le",
            "voici les",
            "choisissez un",
            "choisissez une",
            "les documents sont",
            "la procedure est",
        )

        if normalized.endswith(incomplete_endings):
            return False

        if answer.rstrip().endswith(":"):
            return False

        trailing_words = {
            "a", "au", "aux", "avec", "car", "ce", "ces",
            "cette", "dans", "de", "des", "du", "en", "et",
            "la", "le", "les", "mais", "ou", "par", "pour",
            "sans", "sur", "un", "une", "votre", "vos",
        }
        words = re.findall(
            r"[a-zà-ÿ]+",
            answer.lower(),
        )

        if words and words[-1] in trailing_words:
            return False

        meaningful_lines = [
            re.sub(
                r"^\s*(?:[-*]|\d+[.)])\s*",
                "",
                line,
            ).strip()
            for line in answer.splitlines()
            if line.strip()
        ]

        for line in meaningful_lines:
            if line.startswith(("http://", "https://")):
                continue

            if len(line) < 12:
                return False

            if not line.endswith((".", "!", "?", ";")):
                if "http://" not in line and "https://" not in line:
                    return False

        return True

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
            content = (
                context.get("content")
                or ""
            ).strip()

            formatted_contexts.append(
                (
                    f"[EXTRAIT {index}]\n"
                    f"{content}"
                )
            )

        return "\n\n".join(
            formatted_contexts
        )

    @staticmethod
    def _build_extractive_fallback(
        contexts: list[dict[str, Any]],
        question: str | None = None,
    ) -> str:
        """
        Retourne uniquement les extraits récupérés, sans génération.
        """
        extracted_sections: list[str] = []

        for context in contexts:
            content = NarsaGenerationService._content_for_extraction(
                context
            )

            if not content:
                continue

            concise_content = ""

            if question:
                concise_content = (
                    NarsaGenerationService
                    ._build_concise_extractive_answer(
                        question=question,
                        content=content,
                    )
                )

                if (
                    not concise_content
                    and NarsaGenerationService
                    ._asks_for_procedure(question)
                    and len(contexts) == 1
                ):
                    continue

            candidate = concise_content or content

            if (
                NarsaGenerationService
                ._is_complete_answer(
                    candidate,
                    finish_reason=None,
                )
            ):
                extracted_sections.append(candidate)

                if (
                    concise_content
                    and question
                    and NarsaGenerationService
                    ._asks_for_procedure(question)
                ):
                    break

        if not extracted_sections:
            return (
                "Je ne dispose pas d’un extrait documentaire "
                "suffisamment complet pour répondre sans risque."
            )

        return "\n\n".join(extracted_sections)

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
14. Ne crée aucune section de sources ou de références.
15. Ne cite aucun nom de fichier, titre technique ou chemin.
16. Limite généralement la réponse à 3 à 7 points essentiels.
17. Supprime les répétitions et utilise des phrases courtes.
18. Si une seule phrase suffit, ne crée pas de liste artificielle.
19. Ne coupe pas la réponse au milieu d'une phrase.

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
