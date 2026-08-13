from __future__ import annotations

import time
from typing import Any

import pytest
from fastapi.testclient import TestClient

from backend.app.core import embeddings
from backend.app.main import app
from backend.app.rag import answer_cache, generation_service


client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_caches() -> None:
    answer_cache.clear()
    embeddings.reset_embedding_cache()


def documentary_result(question: str) -> dict[str, Any]:
    return {
        "question": question,
        "answer": "Réponse documentaire complète.",
        "status": "context_found",
        "domain": "permis",
        "intent": "permis_perte",
        "confidence": 0.92,
        "needs_clarification": False,
        "generated": False,
        "model": None,
        "finish_reason": None,
        "sources": [],
        "classification": {},
        "retrieval_query": question,
        "used_contexts_count": 1,
        "generation_attempts": 0,
        "classification_time_ms": 12.0,
        "retrieval_time_ms": 20.0,
        "generation_time_ms": 1500.0,
        "total_time_ms": 1532.0,
        "extractive": True,
    }


class CountingService:
    """Service de génération qui compte ses appels."""

    def __init__(self) -> None:
        self.calls = 0

    def answer(
        self,
        question: str,
        top_k: int = 3,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.calls += 1

        return documentary_result(question)


# ------------------------- Cache de réponses --------------------------


def test_identical_question_is_not_recomputed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = CountingService()
    monkeypatch.setattr(
        generation_service,
        "get_generation_service",
        lambda: service,
    )

    first = generation_service.generate_answer(
        "Comment récupérer mon permis perdu ?"
    )
    second = generation_service.generate_answer(
        "Comment récupérer mon permis perdu ?"
    )

    assert service.calls == 1
    assert first["cached"] is False
    assert second["cached"] is True
    assert second["answer"] == first["answer"]
    # Le temps de génération d'une réponse servie par le cache
    # est nul : c'est tout l'intérêt.
    assert second["generation_time_ms"] == 0.0


def test_cache_ignores_case_and_punctuation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = CountingService()
    monkeypatch.setattr(
        generation_service,
        "get_generation_service",
        lambda: service,
    )

    generation_service.generate_answer(
        "Comment payer une amende ?"
    )
    reused = generation_service.generate_answer(
        "comment payer une amende"
    )

    assert service.calls == 1
    assert reused["cached"] is True


def test_questions_with_personal_data_are_not_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = CountingService()
    monkeypatch.setattr(
        generation_service,
        "get_generation_service",
        lambda: service,
    )

    question = (
        "Mon numéro de dossier est 123456789, où en est-il ?"
    )

    generation_service.generate_answer(question)
    generation_service.generate_answer(question)

    # Aucune mise en cache : la question contient un identifiant.
    assert service.calls == 2


def test_cache_is_invalidated_when_documents_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = CountingService()
    monkeypatch.setattr(
        generation_service,
        "get_generation_service",
        lambda: service,
    )

    corpus = {"signature": "avant-modification"}

    monkeypatch.setattr(
        answer_cache,
        "documents_signature",
        lambda: corpus["signature"],
    )
    monkeypatch.setattr(
        answer_cache,
        "SIGNATURE_CHECK_INTERVAL",
        0.0,
    )

    generation_service.generate_answer(
        "Quels documents pour la carte grise ?"
    )
    reused = generation_service.generate_answer(
        "Quels documents pour la carte grise ?"
    )

    assert service.calls == 1
    assert reused["cached"] is True

    # Un document est modifié : la réponse doit être recalculée.
    corpus["signature"] = "apres-modification"

    refreshed = generation_service.generate_answer(
        "Quels documents pour la carte grise ?"
    )

    assert service.calls == 2
    assert refreshed["cached"] is False


def test_clarifications_are_not_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ClarifyingService:
        def __init__(self) -> None:
            self.calls = 0

        def answer(
            self,
            question: str,
            top_k: int = 3,
            **kwargs: Any,
        ) -> dict[str, Any]:
            self.calls += 1
            result = documentary_result(question)
            result["status"] = "clarification_required"

            return result

    service = ClarifyingService()
    monkeypatch.setattr(
        generation_service,
        "get_generation_service",
        lambda: service,
    )

    generation_service.generate_answer("carte grise")
    generation_service.generate_answer("carte grise")

    assert service.calls == 2


# ------------------------ Modèles et embeddings -----------------------


def test_embedding_model_is_loaded_once() -> None:
    """Le modèle lourd est partagé, pas rechargé à chaque appel."""
    from backend.app.classifier.classifier_service import (
        get_classifier_service,
    )
    from backend.app.rag.retriever import (
        EMBEDDING_MODEL_NAME,
    )

    classifier = get_classifier_service()
    shared = embeddings.get_embedding_model(
        EMBEDDING_MODEL_NAME
    )

    assert classifier.embedding_model is shared
    assert embeddings.embedding_cache_stats()[
        "models_loaded"
    ] == 1


def test_repeated_question_reuses_its_embedding() -> None:
    from backend.app.classifier.classifier_service import (
        predict_intent,
    )

    embeddings.reset_embedding_cache()

    predict_intent("Comment payer une amende ?")
    stats_after_first = embeddings.embedding_cache_stats()

    predict_intent("Comment payer une amende ?")
    stats_after_second = embeddings.embedding_cache_stats()

    assert stats_after_first["misses"] == 1
    assert stats_after_second["hits"] >= 1
    assert stats_after_second["misses"] == 1


def test_classification_runs_once_per_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """La classification ne doit pas être relancée par couche."""
    from backend.app.rag import rag_service

    calls: list[str] = []
    original = rag_service.predict_intent

    def counting_predict(text: str, top_k: int = 3) -> dict:
        calls.append(text)

        return original(text=text, top_k=top_k)

    monkeypatch.setattr(
        rag_service,
        "predict_intent",
        counting_predict,
    )

    service = rag_service.get_rag_service()
    service.search("Comment payer une amende ?", top_k=3)

    assert len(calls) == 1


# --------------------- Réponse locale immédiate -----------------------


def test_quota_exhaustion_skips_gemini(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Quand tous les modèles sont au repos, la réponse est
    documentaire et immédiate : plus d'appel réseau inutile.
    """
    service = object.__new__(
        generation_service.NarsaGenerationService
    )
    service.model_name = "modele-a"
    service.gemini_models = ["modele-a", "modele-b"]
    service.client = object()
    service._model_blocked_until = {
        "modele-a": time.time() + 600,
        "modele-b": time.time() + 600,
    }

    assert service._candidate_models() == []
    assert service.generation_available is False


def test_available_model_is_reported() -> None:
    service = object.__new__(
        generation_service.NarsaGenerationService
    )
    service.model_name = "modele-a"
    service.gemini_models = ["modele-a", "modele-b"]
    service.client = object()
    service._model_blocked_until = {
        "modele-a": time.time() + 600,
    }

    assert service._candidate_models() == ["modele-b"]
    assert service.generation_available is True


def test_quota_error_is_recognized() -> None:
    recognize = (
        generation_service.NarsaGenerationService._is_quota_error
    )

    assert recognize(
        RuntimeError("429 RESOURCE_EXHAUSTED"),
        None,
    )
    assert recognize(RuntimeError("boom"), 429)
    assert not recognize(RuntimeError("500 erreur"), 500)


def test_metrics_are_present_in_the_api_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.app.routes import ask as ask_route

    monkeypatch.setattr(
        ask_route,
        "generate_answer",
        lambda **kwargs: documentary_result(
            kwargs["question"]
        ),
    )

    payload = client.post(
        "/ask",
        json={"question": "Comment récupérer mon permis ?"},
    ).json()

    metadata = payload["metadata"]

    for key in (
        "classification_time_ms",
        "retrieval_time_ms",
        "generation_time_ms",
        "total_time_ms",
    ):
        assert key in metadata

    assert "cached" in metadata
    assert "language" in metadata
