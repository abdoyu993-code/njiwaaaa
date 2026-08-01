from __future__ import annotations

from typing import Any

import pytest

from backend.app.rag import generation_service
from backend.app.rag.generation_service import (
    NarsaGenerationService,
)


class Gemini503Error(Exception):
    code = 503


class FailingModels:
    def __init__(self) -> None:
        self.calls = 0
        self.configs: list[Any] = []

    def generate_content(
        self,
        **kwargs: Any,
    ) -> None:
        self.calls += 1
        self.configs.append(
            kwargs["config"]
        )
        raise Gemini503Error(
            "Service Unavailable"
        )


class FailingClient:
    def __init__(self) -> None:
        self.models = FailingModels()


class FakeRAGService:
    def search(
        self,
        question: str,
        top_k: int,
    ) -> dict[str, Any]:
        contexts = [
            {
                "content": f"Extrait documentaire {index}.",
                "title": f"Source {index}",
                "filename": f"source-{index}.pdf",
                "relative_path": f"docs/source-{index}.pdf",
                "domain": "permis",
                "similarity_score": 0.9 - (index / 100),
            }
            for index in range(1, 5)
        ]

        return {
            "question": question,
            "retrieval_query": question,
            "status": "context_found",
            "domain": "permis",
            "intent": "permis_perte",
            "confidence": 0.95,
            "needs_clarification": False,
            "message": "Contextes trouvés.",
            "classification": {},
            "contexts": contexts,
            "sources": [
                {
                    "title": "Source 1",
                    "filename": "source-1.pdf",
                    "relative_path": "docs/source-1.pdf",
                    "domain": "permis",
                    "best_similarity_score": 0.89,
                }
            ],
        }


def test_gemini_503_retries_once_then_uses_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []

    monkeypatch.setattr(
        generation_service.time,
        "sleep",
        sleeps.append,
    )

    service = object.__new__(
        NarsaGenerationService
    )
    service.api_key = "test-only"
    service.model_name = "gemini-test"
    service.max_contexts = 3
    service.max_retries = 1
    service.retry_base_delay = 1.0
    service.retry_max_delay = 1.0
    service.rag_service = FakeRAGService()
    service.client = FailingClient()

    result = service.answer(
        "Comment remplacer mon permis perdu ?"
    )

    assert service.client.models.calls == 2
    assert sleeps == [1.0]
    assert result["generation_attempts"] == 2
    assert result["generated"] is False
    assert result["used_contexts_count"] == 3
    assert result["sources"][0]["filename"] == (
        "source-1.pdf"
    )
    assert "Extrait documentaire 1." in result["answer"]
    assert "Extrait documentaire 2." in result["answer"]
    assert "Extrait documentaire 3." in result["answer"]
    assert "Extrait documentaire 4." not in result["answer"]

    config = service.client.models.configs[0]

    assert config.max_output_tokens == 1200
    thinking_level = (
        config.thinking_config.thinking_level
    )

    assert thinking_level.value.lower() == "low"
