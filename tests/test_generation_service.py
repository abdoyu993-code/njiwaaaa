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
            "classification_time_ms": 10.0,
            "retrieval_time_ms": 20.0,
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
    service.max_contexts = 2
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
    assert result["used_contexts_count"] == 2
    assert result["sources"][0]["filename"] == (
        "source-1.pdf"
    )
    assert "Extrait documentaire 1." in result["answer"]
    assert "Extrait documentaire 2." in result["answer"]
    assert "Extrait documentaire 3." not in result["answer"]
    assert "Extrait documentaire 4." not in result["answer"]

    config = service.client.models.configs[0]

    assert config.max_output_tokens == 600
    thinking_level = (
        config.thinking_config.thinking_level
    )

    assert thinking_level.value.lower() == "low"
    assert result["classification_time_ms"] == 10.0
    assert result["retrieval_time_ms"] == 20.0
    assert result["generation_time_ms"] >= 0.0
    assert result["total_time_ms"] >= 0.0


class UnexpectedModels:
    @staticmethod
    def generate_content(
        **kwargs: Any,
    ) -> None:
        raise AssertionError(
            "Gemini ne doit pas être appelé."
        )


class UnexpectedClient:
    models = UnexpectedModels()


class DirectRAGService:
    def search(
        self,
        question: str,
        top_k: int,
    ) -> dict[str, Any]:
        content = (
            "Un procès-verbal de contrôle technique pour la "
            "mutation doit être fourni.\n"
            "- Il est délivré par un centre agréé.\n"
            "- Il est établi pour le véhicule concerné."
        )
        context = {
            "content": content,
            "title": "Contrôle technique mutation",
            "filename": "controle_technique_mutation_narsa.md",
            "relative_path": (
                "visite_technique/"
                "controle_technique_mutation_narsa.md"
            ),
            "domain": "visite_technique",
            "similarity_score": 0.96,
            "intent_match": True,
            "context_coverage": 1.0,
            "rank": 1,
        }

        return {
            "question": question,
            "retrieval_query": question,
            "status": "context_found",
            "domain": "vehicule",
            "intent": "mutation_vehicule",
            "confidence": 0.9,
            "needs_clarification": False,
            "message": "Contexte trouvé.",
            "classification": {},
            "contexts": [context],
            "sources": [{
                "title": context["title"],
                "filename": context["filename"],
                "relative_path": context["relative_path"],
                "domain": context["domain"],
                "best_similarity_score": 0.96,
            }],
            "classification_time_ms": 5.0,
            "retrieval_time_ms": 8.0,
        }


def test_direct_documentary_answer_avoids_gemini() -> None:
    service = object.__new__(NarsaGenerationService)
    service.model_name = "gemini-test"
    service.max_contexts = 2
    service.max_retries = 1
    service.retry_base_delay = 1.0
    service.retry_max_delay = 1.0
    service.extractive_score_threshold = 0.88
    service.extractive_max_content_chars = 2200
    service.rag_service = DirectRAGService()
    service.client = UnexpectedClient()

    result = service.answer(
        "Quels documents de visite technique faut-il "
        "pour une mutation ?"
    )

    assert result["extractive"] is True
    assert result["generated"] is False
    assert result["generation_attempts"] == 0
    assert result["generation_time_ms"] == 0.0
    assert result["sources"][0]["filename"].endswith(".md")
    assert "procès-verbal" in result["answer"]
    assert "Sources consultées" not in result["answer"]
    assert ".md" not in result["answer"]


class SuccessfulResponse:
    text = (
        "Réponse citoyenne concise.\n\n"
        "Sources consultées\n"
        "- document_technique.md"
    )
    candidates: list[Any] = []


class SuccessfulModels:
    def __init__(self) -> None:
        self.calls = 0
        self.configs: list[Any] = []

    def generate_content(
        self,
        **kwargs: Any,
    ) -> SuccessfulResponse:
        self.calls += 1
        self.configs.append(kwargs["config"])
        assert "Fichier :" not in kwargs["contents"]
        assert "Chemin :" not in kwargs["contents"]
        return SuccessfulResponse()


class SuccessfulClient:
    def __init__(self) -> None:
        self.models = SuccessfulModels()


def test_complex_answer_uses_gemini_and_hides_sources() -> None:
    service = object.__new__(NarsaGenerationService)
    service.model_name = "gemini-test"
    service.max_contexts = 2
    service.max_retries = 1
    service.retry_base_delay = 1.0
    service.retry_max_delay = 1.0
    service.extractive_score_threshold = 0.88
    service.extractive_max_content_chars = 2200
    service.rag_service = FakeRAGService()
    service.client = SuccessfulClient()

    result = service.answer(
        "Comment remplacer mon permis perdu ?"
    )

    assert service.client.models.calls == 1
    assert result["generated"] is True
    assert result["extractive"] is False
    assert result["used_contexts_count"] == 2
    assert result["answer"] == "Réponse citoyenne concise."
    assert "Sources consultées" not in result["answer"]
    assert ".md" not in result["answer"]
    assert (
        service.client.models.configs[0].max_output_tokens
        == 600
    )


class MaxTokensCandidate:
    finish_reason = "MAX_TOKENS"


class MaxTokensResponse:
    text = "Voici la"
    candidates = [MaxTokensCandidate()]


class MaxTokensModels:
    def __init__(self) -> None:
        self.calls = 0

    def generate_content(
        self,
        **kwargs: Any,
    ) -> MaxTokensResponse:
        self.calls += 1
        return MaxTokensResponse()


class MaxTokensClient:
    def __init__(self) -> None:
        self.models = MaxTokensModels()


class PermitProcedureRAGService:
    def search(
        self,
        question: str,
        top_k: int,
    ) -> dict[str, Any]:
        contents = [
            (
                "## Démarche\n\n"
                "1. Demandez un duplicata du permis.\n"
                "2. Préparez les documents nécessaires.\n"
                "3. Déposez le dossier dans une agence Al Barid "
                "Bank ou Barid Cash offrant les services NARSA.\n"
                "4. Attendez la validation du dossier.\n"
                "5. Récupérez le permis après réception du SMS."
            ),
            (
                "## Documents nécessaires\n\n"
                "- Un formulaire dûment rempli et signé.\n"
                "- Un justificatif d’identité valide.\n"
                "- Une déclaration de perte ou de vol."
            ),
        ]
        contexts = [
            {
                "content": content,
                "title": f"Permis {index}",
                "filename": f"permis-{index}.md",
                "relative_path": f"permis/permis-{index}.md",
                "domain": "permis",
                "similarity_score": 0.95 - index / 100,
                "intent_match": True,
                "context_coverage": 1.0,
                "rank": index,
            }
            for index, content in enumerate(contents, start=1)
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
                    "title": context["title"],
                    "filename": context["filename"],
                    "relative_path": context["relative_path"],
                    "domain": "permis",
                    "best_similarity_score": context[
                        "similarity_score"
                    ],
                }
                for context in contexts
            ],
            "classification_time_ms": 2.0,
            "retrieval_time_ms": 3.0,
        }


def test_a_max_tokens_retries_then_uses_complete_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(
        generation_service.time,
        "sleep",
        sleeps.append,
    )
    service = object.__new__(NarsaGenerationService)
    service.model_name = "gemini-test"
    service.max_contexts = 2
    service.max_retries = 1
    service.retry_base_delay = 1.0
    service.retry_max_delay = 1.0
    service.extractive_score_threshold = 0.88
    service.extractive_max_content_chars = 2200
    service.rag_service = PermitProcedureRAGService()
    service.client = MaxTokensClient()

    result = service.answer(
        "J’ai perdu mon permis de conduire, que dois-je faire ?"
    )

    assert service.client.models.calls == 2
    assert sleeps == [1.0]
    assert result["generated"] is False
    assert result["generation_attempts"] == 2
    assert not result["answer"].rstrip().endswith("Voici")
    assert "Demandez un duplicata" in result["answer"]
    assert "Déposez le dossier" in result["answer"]
    assert "Récupérez le permis" in result["answer"]
    assert NarsaGenerationService._is_complete_answer(
        result["answer"],
        finish_reason=None,
    )


class AppointmentRAGService:
    def search(
        self,
        question: str,
        top_k: int,
    ) -> dict[str, Any]:
        content = (
            "## Métadonnées\n\n"
            "URL : https://controletechnique.ma/Rendez-vous\n\n"
            "## Étapes de prise de rendez-vous\n\n"
            "La procédure comprend les étapes suivantes :\n\n"
            "1. Accédez à la plateforme de rendez-vous.\n"
            "2. Choisissez le type de visite technique.\n"
            "3. Sélectionnez un centre de contrôle agréé.\n"
            "4. Renseignez les informations du véhicule et du "
            "propriétaire.\n"
            "5. Choisissez une date et une heure disponibles.\n"
            "6. Confirmez le rendez-vous."
        )
        context = {
            "content": content,
            "title": "Rendez-vous visite technique",
            "filename": "rendez_vous_controle_technique.md",
            "relative_path": (
                "visite_technique/"
                "rendez_vous_controle_technique.md"
            ),
            "domain": "visite_technique",
            "similarity_score": 0.96,
            "intent_match": True,
            "context_coverage": 1.0,
            "rank": 1,
        }

        return {
            "question": question,
            "retrieval_query": question,
            "status": "context_found",
            "domain": "visite_technique",
            "intent": "visite_technique_rdv",
            "confidence": 0.95,
            "needs_clarification": False,
            "message": "Contexte trouvé.",
            "classification": {},
            "contexts": [context],
            "sources": [{
                "title": context["title"],
                "filename": context["filename"],
                "relative_path": context["relative_path"],
                "domain": context["domain"],
                "best_similarity_score": 0.96,
            }],
            "classification_time_ms": 2.0,
            "retrieval_time_ms": 3.0,
        }


def test_c_appointment_extracts_complete_steps_and_verified_link() -> None:
    service = object.__new__(NarsaGenerationService)
    service.model_name = "gemini-test"
    service.max_contexts = 2
    service.max_retries = 1
    service.retry_base_delay = 1.0
    service.retry_max_delay = 1.0
    service.extractive_score_threshold = 0.88
    service.extractive_max_content_chars = 2200
    service.rag_service = AppointmentRAGService()
    service.client = UnexpectedClient()

    result = service.answer(
        "Comment prendre rendez-vous pour une visite technique ?"
    )

    assert result["extractive"] is True
    assert result["generation_attempts"] == 0
    assert "1. Accédez" in result["answer"]
    assert "6. Confirmez" in result["answer"]
    assert (
        "https://controletechnique.ma/Rendez-vous"
        in result["answer"]
    )
    assert "Choisissez un" not in result["answer"].splitlines()[-1]


def test_appointment_chunks_from_one_document_are_merged() -> None:
    contexts = [
        {
            "content": (
                "## Informations demandées\n\n"
                "- L’heure du rendez-vous."
            ),
            "filename": "rendez_vous_controle_technique.md",
            "relative_path": "visite_technique/rendez-vous.md",
            "similarity_score": 0.97,
            "intent_match": True,
        },
        {
            "content": (
                "## Étapes de prise de rendez-vous\n\n"
                "1. Choisissez le type de visite.\n"
                "2. Sélectionnez un centre agréé.\n"
                "3. Renseignez les informations demandées.\n"
                "4. Choisissez une date et une heure.\n"
                "5. Confirmez le rendez-vous."
            ),
            "filename": "rendez_vous_controle_technique.md",
            "relative_path": "visite_technique/rendez-vous.md",
            "similarity_score": 0.95,
            "intent_match": True,
        },
    ]

    selected = NarsaGenerationService._select_distinct_contexts(
        contexts,
        limit=2,
    )
    answer = NarsaGenerationService._build_concise_extractive_answer(
        question=(
            "Comment prendre rendez-vous pour une visite technique ?"
        ),
        content=selected[0]["content"],
    )

    assert len(selected) == 1
    assert selected[0]["merged_chunks_count"] == 2
    assert "1. Choisissez" in answer
    assert "5. Confirmez" in answer
    assert answer != "L’heure du rendez-vous."


def test_permit_fallback_starts_with_complete_citizen_procedure() -> None:
    document_content = (
        "## Principe\n\n"
        "En cas de perte du permis de conduire, le titulaire peut "
        "demander un duplicata.\n\n"
        "## Lieu de dépôt\n\n"
        "La demande doit être déposée dans une agence Al Barid Bank "
        "ou Barid Cash proposant les services NARSA.\n\n"
        "## Documents nécessaires\n\n"
        "Le dossier comprend une déclaration de perte ou de vol.\n\n"
        "## Traitement de la demande\n\n"
        "1. L’agence réceptionne la demande.\n"
        "2. L’agence enregistre la demande.\n"
        "3. La NARSA vérifie et valide le dossier.\n"
        "4. Le duplicata est produit.\n"
        "5. Le demandeur reçoit un SMS.\n"
        "6. Le permis est récupéré dans l’agence de dépôt.\n\n"
        "## Suivi du dossier\n\n"
        "https://khadamatnarsa.ma/fr/suivi-des-dossiers"
    )
    answer = NarsaGenerationService._build_extractive_fallback(
        contexts=[{
            "content": "Extrait classé.",
            "document_content": document_content,
        }],
        question=(
            "J’ai perdu mon permis de conduire, que dois-je faire ?"
        ),
    )

    assert not answer.startswith(
        "Informations complètes extraites des documents disponibles"
    )
    assert "Préparez les documents nécessaires" in answer
    assert "déclaration de perte ou de vol" in answer
    assert "déposée dans une agence Al Barid Bank" in answer
    assert "vérifie et valide le dossier" in answer
    assert "reçoit un SMS" in answer
    assert "récupéré dans l’agence de dépôt" in answer
