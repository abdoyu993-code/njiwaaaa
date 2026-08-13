from __future__ import annotations

from typing import Any

import pytest

from backend.app.rag import rag_service
from backend.app.rag.generation_service import (
    NarsaGenerationService,
)
from backend.app.rag.rag_service import NarsaRAGService
from backend.app.rag.retriever import (
    DEFAULT_INTENT_BOOST,
    NarsaRetriever,
)
from backend.app.routes.ask import FORM_MAPPING


GENERAL_VISIT_QUESTION = (
    "Quels documents faut-il pour une visite technique ?"
)

MUTATION_VISIT_QUESTION = (
    "Quels documents de visite technique faut-il pour "
    "la mutation d’un véhicule ?"
)

MUTATION_QUESTION = (
    "Quels documents faut-il pour la mutation d’un véhicule ?"
)

VISIT_APPOINTMENT_QUESTION = (
    "Comment prendre rendez-vous pour une visite technique ?"
)


DOCUMENTS = [
    (
        (
            "Lors de la mutation, un procès-verbal de contrôle "
            "technique spécifique à la mutation doit être fourni. "
            "Il est délivré par un centre agréé et établi pour le "
            "véhicule concerné."
        ),
        {
            "domain": "visite_technique",
            "title": "Contrôle technique exigé lors de la mutation",
            "filename": "controle_technique_mutation_narsa.md",
            "relative_path": (
                "visite_technique/"
                "controle_technique_mutation_narsa.md"
            ),
            "chunk_index": 0,
            "intents": (
                "visite_technique_obligation, "
                "visite_technique_documents, "
                "mutation_vehicule, "
                "changement_proprietaire"
            ),
        },
        0.30,
    ),
    (
        "Le portail permet de préparer la mutation du véhicule.",
        {
            "domain": "vehicule",
            "title": "Portail de mutation des véhicules",
            "filename": "portail_mutation_vehicules.md",
            "relative_path": "vehicule/portail_mutation_vehicules.md",
            "chunk_index": 0,
            "intents": (
                "mutation_vehicule, vehicule_vendu, "
                "achat_vehicule, changement_proprietaire"
            ),
        },
        0.10,
    ),
    (
        "La mutation correspond au changement de propriétaire.",
        {
            "domain": "vehicule",
            "title": "Mutation d’un véhicule",
            "filename": "mutation_vehicule_narsa.md",
            "relative_path": "vehicule/mutation_vehicule_narsa.md",
            "chunk_index": 0,
            "intents": (
                "mutation_vehicule, changement_proprietaire, "
                "vehicule_vendu"
            ),
        },
        0.15,
    ),
    (
        "La plateforme permet de choisir un centre et une date.",
        {
            "domain": "visite_technique",
            "title": "Rendez-vous pour une visite technique",
            "filename": "rendez_vous_controle_technique.md",
            "relative_path": (
                "visite_technique/"
                "rendez_vous_controle_technique.md"
            ),
            "chunk_index": 0,
            "intents": (
                "visite_technique_rdv, visite_technique_centre, "
                "visite_technique_obligation"
            ),
        },
        0.20,
    ),
]


class FakeEmbeddingModel:
    def encode(
        self,
        text: str,
        **kwargs: Any,
    ) -> Any:
        import numpy as np

        return np.array([0.1, 0.2], dtype="float32")


class FakeCollection:
    def __init__(self) -> None:
        self.last_where: dict[str, Any] | None = None

    @staticmethod
    def count() -> int:
        return len(DOCUMENTS)

    def query(
        self,
        **kwargs: Any,
    ) -> dict[str, list[list[Any]]]:
        self.last_where = kwargs.get("where")
        allowed_domains = self._allowed_domains(
            self.last_where
        )
        selected = [
            item
            for item in DOCUMENTS
            if (
                not allowed_domains
                or item[1]["domain"] in allowed_domains
            )
        ]
        selected.sort(key=lambda item: item[2])

        return {
            "documents": [[item[0] for item in selected]],
            "metadatas": [[item[1] for item in selected]],
            "distances": [[item[2] for item in selected]],
        }

    @staticmethod
    def _allowed_domains(
        where: dict[str, Any] | None,
    ) -> set[str]:
        if not where:
            return set()

        domain_filter = where["domain"]

        if isinstance(domain_filter, str):
            return {domain_filter}

        return set(domain_filter["$in"])


def build_retriever() -> NarsaRetriever:
    retriever = object.__new__(NarsaRetriever)
    retriever.intent_boost = DEFAULT_INTENT_BOOST
    retriever.embedding_model = FakeEmbeddingModel()
    retriever.collection = FakeCollection()
    return retriever


def classification(
    domain: str,
    intent: str,
    confidence: float = 0.9,
) -> dict[str, Any]:
    return {
        "domain": domain,
        "intent": intent,
        "confidence": confidence,
        "needs_clarification": False,
        "alternatives": [],
    }


def build_rag(
    retriever: NarsaRetriever,
) -> NarsaRAGService:
    service = object.__new__(NarsaRAGService)
    service.clarification_threshold = 0.60
    service.minimum_similarity_score = 0.30
    service.retriever = retriever
    return service


def test_a_general_visit_documents_has_no_mutation_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        rag_service,
        "predict_intent",
        lambda text, top_k: classification(
            "visite_technique",
            "visite_technique_documents",
        ),
    )
    result = build_rag(build_retriever()).search(
        GENERAL_VISIT_QUESTION
    )

    assert result["domain"] == "visite_technique"
    assert result["intent"] == "visite_technique_documents"
    assert result["status"] == "no_context_found"
    assert result["contexts"] == []
    assert result["sources"] == []


def test_b_combined_question_prioritizes_control_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        rag_service,
        "predict_intent",
        lambda text, top_k: classification(
            "vehicule",
            "mutation_vehicule",
        ),
    )
    retriever = build_retriever()
    result = build_rag(retriever).search(
        MUTATION_VISIT_QUESTION
    )

    assert retriever.collection.last_where == {
        "domain": {
            "$in": ["vehicule", "visite_technique"]
        }
    }
    assert result["status"] == "context_found"
    assert result["contexts"][0]["filename"] == (
        "controle_technique_mutation_narsa.md"
    )
    assert result["contexts"][0]["context_coverage"] == 1.0
    assert result["sources"][0]["filename"] == (
        "controle_technique_mutation_narsa.md"
    )
    assert "procès-verbal de contrôle technique" in (
        result["contexts"][0]["content"]
    )
    assert "centre agréé" in result["contexts"][0]["content"]
    assert "véhicule concerné" in (
        result["contexts"][0]["content"]
    )

    grounded_answer = (
        NarsaGenerationService
        ._build_extractive_fallback(
            result["contexts"]
        )
    )

    assert "procès-verbal de contrôle technique" in (
        grounded_answer
    )
    assert "centre agréé" in grounded_answer
    assert "véhicule concerné" in grounded_answer
    assert (
        "informations fiables suffisantes"
        not in grounded_answer
    )
    assert "visite technique" in result["retrieval_query"].lower()
    assert "documents" in result["retrieval_query"].lower()
    assert "mutation" in result["retrieval_query"].lower()


def test_c_plain_mutation_keeps_general_documents_and_forms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        rag_service,
        "predict_intent",
        lambda text, top_k: classification(
            "vehicule",
            "mutation_vehicule",
        ),
    )
    retriever = build_retriever()
    result = build_rag(retriever).search(
        MUTATION_QUESTION
    )

    assert retriever.collection.last_where == {
        "domain": "vehicule"
    }
    assert {
        context["filename"]
        for context in result["contexts"]
    } == {
        "portail_mutation_vehicules.md",
        "mutation_vehicule_narsa.md",
    }
    assert [
        form["filename"]
        for form in FORM_MAPPING["mutation_vehicule"]
    ] == [
        "formulaire_carte_grise_formule_bi.pdf",
        "formulaire_carte_grise_formule_bii.pdf",
    ]


def test_d_visit_appointment_keeps_appointment_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        rag_service,
        "predict_intent",
        lambda text, top_k: classification(
            "visite_technique",
            "visite_technique_rdv",
        ),
    )
    result = build_rag(build_retriever()).search(
        VISIT_APPOINTMENT_QUESTION
    )

    assert result["status"] == "context_found"
    assert result["contexts"][0]["filename"] == (
        "rendez_vous_controle_technique.md"
    )
