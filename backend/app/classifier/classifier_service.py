from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from backend.app.core.embeddings import (
    encode_query,
    get_embedding_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = PROJECT_ROOT / "backend" / "models"

CLASSIFIER_PATH = MODELS_DIR / "intent_classifier.joblib"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"
INTENT_DOMAIN_PATH = MODELS_DIR / "intent_to_domain.json"
MODEL_NAME_PATH = MODELS_DIR / "sentence_transformer_model.txt"

DEFAULT_CONFIDENCE_THRESHOLD = 0.60


class IntentClassifierService:
    """Service de prédiction du domaine et de l'intention."""

    def __init__(
        self,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        self.confidence_threshold = confidence_threshold

        self._validate_files()

        self.classifier = joblib.load(CLASSIFIER_PATH)
        self.label_encoder = joblib.load(LABEL_ENCODER_PATH)

        with INTENT_DOMAIN_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            self.intent_to_domain = json.load(file)

        self.model_name = MODEL_NAME_PATH.read_text(
            encoding="utf-8",
        ).strip()

        # Modèle partagé avec le retriever RAG : il n'est chargé
        # qu'une seule fois pour l'ensemble de l'application.
        self.embedding_model = get_embedding_model(
            self.model_name
        )

        print("Service de classification chargé.")
        print(
            "Nombre d'intentions :",
            len(self.label_encoder.classes_),
        )

    def _validate_files(self) -> None:
        """Vérifie que les fichiers nécessaires existent."""
        required_files = [
            CLASSIFIER_PATH,
            LABEL_ENCODER_PATH,
            INTENT_DOMAIN_PATH,
            MODEL_NAME_PATH,
        ]

        missing_files = [
            path
            for path in required_files
            if not path.exists()
        ]

        if missing_files:
            formatted = "\n".join(
                str(path)
                for path in missing_files
            )

            raise FileNotFoundError(
                "Fichiers de modèle introuvables :\n"
                f"{formatted}"
            )

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Effectue une normalisation légère.

        Cette normalisation ne traduit pas la darija.
        Elle nettoie uniquement les espaces inutiles.
        """
        if not isinstance(text, str):
            raise TypeError(
                "Le texte doit être une chaîne de caractères."
            )

        normalized = " ".join(text.strip().split())

        if not normalized:
            raise ValueError(
                "Le texte ne peut pas être vide."
            )

        return normalized

    def predict(
        self,
        text: str,
        top_k: int = 3,
    ) -> dict[str, Any]:
        """Prédit l'intention, le domaine et la confiance."""
        normalized_text = self.normalize_text(text)

        # Encodage mis en cache : une question déjà classée ne
        # repasse pas par le modèle.
        embedding = encode_query(
            model_name=self.model_name,
            text=normalized_text,
            model=self.embedding_model,
        ).reshape(1, -1)

        probabilities = self.classifier.predict_proba(
            embedding
        )[0]

        sorted_indices = np.argsort(
            probabilities
        )[::-1]

        best_index = int(sorted_indices[0])
        best_confidence = float(
            probabilities[best_index]
        )

        predicted_intent = (
            self.label_encoder.inverse_transform(
                [best_index]
            )[0]
        )

        predicted_domain = self.intent_to_domain[
            predicted_intent
        ]

        top_k = max(
            1,
            min(
                top_k,
                len(sorted_indices),
            ),
        )

        alternatives = []

        for index in sorted_indices[:top_k]:
            intent = self.label_encoder.inverse_transform(
                [int(index)]
            )[0]

            alternatives.append(
                {
                    "intent": intent,
                    "domain": self.intent_to_domain[intent],
                    "confidence": round(
                        float(probabilities[index]),
                        4,
                    ),
                }
            )

        needs_clarification = (
            best_confidence
            < self.confidence_threshold
        )

        return {
            "text": text,
            "normalized_text": normalized_text,
            "domain": predicted_domain,
            "intent": predicted_intent,
            "confidence": round(
                best_confidence,
                4,
            ),
            "confidence_threshold": (
                self.confidence_threshold
            ),
            "needs_clarification": (
                needs_clarification
            ),
            "alternatives": alternatives,
        }


_classifier_service: IntentClassifierService | None = None


def get_classifier_service() -> IntentClassifierService:
    """Retourne une instance unique du service."""
    global _classifier_service

    if _classifier_service is None:
        _classifier_service = IntentClassifierService()

    return _classifier_service


def predict_intent(
    text: str,
    top_k: int = 3,
) -> dict[str, Any]:
    """Fonction simple utilisable par les autres modules."""
    service = get_classifier_service()

    return service.predict(
        text=text,
        top_k=top_k,
    )