from __future__ import annotations

import threading
from typing import Any

import numpy as np


"""
Fournit une instance unique de chaque modèle d'embedding.

Le classifieur d'intentions et le retriever RAG utilisent le même
modèle : sans mise en commun, il était chargé deux fois (mémoire
doublée et démarrage deux fois plus long).

Un petit cache des encodages de requêtes évite également de
recalculer l'embedding d'une question déjà posée.
"""


_models: dict[str, Any] = {}
_models_lock = threading.Lock()

QUERY_CACHE_SIZE = 512
_query_cache: dict[tuple[str, str], Any] = {}
_query_cache_order: list[tuple[str, str]] = []
_query_cache_lock = threading.Lock()

_query_cache_hits = 0
_query_cache_misses = 0


def get_embedding_model(model_name: str) -> Any:
    """Retourne le SentenceTransformer partagé pour ce modèle."""
    model = _models.get(model_name)

    if model is not None:
        return model

    with _models_lock:
        model = _models.get(model_name)

        if model is None:
            from sentence_transformers import SentenceTransformer

            print(f"Chargement du modèle d'embedding : {model_name}")
            model = SentenceTransformer(model_name)
            _models[model_name] = model

        return model


def encode_query(
    model_name: str,
    text: str,
    model: Any = None,
) -> np.ndarray:
    """
    Encode une requête avec mise en cache.

    Les requêtes sont normalisées : le vecteur retourné est prêt
    pour une similarité cosinus.
    """
    global _query_cache_hits, _query_cache_misses

    key = (model_name, text)

    with _query_cache_lock:
        cached = _query_cache.get(key)

        if cached is not None:
            _query_cache_hits += 1
            return cached

    encoder = model or get_embedding_model(model_name)
    embedding = encoder.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    with _query_cache_lock:
        _query_cache_misses += 1
        _query_cache[key] = embedding
        _query_cache_order.append(key)

        while len(_query_cache_order) > QUERY_CACHE_SIZE:
            oldest = _query_cache_order.pop(0)
            _query_cache.pop(oldest, None)

    return embedding


def embedding_cache_stats() -> dict[str, int]:
    return {
        "hits": _query_cache_hits,
        "misses": _query_cache_misses,
        "size": len(_query_cache),
        "models_loaded": len(_models),
    }


def reset_embedding_cache() -> None:
    """Vide le cache d'encodage (utilisé par les tests)."""
    global _query_cache_hits, _query_cache_misses

    with _query_cache_lock:
        _query_cache.clear()
        _query_cache_order.clear()
        _query_cache_hits = 0
        _query_cache_misses = 0
