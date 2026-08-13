from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException

from backend.app.rag.generation_service import (
    NarsaGenerationService,
)


router = APIRouter(
    prefix="/documents",
    tags=["Documents sources"],
)


BASE_DIR = Path(__file__).resolve().parents[2]
DOCUMENTS_DIR = (BASE_DIR / "data" / "documents").resolve()


@lru_cache(maxsize=1)
def _indexed_documents() -> dict[str, Path]:
    """Liste blanche des Markdown réellement indexés."""
    if not DOCUMENTS_DIR.is_dir():
        return {}

    return {
        str(path.relative_to(DOCUMENTS_DIR)).replace(
            "\\",
            "/",
        ): path
        for path in DOCUMENTS_DIR.rglob("*.md")
    }


@router.get("")
def list_documents() -> dict[str, list[dict[str, str]]]:
    """Inventaire des documents consultables."""
    documents = [
        {
            "relative_path": relative_path,
            "domain": relative_path.split("/")[0],
            "filename": path.name,
        }
        for relative_path, path in sorted(
            _indexed_documents().items()
        )
    ]

    return {"documents": documents}


@router.get("/content")
def read_document(
    path: str,
) -> dict[str, str]:
    """
    Retourne le texte lisible d'un document source.

    Utilisé par l'interface pour afficher le passage exact
    ayant servi à construire une réponse.
    """
    documents = _indexed_documents()
    normalized = path.strip().lstrip("/").replace("\\", "/")

    file_path = documents.get(normalized)

    if file_path is None:
        raise HTTPException(
            status_code=404,
            detail="Document introuvable ou non indexé.",
        )

    raw = file_path.read_text(encoding="utf-8")
    title = next(
        (
            line.lstrip("#").strip()
            for line in raw.splitlines()
            if line.startswith("# ")
        ),
        file_path.stem.replace("_", " "),
    )

    return {
        "relative_path": normalized,
        "title": title,
        "domain": normalized.split("/")[0],
        "content": (
            NarsaGenerationService._clean_document_excerpt(raw)
        ),
    }
