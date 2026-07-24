from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse


router = APIRouter(
    prefix="/forms",
    tags=["Formulaires"],
)


BASE_DIR = Path(__file__).resolve().parents[2]
FORMS_DIR = (BASE_DIR / "data" / "forms").resolve()


ALLOWED_FORMS: dict[str, dict[str, str]] = {
    "formulaire_carte_grise_formule_bi.pdf": {
        "title": "Formulaire carte grise - Formule BI",
        "intent": "mutation_vehicule",
    },
    "formulaire_carte_grise_formule_bii.pdf": {
        "title": "Formulaire carte grise - Formule BII",
        "intent": "mutation_vehicule",
    },
    "formulaire_immatriculation_formule_i.pdf": {
        "title": "Formulaire d'immatriculation - Formule I",
        "intent": "immatriculation_vehicule",
    },
    "formulaire_immatriculation_formule_ii.pdf": {
        "title": "Formulaire d'immatriculation - Formule II",
        "intent": "immatriculation_vehicule",
    },
    "formulaire_reclamation_contravention.pdf": {
        "title": (
            "Formulaire de reclamation "
            "sur avis de contravention"
        ),
        "intent": "contestation_amende",
    },
}


@router.get("")
def list_forms() -> dict[str, list[dict[str, str]]]:
    forms: list[dict[str, str]] = []

    for filename, metadata in ALLOWED_FORMS.items():
        file_path = FORMS_DIR / filename

        if not file_path.is_file():
            continue

        forms.append(
            {
                "filename": filename,
                "title": metadata["title"],
                "intent": metadata["intent"],
                "download_url": (
                    f"/forms/{quote(filename)}"
                ),
            }
        )

    return {
        "forms": forms,
    }


@router.get("/{filename}")
def download_form(
    filename: str,
) -> FileResponse:
    if filename not in ALLOWED_FORMS:
        raise HTTPException(
            status_code=404,
            detail="Formulaire introuvable.",
        )

    file_path = (FORMS_DIR / filename).resolve()

    try:
        file_path.relative_to(FORMS_DIR)
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail="Chemin de fichier invalide.",
        ) from error

    if not file_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=(
                "Le fichier PDF est absent du dossier forms."
            ),
        )

    return FileResponse(
        path=str(file_path),
        media_type="application/pdf",
        filename=filename,
        content_disposition_type="attachment",
    )