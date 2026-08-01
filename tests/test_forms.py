from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(app)

EXPECTED_FILENAMES = {
    "formulaire_carte_grise_formule_bi.pdf",
    "formulaire_carte_grise_formule_bii.pdf",
    "formulaire_immatriculation_formule_i.pdf",
    "formulaire_immatriculation_formule_ii.pdf",
    "formulaire_reclamation_contravention.pdf",
}


def test_list_forms_returns_every_available_pdf() -> None:
    response = client.get("/forms")

    assert response.status_code == 200
    assert {
        form["filename"]
        for form in response.json()["forms"]
    } == EXPECTED_FILENAMES


def test_every_form_can_be_downloaded() -> None:
    for filename in EXPECTED_FILENAMES:
        response = client.get(
            f"/forms/{filename}"
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == (
            "application/pdf"
        )

        content_disposition = response.headers[
            "content-disposition"
        ]

        content_disposition.encode("ascii")

        assert filename in content_disposition
        assert response.content.startswith(b"%PDF")

