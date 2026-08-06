from __future__ import annotations

from pathlib import Path

from frontend.rendering import (
    local_forms_for_display,
    previous_context_from_messages,
    service_links_for_display,
    sources_for_display,
)


RESPONSE_DATA = {
    "sources": [
        {
            "title": "Permis perdu",
            "filename": "permis_perte_vol_narsa.md",
        }
    ]
}


def test_sources_are_hidden_in_citizen_mode() -> None:
    assert sources_for_display(
        RESPONSE_DATA,
        technical_details_enabled=False,
    ) == []


def test_sources_are_available_in_technical_mode() -> None:
    assert sources_for_display(
        RESPONSE_DATA,
        technical_details_enabled=True,
    ) == RESPONSE_DATA["sources"]


def test_streamlit_has_one_technical_section_and_no_legacy_section() -> None:
    app_source = (
        Path(__file__).resolve().parents[1]
        / "frontend"
        / "app.py"
    ).read_text(encoding="utf-8")

    assert "Documents de référence" not in app_source
    assert app_source.count(
        "Sources et détails techniques"
    ) == 1
    assert "display_sources" not in app_source


def test_previous_context_comes_from_last_assistant_response() -> None:
    messages = [
        {
            "role": "assistant",
            "content": "Réponse permis.",
            "metadata": {
                "question": "J’ai perdu mon permis.",
                "domain": "permis",
                "intent": "permis_perte",
            },
        }
    ]

    assert previous_context_from_messages(messages) == {
        "previous_question": "J’ai perdu mon permis.",
        "previous_domain": "permis",
        "previous_intent": "permis_perte",
    }


def test_empty_or_external_forms_do_not_show_local_form_section() -> None:
    assert local_forms_for_display([]) == []
    assert local_forms_for_display([
        {
            "filename": None,
            "is_external": True,
            "download_url": "https://example.test/service",
        }
    ]) == []


def test_real_pdf_forms_remain_visible() -> None:
    forms = [
        {
            "filename": "formulaire_carte_grise_formule_bi.pdf",
            "download_url": (
                "/forms/formulaire_carte_grise_formule_bi.pdf"
            ),
        },
        {
            "filename": "formulaire_immatriculation_formule_i.pdf",
            "download_url": (
                "/forms/formulaire_immatriculation_formule_i.pdf"
            ),
        },
        {
            "filename": "formulaire_reclamation_contravention.pdf",
            "download_url": (
                "/forms/formulaire_reclamation_contravention.pdf"
            ),
        },
    ]

    assert local_forms_for_display(forms) == forms


def test_service_link_is_exposed_as_a_separate_action() -> None:
    response = {
        "forms": [],
        "service_links": [{
            "label": "Accéder au service officiel",
            "url": "https://example.test/service",
        }],
    }

    assert service_links_for_display(response) == response["service_links"]
