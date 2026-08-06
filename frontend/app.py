from __future__ import annotations

from typing import Any

import requests
import streamlit as st

if __package__:
    from .rendering import (
        local_forms_for_display,
        previous_context_from_messages,
        service_links_for_display,
        sources_for_display,
    )
else:
    from rendering import (
        local_forms_for_display,
        previous_context_from_messages,
        service_links_for_display,
        sources_for_display,
    )


API_BASE_URL = "http://127.0.0.1:8000"
ASK_API_URL = f"{API_BASE_URL}/ask"


st.set_page_config(
    page_title="Assistant NARSA",
    page_icon="🚗",
    layout="centered",
)


def initialize_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Bonjour 👋\n\n"
                    "Je suis l’assistant intelligent NARSA. "
                    "Posez-moi une question concernant le permis "
                    "de conduire, les infractions, les véhicules, "
                    "les accidents ou la visite technique."
                ),
                "metadata": None,
            }
        ]


def call_api(
    question: str,
    top_k: int = 3,
    context: dict[str, str] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "question": question,
        "top_k": top_k,
    }

    if context:
        payload["context"] = context

    response = requests.post(
        ASK_API_URL,
        json=payload,
        timeout=120,
    )

    response.raise_for_status()
    return response.json()


def display_forms(
    forms: list[dict[str, Any]],
) -> None:
    """
    Affiche des boutons directs vers les PDF servis par FastAPI.
    Streamlit ne télécharge plus le fichier en mémoire.
    """
    local_forms = local_forms_for_display(forms)

    if not local_forms:
        return

    st.markdown("### Formulaires disponibles")

    for form in local_forms:
        title = form.get(
            "title",
            "Formulaire NARSA",
        )

        download_url = form.get(
            "download_url"
        )

        if not download_url:
            continue

        full_url = (
            download_url
            if download_url.startswith(
                ("http://", "https://")
            )
            else f"{API_BASE_URL}{download_url}"
        )

        button_label = form.get(
            "button_label"
        ) or f"📄 Télécharger : {title}"

        st.link_button(
            label=button_label,
            url=full_url,
            use_container_width=True,
        )


def display_metadata(
    response_data: dict[str, Any],
) -> None:
    confidence = response_data.get(
        "confidence",
        0.0,
    )

    domain = response_data.get(
        "domain",
        "inconnu",
    )

    intent = response_data.get(
        "intent",
        "inconnue",
    )

    generated = response_data.get(
        "generated",
        False,
    )

    metadata = response_data.get(
        "metadata",
        {},
    )

    attempts = metadata.get(
        "generation_attempts",
        0,
    )

    used_contexts_count = metadata.get(
        "used_contexts_count",
        0,
    )

    with st.expander(
        "Sources et détails techniques",
        expanded=False,
    ):
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Confiance",
                f"{confidence * 100:.1f} %",
            )

            st.write(
                f"**Domaine :** `{domain}`"
            )

        with col2:
            st.metric(
                "Tentatives Gemini",
                attempts,
            )

            st.write(
                f"**Intention :** `{intent}`"
            )

        st.write(
            "**Réponse générée :**",
            "Oui" if generated else "Non",
        )

        st.write(
            "**Contextes utilisés :**",
            used_contexts_count,
        )

        finish_reason = response_data.get(
            "finish_reason"
        )

        if finish_reason:
            st.write(
                f"**Fin Gemini :** `{finish_reason}`"
            )

        sources = sources_for_display(
            response_data=response_data,
            technical_details_enabled=True,
        )
        if sources:
            st.markdown(
                "**Sources techniques :**"
            )

            for source in sources:
                title = (
                    source.get("title")
                    or "Document NARSA"
                )

                filename = source.get(
                    "filename",
                    "",
                )

                score = source.get(
                    "best_similarity_score"
                )

                st.markdown(
                    f"- **{title}**"
                )

                if filename:
                    st.caption(
                        f"Fichier : {filename}"
                    )

                if score is not None:
                    st.caption(
                        "Score de similarité : "
                        f"{score:.4f}"
                    )

        timing_labels = (
            ("Classification", "classification_time_ms"),
            ("Recherche", "retrieval_time_ms"),
            ("Génération", "generation_time_ms"),
            ("Total", "total_time_ms"),
        )

        st.markdown("**Temps de traitement :**")

        for label, key in timing_labels:
            duration = metadata.get(key, 0.0)
            st.caption(
                f"{label} : {duration:.2f} ms"
            )


def display_service_links(
    response_data: dict[str, Any],
) -> None:
    service_links = service_links_for_display(response_data)

    if not service_links:
        return

    st.markdown("### Services officiels")

    for service_link in service_links:
        st.link_button(
            label=service_link["label"],
            url=service_link["url"],
            use_container_width=True,
        )


def append_assistant_message(
    content: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": content,
            "metadata": metadata,
        }
    )


initialize_session_state()


st.title("🚗 Assistant intelligent NARSA")

st.caption(
    "Assistant basé sur la classification d’intentions, "
    "le RAG et Gemini."
)


with st.sidebar:
    st.header("Paramètres")

    top_k = st.slider(
        "Nombre de chunks recherchés",
        min_value=1,
        max_value=10,
        value=3,
    )

    show_technical_details = st.checkbox(
        "Afficher les détails techniques",
        value=False,
    )

    st.divider()

    if st.button(
        "Effacer la conversation",
        use_container_width=True,
    ):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Bonjour 👋\n\n"
                    "Je suis l’assistant intelligent NARSA. "
                    "Posez-moi une question concernant le permis "
                    "de conduire, les infractions, les véhicules, "
                    "les accidents ou la visite technique."
                ),
                "metadata": None,
            }
        ]

        st.rerun()

    st.info(
        "L’API FastAPI doit rester active sur "
        "http://127.0.0.1:8000"
    )


for message in st.session_state.messages:
    with st.chat_message(
        message["role"]
    ):
        st.markdown(
            message["content"]
        )

        metadata = message.get(
            "metadata"
        )

        if (
            message["role"] == "assistant"
            and metadata
        ):
            display_forms(
                metadata.get(
                    "forms",
                    [],
                )
            )
            display_service_links(metadata)

            if show_technical_details:
                display_metadata(
                    metadata
                )


question = st.chat_input(
    "Posez votre question à la NARSA..."
)


if question:
    previous_context = previous_context_from_messages(
        st.session_state.messages
    )

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
            "metadata": None,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner(
            "Recherche dans les documents NARSA..."
        ):
            try:
                response_data = call_api(
                    question=question,
                    top_k=top_k,
                    context=previous_context,
                )

                answer = response_data.get(
                    "answer",
                    "Aucune réponse reçue.",
                )

                st.markdown(answer)

                display_forms(
                    response_data.get(
                        "forms",
                        [],
                    )
                )
                display_service_links(response_data)

                if show_technical_details:
                    display_metadata(
                        response_data
                    )

                append_assistant_message(
                    content=answer,
                    metadata=response_data,
                )

            except requests.ConnectionError:
                error_message = (
                    "Impossible de se connecter à l’API. "
                    "Vérifiez que Uvicorn est actif sur "
                    "http://127.0.0.1:8000."
                )

                st.error(error_message)

                append_assistant_message(
                    content=error_message,
                )

            except requests.Timeout:
                error_message = (
                    "La requête a pris trop de temps. "
                    "Veuillez réessayer."
                )

                st.error(error_message)

                append_assistant_message(
                    content=error_message,
                )

            except requests.HTTPError as error:
                error_message = (
                    "L’API a retourné une erreur : "
                    f"{error}"
                )

                st.error(error_message)

                append_assistant_message(
                    content=error_message,
                )

            except Exception as error:
                error_message = (
                    "Une erreur inattendue est survenue : "
                    f"{error}"
                )

                st.error(error_message)

                append_assistant_message(
                    content=error_message,
                )
