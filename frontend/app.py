from __future__ import annotations

from typing import Any

import requests
import streamlit as st


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
    top_k: int = 4,
) -> dict[str, Any]:
    response = requests.post(
        ASK_API_URL,
        json={
            "question": question,
            "top_k": top_k,
        },
        timeout=120,
    )

    response.raise_for_status()
    return response.json()


def display_sources(
    sources: list[dict[str, Any]],
) -> None:
    if not sources:
        return

    displayed_titles: set[str] = set()

    with st.expander(
        "Documents de référence",
        expanded=False,
    ):
        for source in sources:
            title = (
                source.get("title")
                or "Document NARSA"
            )

            if title in displayed_titles:
                continue

            displayed_titles.add(title)
            st.markdown(f"- {title}")


def display_forms(
    forms: list[dict[str, Any]],
) -> None:
    """
    Affiche des boutons directs vers les PDF servis par FastAPI.
    Streamlit ne télécharge plus le fichier en mémoire.
    """
    if not forms:
        return

    st.markdown("### Formulaires disponibles")

    for form in forms:
        title = form.get(
            "title",
            "Formulaire NARSA",
        )

        download_url = form.get(
            "download_url"
        )

        if not download_url:
            continue

        full_url = f"{API_BASE_URL}{download_url}"

        st.link_button(
            label=f"📄 Télécharger : {title}",
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
        "Détails techniques",
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

        sources = response_data.get(
            "sources",
            [],
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
        value=4,
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
            display_sources(
                metadata.get(
                    "sources",
                    [],
                )
            )

            display_forms(
                metadata.get(
                    "forms",
                    [],
                )
            )

            if show_technical_details:
                display_metadata(
                    metadata
                )


question = st.chat_input(
    "Posez votre question à la NARSA..."
)


if question:
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
                )

                answer = response_data.get(
                    "answer",
                    "Aucune réponse reçue.",
                )

                st.markdown(answer)

                display_sources(
                    response_data.get(
                        "sources",
                        [],
                    )
                )

                display_forms(
                    response_data.get(
                        "forms",
                        [],
                    )
                )

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