from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DOCUMENTS_DIR = PROJECT_ROOT / "backend" / "data" / "documents"
CHROMA_DIR = PROJECT_ROOT / "backend" / "data" / "chroma_db"

COLLECTION_NAME = "narsa_documents"

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


def normalize_text(text: str) -> str:
    """
    Nettoie le texte tout en conservant la structure Markdown.
    """
    text = text.replace("\ufeff", "")
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Découpe un document en chunks avec chevauchement.
    """
    text = normalize_text(text)

    if not text:
        return []

    chunks: list[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(
            start + chunk_size,
            text_length,
        )

        if end < text_length:
            search_start = start + int(
                chunk_size * 0.60
            )

            paragraph_break = text.rfind(
                "\n\n",
                search_start,
                end,
            )

            sentence_break = text.rfind(
                ". ",
                search_start,
                end,
            )

            heading_break = text.rfind(
                "\n## ",
                search_start,
                end,
            )

            best_break = max(
                paragraph_break,
                sentence_break,
                heading_break,
            )

            if best_break > start:
                end = best_break + 1

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        next_start = end - overlap

        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


def get_domain(file_path: Path) -> str:
    """
    Récupère le domaine à partir du premier dossier
    situé sous documents/.
    """
    relative_path = file_path.relative_to(
        DOCUMENTS_DIR
    )

    if len(relative_path.parts) < 2:
        return "inconnu"

    return relative_path.parts[0]


def extract_title(
    text: str,
    fallback: str,
) -> str:
    """
    Extrait le premier titre Markdown de niveau 1.
    """
    for line in text.splitlines():
        line = line.strip()

        if line.startswith("# "):
            return line.removeprefix("# ").strip()

    return fallback


def extract_intents(text: str) -> list[str]:
    """
    Extrait les intentions indiquées dans la partie :

    - Intentions concernées :
      - permis_perte
      - permis_vol
    """
    intents: list[str] = []
    reading_intents = False

    for raw_line in text.splitlines():
        stripped_line = raw_line.strip()

        normalized_line = (
            stripped_line
            .lower()
            .replace("é", "e")
            .replace("è", "e")
            .replace("ê", "e")
        )

        if normalized_line.startswith(
            "- intentions concernees"
        ):
            reading_intents = True
            continue

        if not reading_intents:
            continue

        if stripped_line.startswith("#"):
            break

        if not stripped_line:
            continue

        if (
            stripped_line.startswith("- ")
            and ":" in stripped_line
        ):
            break

        if stripped_line.startswith("-"):
            candidate = stripped_line.lstrip("-").strip()

            if re.fullmatch(
                r"[a-z0-9_]+",
                candidate,
            ):
                intents.append(candidate)
                continue

        if intents:
            break

    return list(dict.fromkeys(intents))


def create_chunk_id(
    relative_path: str,
    chunk_index: int,
    chunk_text: str,
) -> str:
    """
    Crée un identifiant stable pour chaque chunk.
    """
    raw_value = (
        f"{relative_path}:"
        f"{chunk_index}:"
        f"{chunk_text}"
    )

    return hashlib.sha256(
        raw_value.encode("utf-8")
    ).hexdigest()


def load_markdown_documents() -> list[dict[str, Any]]:
    """
    Charge tous les fichiers Markdown.
    """
    documents: list[dict[str, Any]] = []

    markdown_files = sorted(
        DOCUMENTS_DIR.rglob("*.md")
    )

    for file_path in markdown_files:
        try:
            text = file_path.read_text(
                encoding="utf-8"
            )

        except UnicodeDecodeError:
            text = file_path.read_text(
                encoding="utf-8",
                errors="replace",
            )

        text = normalize_text(text)

        if not text:
            print(
                f"[IGNORÉ] Fichier vide : {file_path}"
            )
            continue

        relative_path = file_path.relative_to(
            DOCUMENTS_DIR
        )

        domain = get_domain(file_path)

        title = extract_title(
            text=text,
            fallback=file_path.stem,
        )

        intents = extract_intents(text)

        documents.append(
            {
                "path": file_path,
                "relative_path": str(
                    relative_path
                ).replace("\\", "/"),
                "filename": file_path.name,
                "domain": domain,
                "title": title,
                "intents": intents,
                "text": text,
            }
        )

    return documents


def reset_chroma_database() -> None:
    """
    Supprime l'ancien index afin d'éviter les doublons.
    """
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    CHROMA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def ingest_documents() -> None:
    if not DOCUMENTS_DIR.exists():
        raise FileNotFoundError(
            "Dossier des documents introuvable : "
            f"{DOCUMENTS_DIR}"
        )

    print("Chargement du modèle d'embedding...")

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    source_documents = load_markdown_documents()

    if not source_documents:
        raise RuntimeError(
            "Aucun fichier Markdown trouvé dans "
            f"{DOCUMENTS_DIR}"
        )

    print(
        "Documents Markdown trouvés : "
        f"{len(source_documents)}"
    )

    documents_without_intents = [
        document["relative_path"]
        for document in source_documents
        if not document["intents"]
    ]

    if documents_without_intents:
        print(
            "\n[ATTENTION] Documents sans intention détectée :"
        )

        for relative_path in documents_without_intents:
            print(f"  - {relative_path}")

    reset_chroma_database()

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": (
                "Documents officiels et informatifs NARSA"
            ),
            "embedding_model": EMBEDDING_MODEL_NAME,
            "hnsw:space": "cosine",
        },
    )

    all_ids: list[str] = []

    # Texte réel qui sera retourné par le retriever.
    all_documents: list[str] = []

    # Texte enrichi utilisé uniquement pour calculer l'embedding.
    all_embedding_texts: list[str] = []

    all_metadatas: list[dict[str, Any]] = []

    domain_counts: dict[str, int] = {}
    intent_counts: dict[str, int] = {}

    for document in source_documents:
        chunks = split_text(
            document["text"]
        )

        domain = document["domain"]
        intents = document["intents"]

        intents_text = ", ".join(intents)

        domain_counts.setdefault(domain, 0)
        domain_counts[domain] += len(chunks)

        for intent in intents:
            intent_counts.setdefault(intent, 0)
            intent_counts[intent] += len(chunks)

        for chunk_index, chunk in enumerate(chunks):
            chunk_id = create_chunk_id(
                relative_path=document[
                    "relative_path"
                ],
                chunk_index=chunk_index,
                chunk_text=chunk,
            )

            embedding_text = (
                f"Titre : {document['title']}\n"
                f"Domaine : {domain}\n"
                f"Intentions : {intents_text}\n"
                f"Fichier : {document['filename']}\n\n"
                f"{chunk}"
            )

            metadata = {
                "domain": domain,
                "title": document["title"],
                "filename": document["filename"],
                "relative_path": document[
                    "relative_path"
                ],
                "chunk_index": chunk_index,
                "document_type": "markdown",

                # ChromaDB stocke ici une chaîne,
                # pas directement une liste Python.
                "intents": intents_text,
            }

            all_ids.append(chunk_id)
            all_documents.append(chunk)
            all_embedding_texts.append(
                embedding_text
            )
            all_metadatas.append(metadata)

    print(
        f"Chunks générés : {len(all_documents)}"
    )

    print("Génération des embeddings...")

    embeddings = embedding_model.encode(
        all_embedding_texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    collection.add(
        ids=all_ids,
        documents=all_documents,
        embeddings=embeddings.tolist(),
        metadatas=all_metadatas,
    )

    print("\nIndexation terminée.")

    print(
        f"Base ChromaDB : {CHROMA_DIR}"
    )

    print(
        f"Collection : {COLLECTION_NAME}"
    )

    print(
        "Nombre total de chunks : "
        f"{collection.count()}"
    )

    print("\nChunks par domaine :")

    for domain, count in sorted(
        domain_counts.items()
    ):
        print(f"  - {domain}: {count}")

    print("\nIntentions détectées :")

    for intent, count in sorted(
        intent_counts.items()
    ):
        print(f"  - {intent}: {count}")


if __name__ == "__main__":
    ingest_documents()