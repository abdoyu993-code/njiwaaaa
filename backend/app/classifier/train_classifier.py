from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[3]

TRAIN_PATH = PROJECT_ROOT / "backend" / "data" / "train.csv"
TEST_PATH = PROJECT_ROOT / "backend" / "data" / "test.csv"

MODELS_DIR = PROJECT_ROOT / "backend" / "models"
RESULTS_DIR = PROJECT_ROOT / "reports" / "results"

CLASSIFIER_PATH = MODELS_DIR / "intent_classifier.joblib"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"
INTENT_DOMAIN_PATH = MODELS_DIR / "intent_to_domain.json"
METADATA_PATH = MODELS_DIR / "classifier_metadata.json"
MODEL_NAME_PATH = MODELS_DIR / "sentence_transformer_model.txt"
TRAINING_SUMMARY_PATH = RESULTS_DIR / "training_summary.json"

TRAIN_EMBEDDINGS_PATH = MODELS_DIR / "train_embeddings.npy"
TEST_EMBEDDINGS_PATH = MODELS_DIR / "test_embeddings.npy"

MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

EXPECTED_COLUMNS = {"id", "text", "domain", "intent"}
EXPECTED_INTENT_COUNT = 60
EXPECTED_DOMAIN_COUNT = 7

BATCH_SIZE = 64
RANDOM_STATE = 42


def create_directories() -> None:
    """Crée les dossiers de sortie nécessaires."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(path: Path, name: str) -> pd.DataFrame:
    """Charge et valide un fichier CSV."""
    if not path.exists():
        raise FileNotFoundError(
            f"Le fichier {name} est introuvable : {path}"
        )

    dataframe = pd.read_csv(path)

    missing_columns = EXPECTED_COLUMNS - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Colonnes manquantes dans {name} : "
            f"{sorted(missing_columns)}"
        )

    if dataframe[list(EXPECTED_COLUMNS)].isna().any().any():
        raise ValueError(
            f"Des valeurs manquantes sont présentes dans {name}."
        )

    dataframe = dataframe.copy()

    for column in ["text", "domain", "intent"]:
        dataframe[column] = (
            dataframe[column]
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )

    print(f"{name} chargé : {dataframe.shape}")

    return dataframe


def validate_datasets(
    train_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
) -> None:
    """Vérifie la cohérence globale des datasets."""
    train_intents = set(train_dataframe["intent"].unique())
    test_intents = set(test_dataframe["intent"].unique())

    train_domains = set(train_dataframe["domain"].unique())
    test_domains = set(test_dataframe["domain"].unique())

    if len(train_intents) != EXPECTED_INTENT_COUNT:
        raise ValueError(
            "Le train doit contenir exactement "
            f"{EXPECTED_INTENT_COUNT} intentions, "
            f"mais il en contient {len(train_intents)}."
        )

    if len(test_intents) != EXPECTED_INTENT_COUNT:
        raise ValueError(
            "Le test doit contenir exactement "
            f"{EXPECTED_INTENT_COUNT} intentions, "
            f"mais il en contient {len(test_intents)}."
        )

    if len(train_domains) != EXPECTED_DOMAIN_COUNT:
        raise ValueError(
            "Le train doit contenir exactement "
            f"{EXPECTED_DOMAIN_COUNT} domaines."
        )

    if train_intents != test_intents:
        raise ValueError(
            "Les intentions du train et du test sont différentes."
        )

    if train_domains != test_domains:
        raise ValueError(
            "Les domaines du train et du test sont différents."
        )

    overlap = set(train_dataframe["text"]) & set(
        test_dataframe["text"]
    )

    if overlap:
        raise ValueError(
            f"{len(overlap)} textes sont présents à la fois "
            "dans le train et dans le test."
        )

    print("Validation des datasets : correcte")
    print(f"Intentions : {len(train_intents)}")
    print(f"Domaines : {len(train_domains)}")


def build_intent_to_domain_mapping(
    train_dataframe: pd.DataFrame,
) -> dict[str, str]:
    """
    Crée le dictionnaire intent → domain.

    Une intention doit appartenir à un seul domaine.
    """
    domain_counts = (
        train_dataframe.groupby("intent")["domain"]
        .nunique()
    )

    inconsistent_intents = domain_counts[
        domain_counts > 1
    ]

    if not inconsistent_intents.empty:
        raise ValueError(
            "Certaines intentions sont associées à plusieurs domaines :\n"
            f"{inconsistent_intents}"
        )

    mapping = (
        train_dataframe[
            ["intent", "domain"]
        ]
        .drop_duplicates()
        .set_index("intent")["domain"]
        .sort_index()
        .to_dict()
    )

    if len(mapping) != EXPECTED_INTENT_COUNT:
        raise ValueError(
            "Le mapping intent → domain ne contient pas "
            f"{EXPECTED_INTENT_COUNT} entrées."
        )

    return mapping


def load_embedding_model() -> SentenceTransformer:
    """Charge le modèle Sentence-BERT."""
    print("\nChargement du modèle Sentence-BERT...")
    print(f"Modèle : {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    embedding_dimension = model.get_sentence_embedding_dimension()

    print(
        "Dimension des embeddings :",
        embedding_dimension,
    )

    return model


def encode_texts(
    model: SentenceTransformer,
    texts: list[str],
    dataset_name: str,
) -> np.ndarray:
    """Transforme une liste de textes en embeddings."""
    print(
        f"\nCréation des embeddings pour {dataset_name}..."
    )

    start_time = time.perf_counter()

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    elapsed_time = time.perf_counter() - start_time

    print(
        f"Embeddings {dataset_name} créés : "
        f"{embeddings.shape}"
    )
    print(
        f"Temps d'encodage {dataset_name} : "
        f"{elapsed_time:.2f} secondes"
    )

    return embeddings


def train_classifier(
    train_embeddings: np.ndarray,
    encoded_labels: np.ndarray,
) -> LogisticRegression:
    """Entraîne la Logistic Regression."""
    print("\nEntraînement de la Logistic Regression...")

    classifier = LogisticRegression(
        C=3.0,
        solver="lbfgs",
        max_iter=2000,
        random_state=RANDOM_STATE,
        n_jobs=None,
        verbose=0,
    )

    start_time = time.perf_counter()

    classifier.fit(
        train_embeddings,
        encoded_labels,
    )

    elapsed_time = time.perf_counter() - start_time

    print(
        f"Entraînement terminé en "
        f"{elapsed_time:.2f} secondes"
    )

    print(
        "Nombre d'itérations maximal observé :",
        int(classifier.n_iter_.max()),
    )

    return classifier


def evaluate_quickly(
    classifier: LogisticRegression,
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
) -> dict[str, float]:
    """
    Effectue une première évaluation rapide.

    L'évaluation détaillée sera réalisée à l'étape suivante.
    """
    train_predictions = classifier.predict(
        train_embeddings
    )

    test_predictions = classifier.predict(
        test_embeddings
    )

    metrics = {
        "train_accuracy": float(
            accuracy_score(
                train_labels,
                train_predictions,
            )
        ),
        "train_macro_f1": float(
            f1_score(
                train_labels,
                train_predictions,
                average="macro",
            )
        ),
        "test_accuracy": float(
            accuracy_score(
                test_labels,
                test_predictions,
            )
        ),
        "test_macro_f1": float(
            f1_score(
                test_labels,
                test_predictions,
                average="macro",
            )
        ),
    }

    print("\nPremière évaluation")
    print(
        f"Train accuracy : "
        f"{metrics['train_accuracy']:.4f}"
    )
    print(
        f"Train macro F1 : "
        f"{metrics['train_macro_f1']:.4f}"
    )
    print(
        f"Test accuracy : "
        f"{metrics['test_accuracy']:.4f}"
    )
    print(
        f"Test macro F1 : "
        f"{metrics['test_macro_f1']:.4f}"
    )

    return metrics


def save_artifacts(
    classifier: LogisticRegression,
    label_encoder: LabelEncoder,
    intent_to_domain: dict[str, str],
    embedding_dimension: int,
    metrics: dict[str, float],
    train_shape: tuple[int, int],
    test_shape: tuple[int, int],
) -> None:
    """Sauvegarde les modèles et métadonnées."""
    joblib.dump(
        classifier,
        CLASSIFIER_PATH,
    )

    joblib.dump(
        label_encoder,
        LABEL_ENCODER_PATH,
    )

    with INTENT_DOMAIN_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            intent_to_domain,
            file,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )

    with MODEL_NAME_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(MODEL_NAME)

    metadata = {
        "model_name": MODEL_NAME,
        "embedding_dimension": embedding_dimension,
        "classifier": "LogisticRegression",
        "classifier_parameters": {
            "C": classifier.C,
            "solver": classifier.solver,
            "max_iter": classifier.max_iter,
            "random_state": RANDOM_STATE,
        },
        "number_of_intents": int(
            len(label_encoder.classes_)
        ),
        "number_of_domains": int(
            len(set(intent_to_domain.values()))
        ),
        "classes": label_encoder.classes_.tolist(),
        "metrics": metrics,
        "train_shape": list(train_shape),
        "test_shape": list(test_shape),
    }

    with METADATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )

    with TRAINING_SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\nFichiers sauvegardés")
    print(f"- {CLASSIFIER_PATH}")
    print(f"- {LABEL_ENCODER_PATH}")
    print(f"- {INTENT_DOMAIN_PATH}")
    print(f"- {METADATA_PATH}")
    print(f"- {MODEL_NAME_PATH}")
    print(f"- {TRAINING_SUMMARY_PATH}")


def main() -> None:
    """Point d'entrée principal."""
    create_directories()

    print("=" * 70)
    print("ENTRAÎNEMENT DU CLASSIFIEUR D'INTENTIONS")
    print("=" * 70)

    train_dataframe = load_dataset(
        TRAIN_PATH,
        "train",
    )

    test_dataframe = load_dataset(
        TEST_PATH,
        "test",
    )

    validate_datasets(
        train_dataframe,
        test_dataframe,
    )

    intent_to_domain = build_intent_to_domain_mapping(
        train_dataframe
    )

    label_encoder = LabelEncoder()

    train_labels = label_encoder.fit_transform(
        train_dataframe["intent"]
    )

    test_labels = label_encoder.transform(
        test_dataframe["intent"]
    )

    print(
        "\nNombre de classes encodées :",
        len(label_encoder.classes_),
    )

    model = load_embedding_model()

    train_embeddings = encode_texts(
        model,
        train_dataframe["text"].tolist(),
        "train",
    )

    test_embeddings = encode_texts(
        model,
        test_dataframe["text"].tolist(),
        "test",
    )

    np.save(
        TRAIN_EMBEDDINGS_PATH,
        train_embeddings,
    )

    np.save(
        TEST_EMBEDDINGS_PATH,
        test_embeddings,
    )

    print("\nEmbeddings sauvegardés")
    print(f"- {TRAIN_EMBEDDINGS_PATH}")
    print(f"- {TEST_EMBEDDINGS_PATH}")

    classifier = train_classifier(
        train_embeddings,
        train_labels,
    )

    metrics = evaluate_quickly(
        classifier,
        train_embeddings,
        test_embeddings,
        train_labels,
        test_labels,
    )

    embedding_dimension = (
        model.get_sentence_embedding_dimension()
    )

    if embedding_dimension is None:
        embedding_dimension = int(
            train_embeddings.shape[1]
        )

    save_artifacts(
        classifier=classifier,
        label_encoder=label_encoder,
        intent_to_domain=intent_to_domain,
        embedding_dimension=int(
            embedding_dimension
        ),
        metrics=metrics,
        train_shape=train_embeddings.shape,
        test_shape=test_embeddings.shape,
    )

    print("\n" + "=" * 70)
    print("ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS")
    print("=" * 70)


if __name__ == "__main__":
    main()