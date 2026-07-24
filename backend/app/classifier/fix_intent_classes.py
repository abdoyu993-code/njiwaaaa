from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "backend" / "data"

ORIGINAL_TRAIN_PATH = DATA_DIR / "train_original.csv"
ORIGINAL_TEST_PATH = DATA_DIR / "test_original.csv"

FINAL_TRAIN_PATH = DATA_DIR / "train.csv"
FINAL_TEST_PATH = DATA_DIR / "test.csv"

TARGET_INTENT = "prise_rendez_vous"
REMOVED_INTENT = "horaires_agence"
TARGET_DOMAIN = "contact_reclamation"


APPOINTMENT_TEMPLATES = [
    "Je voudrais prendre un rendez-vous avec une agence NARSA.",
    "Comment puis-je réserver un rendez-vous auprès de la NARSA ?",
    "Je souhaite fixer une date de rendez-vous dans une agence.",
    "Est-il possible de prendre rendez-vous en ligne avec la NARSA ?",
    "Je veux réserver un créneau pour me rendre à la NARSA.",
    "Pouvez-vous m'aider à demander un rendez-vous en agence ?",
    "Je souhaite programmer une visite dans une agence NARSA.",
    "Comment choisir un créneau de rendez-vous à la NARSA ?",
    "Je voudrais obtenir un rendez-vous pour effectuer une démarche.",
    "Puis-je réserver une date pour rencontrer un agent NARSA ?",
    "Je cherche à fixer un rendez-vous pour déposer mon dossier.",
    "Comment prendre rendez-vous pour une démarche administrative ?",
    "Je veux planifier un rendez-vous dans le centre NARSA le plus proche.",
    "Pouvez-vous m'indiquer comment réserver un créneau en agence ?",
    "Je voudrais demander un rendez-vous pour être reçu par un agent.",
    "Comment programmer une rencontre avec un conseiller NARSA ?",
    "Je souhaite prendre rendez-vous pour le suivi de mon dossier.",
    "Est-ce que la prise de rendez-vous peut se faire sur Internet ?",
    "Je veux choisir une date pour me présenter à une agence NARSA.",
    "Comment réserver un rendez-vous pour déposer une demande ?",
    "Je souhaite obtenir un créneau de passage auprès de la NARSA.",
    "Puis-je prendre rendez-vous par téléphone avec une agence NARSA ?",
    "Je voudrais fixer un rendez-vous pour avoir des informations.",
    "Comment réserver une visite dans une agence NARSA ?",
    "Je veux demander un rendez-vous pour finaliser une procédure.",
    "Pouvez-vous programmer un rendez-vous pour ma démarche ?",
    "Je souhaite rencontrer un agent NARSA sur rendez-vous.",
    "Je voudrais réserver un créneau disponible cette semaine.",
    "Comment prendre un rendez-vous pour présenter mes documents ?",
    "Je cherche un créneau pour effectuer une démarche à la NARSA.",
    "Je veux fixer une date afin de me rendre dans une agence.",
    "Puis-je demander un rendez-vous pour obtenir une assistance ?",
    "Je souhaite réserver un passage dans une agence NARSA.",
    "Comment choisir une agence et une date de rendez-vous ?",
    "Je voudrais planifier une visite pour déposer mes documents.",
    "Pouvez-vous m'aider à réserver un rendez-vous administratif ?",
    "Je souhaite prendre rendez-vous avant de me déplacer.",
    "Comment obtenir un créneau pour être reçu à la NARSA ?",
    "Je veux prendre rendez-vous pour poser des questions sur mon dossier.",
    "Puis-je réserver un rendez-vous dans une autre agence NARSA ?",
    "Je voudrais modifier ou choisir une date de rendez-vous.",
    "Comment programmer mon passage auprès d'une agence NARSA ?",
    "Je souhaite demander un rendez-vous pour compléter mon dossier.",
    "Je veux réserver une date pour rencontrer le service concerné.",
    "Pouvez-vous me proposer un créneau de rendez-vous disponible ?",
    "Comment faire une demande de rendez-vous auprès de la NARSA ?",
    "Je souhaite fixer un rendez-vous pour une procédure en cours.",
    "Je voudrais être reçu en agence après avoir pris rendez-vous.",
    "Je veux planifier mon déplacement vers une agence NARSA.",
    "Comment réserver officiellement un rendez-vous avec la NARSA ?",
]


def validate_source_files() -> None:
    """Vérifie que les sauvegardes d'origine existent."""
    missing_files = [
        path
        for path in [ORIGINAL_TRAIN_PATH, ORIGINAL_TEST_PATH]
        if not path.exists()
    ]

    if missing_files:
        formatted_paths = "\n".join(str(path) for path in missing_files)
        raise FileNotFoundError(
            "Les sauvegardes d'origine sont introuvables :\n"
            f"{formatted_paths}"
        )


def create_appointment_examples(start_id: int) -> pd.DataFrame:
    """Crée 50 nouveaux exemples pour l'intention de rendez-vous."""
    if len(APPOINTMENT_TEMPLATES) != 50:
        raise ValueError(
            "La liste doit contenir exactement 50 phrases."
        )

    return pd.DataFrame(
        {
            "id": range(start_id, start_id + len(APPOINTMENT_TEMPLATES)),
            "text": APPOINTMENT_TEMPLATES,
            "domain": TARGET_DOMAIN,
            "intent": TARGET_INTENT,
        }
    )


def main() -> None:
    validate_source_files()

    original_train = pd.read_csv(ORIGINAL_TRAIN_PATH)
    original_test = pd.read_csv(ORIGINAL_TEST_PATH)

    full_dataset = pd.concat(
        [original_train, original_test],
        ignore_index=True,
    )

    print("Dataset original fusionné :", full_dataset.shape)
    print(
        "Nombre initial d'intentions :",
        full_dataset["intent"].nunique(),
    )

    removed_count = int(
        (full_dataset["intent"] == REMOVED_INTENT).sum()
    )

    print(
        f"Exemples à supprimer pour {REMOVED_INTENT} :",
        removed_count,
    )

    # Supprimer la classe non retenue.
    corrected_dataset = full_dataset[
        full_dataset["intent"] != REMOVED_INTENT
    ].copy()

    # Créer 50 exemples supplémentaires pour la classe retenue.
    max_existing_id = pd.to_numeric(
        corrected_dataset["id"],
        errors="coerce",
    ).max()

    start_id = (
        int(max_existing_id) + 1
        if pd.notna(max_existing_id)
        else len(corrected_dataset) + 1
    )

    additional_examples = create_appointment_examples(start_id)

    corrected_dataset = pd.concat(
        [corrected_dataset, additional_examples],
        ignore_index=True,
    )

    # Vérifier les doublons de texte.
    duplicated_texts = corrected_dataset["text"].duplicated().sum()

    if duplicated_texts:
        raise ValueError(
            f"{duplicated_texts} doublons de texte ont été détectés."
        )

    class_counts = corrected_dataset["intent"].value_counts()

    print("\nDataset corrigé :", corrected_dataset.shape)
    print(
        "Nombre final d'intentions :",
        corrected_dataset["intent"].nunique(),
    )
    print(
        "Minimum d'exemples par intention :",
        class_counts.min(),
    )
    print(
        "Maximum d'exemples par intention :",
        class_counts.max(),
    )

    if corrected_dataset.shape != (15000, 4):
        raise ValueError(
            "Le dataset final devrait contenir exactement "
            "15 000 lignes et 4 colonnes."
        )

    if corrected_dataset["intent"].nunique() != 60:
        raise ValueError(
            "Le dataset final devrait contenir exactement "
            "60 intentions."
        )

    if class_counts.min() != 250 or class_counts.max() != 250:
        problematic_classes = class_counts[
            class_counts != 250
        ]

        raise ValueError(
            "Certaines intentions n'ont pas 250 exemples :\n"
            f"{problematic_classes}"
        )

    # Nouvelle séparation stratifiée 80/20.
    new_train, new_test = train_test_split(
        corrected_dataset,
        test_size=0.20,
        random_state=42,
        stratify=corrected_dataset["intent"],
    )

    new_train = new_train.reset_index(drop=True)
    new_test = new_test.reset_index(drop=True)

    # Recréer des identifiants uniques dans chaque fichier.
    new_train["id"] = range(1, len(new_train) + 1)
    new_test["id"] = range(1, len(new_test) + 1)

    new_train.to_csv(
        FINAL_TRAIN_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    new_test.to_csv(
        FINAL_TEST_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    train_counts = new_train["intent"].value_counts()
    test_counts = new_test["intent"].value_counts()

    print("\nSéparation finale terminée")
    print("Train :", new_train.shape)
    print("Test :", new_test.shape)
    print("Intentions train :", new_train["intent"].nunique())
    print("Intentions test :", new_test["intent"].nunique())
    print(
        "Exemples par intention dans train :",
        f"{train_counts.min()} à {train_counts.max()}",
    )
    print(
        "Exemples par intention dans test :",
        f"{test_counts.min()} à {test_counts.max()}",
    )

    train_only = sorted(
        set(new_train["intent"]) - set(new_test["intent"])
    )

    test_only = sorted(
        set(new_test["intent"]) - set(new_train["intent"])
    )

    print("Intentions uniquement dans train :", train_only)
    print("Intentions uniquement dans test :", test_only)


if __name__ == "__main__":
    main()