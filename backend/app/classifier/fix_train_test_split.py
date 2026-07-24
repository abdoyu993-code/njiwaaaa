from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[3]

TRAIN_PATH = PROJECT_ROOT / "backend" / "data" / "train.csv"
TEST_PATH = PROJECT_ROOT / "backend" / "data" / "test.csv"

BACKUP_TRAIN_PATH = PROJECT_ROOT / "backend" / "data" / "train_original.csv"
BACKUP_TEST_PATH = PROJECT_ROOT / "backend" / "data" / "test_original.csv"


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    # Sauvegarder les fichiers d'origine
    train.to_csv(BACKUP_TRAIN_PATH, index=False, encoding="utf-8-sig")
    test.to_csv(BACKUP_TEST_PATH, index=False, encoding="utf-8-sig")

    # Réunir les 15 000 lignes
    full_dataset = pd.concat([train, test], ignore_index=True)

    # Vérifications
    print("Dimensions complètes :", full_dataset.shape)
    print("Nombre d'intentions :", full_dataset["intent"].nunique())
    print("\nNombre d'exemples par intention :")
    print(full_dataset["intent"].value_counts().sort_index())

    # Séparation stratifiée 80/20
    new_train, new_test = train_test_split(
        full_dataset,
        test_size=0.20,
        random_state=42,
        stratify=full_dataset["intent"],
    )

    # Réinitialiser les index
    new_train = new_train.reset_index(drop=True)
    new_test = new_test.reset_index(drop=True)

    # Recréer des identifiants propres
    new_train["id"] = range(1, len(new_train) + 1)
    new_test["id"] = range(1, len(new_test) + 1)

    # Sauvegarder les nouveaux fichiers
    new_train.to_csv(
        TRAIN_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    new_test.to_csv(
        TEST_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nNouveaux fichiers créés.")
    print("Train :", new_train.shape)
    print("Test :", new_test.shape)
    print("Intentions train :", new_train["intent"].nunique())
    print("Intentions test :", new_test["intent"].nunique())

    print("\nIntentions uniquement dans train :")
    print(
        sorted(
            set(new_train["intent"])
            - set(new_test["intent"])
        )
    )

    print("\nIntentions uniquement dans test :")
    print(
        sorted(
            set(new_test["intent"])
            - set(new_train["intent"])
        )
    )


if __name__ == "__main__":
    main()
