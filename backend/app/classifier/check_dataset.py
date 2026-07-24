from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[3]

TRAIN_PATH = PROJECT_ROOT / "backend" / "data" / "train.csv"
TEST_PATH = PROJECT_ROOT / "backend" / "data" / "test.csv"

RESULTS_DIR = PROJECT_ROOT / "reports" / "results"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

EXPECTED_COLUMNS = ["id", "text", "domain", "intent"]

EXPECTED_DOMAINS = {
    "permis",
    "vehicule",
    "visite_technique",
    "infractions",
    "prevention",
    "accident",
    "contact_reclamation",
}


def create_output_directories() -> None:
    """Crée les dossiers de résultats s'ils n'existent pas."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(path: Path, dataset_name: str) -> pd.DataFrame:
    """Charge un fichier CSV et vérifie son existence."""
    if not path.exists():
        raise FileNotFoundError(
            f"Le fichier {dataset_name} est introuvable : {path}"
        )

    dataframe = pd.read_csv(path)

    print(f"\n{'=' * 70}")
    print(f"Chargement de {dataset_name}")
    print(f"{'=' * 70}")
    print(f"Chemin : {path}")
    print(f"Dimensions : {dataframe.shape}")

    return dataframe


def validate_columns(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> None:
    """Vérifie que les colonnes attendues sont présentes."""
    current_columns = dataframe.columns.tolist()

    print(f"\nColonnes de {dataset_name} : {current_columns}")

    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in current_columns
    ]

    extra_columns = [
        column
        for column in current_columns
        if column not in EXPECTED_COLUMNS
    ]

    if missing_columns:
        raise ValueError(
            f"Colonnes manquantes dans {dataset_name} : "
            f"{missing_columns}"
        )

    if extra_columns:
        print(
            f"Attention : colonnes supplémentaires dans "
            f"{dataset_name} : {extra_columns}"
        )
    else:
        print("Structure des colonnes : correcte")


def clean_text_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie légèrement les colonnes textuelles.

    Cette fonction :
    - conserve les valeurs manquantes ;
    - enlève les espaces au début et à la fin ;
    - remplace les suites d'espaces internes par un espace simple.
    """
    cleaned = dataframe.copy()

    for column in ["text", "domain", "intent"]:
        cleaned[column] = cleaned[column].apply(
            lambda value: (
                " ".join(str(value).strip().split())
                if pd.notna(value)
                else value
            )
        )

    return cleaned


def analyze_missing_values(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """Calcule le nombre et le pourcentage de valeurs manquantes."""
    missing_count = dataframe.isna().sum()
    missing_rate = dataframe.isna().mean() * 100

    report = pd.DataFrame(
        {
            "column": dataframe.columns,
            "missing_count": missing_count.values,
            "missing_rate_percent": missing_rate.values,
        }
    )

    print(f"\nValeurs manquantes — {dataset_name}")
    print(report.to_string(index=False))

    report.to_csv(
        RESULTS_DIR / f"{dataset_name}_missing_values.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return report


def analyze_duplicates(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> dict[str, int]:
    """Compte plusieurs types de doublons."""
    full_duplicates = int(dataframe.duplicated().sum())
    id_duplicates = int(dataframe["id"].duplicated().sum())
    text_duplicates = int(dataframe["text"].duplicated().sum())

    text_intent_duplicates = int(
        dataframe.duplicated(subset=["text", "intent"]).sum()
    )

    results = {
        "full_duplicates": full_duplicates,
        "id_duplicates": id_duplicates,
        "text_duplicates": text_duplicates,
        "text_intent_duplicates": text_intent_duplicates,
    }

    print(f"\nDoublons — {dataset_name}")
    for key, value in results.items():
        print(f"{key}: {value}")

    pd.DataFrame([results]).to_csv(
        RESULTS_DIR / f"{dataset_name}_duplicates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    duplicated_texts = dataframe[
        dataframe.duplicated(subset=["text"], keep=False)
    ].sort_values("text")

    duplicated_texts.to_csv(
        RESULTS_DIR / f"{dataset_name}_duplicated_texts.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return results


def analyze_domains(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """Analyse la répartition des domaines."""
    distribution = (
        dataframe["domain"]
        .value_counts(dropna=False)
        .rename_axis("domain")
        .reset_index(name="count")
    )

    distribution["percentage"] = (
        distribution["count"] / len(dataframe) * 100
    )

    print(f"\nDistribution des domaines — {dataset_name}")
    print(distribution.to_string(index=False))

    distribution.to_csv(
        RESULTS_DIR / f"{dataset_name}_domain_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )

    observed_domains = set(
        dataframe["domain"].dropna().astype(str).unique()
    )

    missing_expected = EXPECTED_DOMAINS - observed_domains
    unexpected = observed_domains - EXPECTED_DOMAINS

    print(f"\nNombre de domaines : {len(observed_domains)}")

    if missing_expected:
        print(
            "Domaines attendus mais absents : "
            f"{sorted(missing_expected)}"
        )

    if unexpected:
        print(
            "Domaines inattendus : "
            f"{sorted(unexpected)}"
        )

    if not missing_expected and not unexpected:
        print("Les 7 domaines attendus sont présents.")

    return distribution


def analyze_intents(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """Analyse la répartition des intentions."""
    distribution = (
        dataframe["intent"]
        .value_counts(dropna=False)
        .rename_axis("intent")
        .reset_index(name="count")
    )

    distribution["percentage"] = (
        distribution["count"] / len(dataframe) * 100
    )

    print(f"\nNombre d'intentions — {dataset_name}")
    print(dataframe["intent"].nunique(dropna=True))

    print(f"\nDistribution des intentions — {dataset_name}")
    print(distribution.to_string(index=False))

    distribution.to_csv(
        RESULTS_DIR / f"{dataset_name}_intent_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return distribution


def analyze_text_lengths(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """Calcule les statistiques de longueur des textes."""
    text_series = dataframe["text"].fillna("").astype(str)

    character_lengths = text_series.str.len()
    word_lengths = text_series.str.split().str.len()

    report = pd.DataFrame(
        {
            "metric": [
                "count",
                "mean_characters",
                "median_characters",
                "min_characters",
                "max_characters",
                "mean_words",
                "median_words",
                "min_words",
                "max_words",
            ],
            "value": [
                len(dataframe),
                character_lengths.mean(),
                character_lengths.median(),
                character_lengths.min(),
                character_lengths.max(),
                word_lengths.mean(),
                word_lengths.median(),
                word_lengths.min(),
                word_lengths.max(),
            ],
        }
    )

    print(f"\nLongueur des textes — {dataset_name}")
    print(report.to_string(index=False))

    report.to_csv(
        RESULTS_DIR / f"{dataset_name}_text_length_statistics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    enriched = dataframe.copy()
    enriched["text_length_characters"] = character_lengths
    enriched["text_length_words"] = word_lengths

    enriched.to_csv(
        RESULTS_DIR / f"{dataset_name}_with_text_lengths.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return report


def analyze_intent_domain_consistency(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Vérifie si une même intention est associée à plusieurs domaines.

    Une intention devrait normalement appartenir à un seul domaine.
    """
    mapping = (
        dataframe.groupby("intent")["domain"]
        .agg(
            domain_count="nunique",
            domains=lambda values: " | ".join(
                sorted(set(values.dropna().astype(str)))
            ),
        )
        .reset_index()
    )

    inconsistent = mapping[mapping["domain_count"] > 1]

    print(
        f"\nCohérence intention → domaine — {dataset_name}"
    )
    print(
        f"Intentions associées à plusieurs domaines : "
        f"{len(inconsistent)}"
    )

    if not inconsistent.empty:
        print(inconsistent.to_string(index=False))
    else:
        print(
            "Chaque intention est associée à un seul domaine."
        )

    mapping.to_csv(
        RESULTS_DIR / f"{dataset_name}_intent_domain_mapping.csv",
        index=False,
        encoding="utf-8-sig",
    )

    inconsistent.to_csv(
        RESULTS_DIR
        / f"{dataset_name}_inconsistent_intent_domain.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return mapping


def analyze_conflicting_labels(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Cherche les textes identiques associés à plusieurs intentions.

    Exemple problématique :
    même question → intent_A
    même question → intent_B
    """
    conflicts = (
        dataframe.groupby("text")
        .agg(
            intent_count=("intent", "nunique"),
            intents=(
                "intent",
                lambda values: " | ".join(
                    sorted(set(values.dropna().astype(str)))
                ),
            ),
            domain_count=("domain", "nunique"),
            domains=(
                "domain",
                lambda values: " | ".join(
                    sorted(set(values.dropna().astype(str)))
                ),
            ),
        )
        .reset_index()
    )

    conflicts = conflicts[
        (conflicts["intent_count"] > 1)
        | (conflicts["domain_count"] > 1)
    ]

    print(f"\nConflits d'étiquettes — {dataset_name}")
    print(
        f"Textes identiques avec plusieurs labels : "
        f"{len(conflicts)}"
    )

    conflicts.to_csv(
        RESULTS_DIR / f"{dataset_name}_label_conflicts.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return conflicts


def compare_train_and_test(
    train_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
) -> None:
    """Compare les classes et les textes entre train et test."""
    train_intents = set(train_dataframe["intent"].dropna().unique())
    test_intents = set(test_dataframe["intent"].dropna().unique())

    train_domains = set(train_dataframe["domain"].dropna().unique())
    test_domains = set(test_dataframe["domain"].dropna().unique())

    intents_only_train = sorted(train_intents - test_intents)
    intents_only_test = sorted(test_intents - train_intents)

    domains_only_train = sorted(train_domains - test_domains)
    domains_only_test = sorted(test_domains - train_domains)

    train_texts = set(
        train_dataframe["text"].dropna().astype(str)
    )
    test_texts = set(
        test_dataframe["text"].dropna().astype(str)
    )

    overlapping_texts = sorted(train_texts & test_texts)

    print(f"\n{'=' * 70}")
    print("Comparaison train / test")
    print(f"{'=' * 70}")

    print(f"Intentions uniquement dans train : {intents_only_train}")
    print(f"Intentions uniquement dans test : {intents_only_test}")
    print(f"Domaines uniquement dans train : {domains_only_train}")
    print(f"Domaines uniquement dans test : {domains_only_test}")
    print(
        f"Textes identiques présents dans train et test : "
        f"{len(overlapping_texts)}"
    )

    comparison = pd.DataFrame(
        {
            "indicator": [
                "train_rows",
                "test_rows",
                "train_intents",
                "test_intents",
                "train_domains",
                "test_domains",
                "intents_only_train",
                "intents_only_test",
                "domains_only_train",
                "domains_only_test",
                "overlapping_texts",
            ],
            "value": [
                len(train_dataframe),
                len(test_dataframe),
                len(train_intents),
                len(test_intents),
                len(train_domains),
                len(test_domains),
                len(intents_only_train),
                len(intents_only_test),
                len(domains_only_train),
                len(domains_only_test),
                len(overlapping_texts),
            ],
        }
    )

    comparison.to_csv(
        RESULTS_DIR / "train_test_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        {"intent_only_train": intents_only_train}
    ).to_csv(
        RESULTS_DIR / "intents_only_train.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        {"intent_only_test": intents_only_test}
    ).to_csv(
        RESULTS_DIR / "intents_only_test.csv",
        index=False,
        encoding="utf-8-sig",
    )

    overlap_dataframe = test_dataframe[
        test_dataframe["text"].isin(overlapping_texts)
    ].copy()

    overlap_dataframe.to_csv(
        RESULTS_DIR / "train_test_overlapping_texts.csv",
        index=False,
        encoding="utf-8-sig",
    )


def save_summary(
    train_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
) -> None:
    """Sauvegarde un résumé général du dataset."""
    summary = pd.DataFrame(
        [
            {
                "dataset": "train",
                "rows": len(train_dataframe),
                "columns": len(train_dataframe.columns),
                "domains": train_dataframe["domain"].nunique(),
                "intents": train_dataframe["intent"].nunique(),
                "missing_values": int(
                    train_dataframe.isna().sum().sum()
                ),
                "full_duplicates": int(
                    train_dataframe.duplicated().sum()
                ),
                "text_duplicates": int(
                    train_dataframe["text"].duplicated().sum()
                ),
            },
            {
                "dataset": "test",
                "rows": len(test_dataframe),
                "columns": len(test_dataframe.columns),
                "domains": test_dataframe["domain"].nunique(),
                "intents": test_dataframe["intent"].nunique(),
                "missing_values": int(
                    test_dataframe.isna().sum().sum()
                ),
                "full_duplicates": int(
                    test_dataframe.duplicated().sum()
                ),
                "text_duplicates": int(
                    test_dataframe["text"].duplicated().sum()
                ),
            },
        ]
    )

    print(f"\n{'=' * 70}")
    print("Résumé général")
    print(f"{'=' * 70}")
    print(summary.to_string(index=False))

    summary.to_csv(
        RESULTS_DIR / "dataset_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )


def plot_domain_distribution(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> None:
    """Crée un graphique de la distribution des domaines."""
    plt.figure(figsize=(12, 6))

    order = dataframe["domain"].value_counts().index

    sns.countplot(
        data=dataframe,
        x="domain",
        order=order,
    )

    plt.title(
        f"Distribution des domaines — {dataset_name}"
    )
    plt.xlabel("Domaine")
    plt.ylabel("Nombre d'exemples")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()

    output_path = (
        FIGURES_DIR
        / f"{dataset_name}_domain_distribution.png"
    )

    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_intent_distribution(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> None:
    """Crée un graphique horizontal des intentions."""
    counts = dataframe["intent"].value_counts().sort_values()

    plt.figure(figsize=(12, 18))
    counts.plot(kind="barh")

    plt.title(
        f"Distribution des intentions — {dataset_name}"
    )
    plt.xlabel("Nombre d'exemples")
    plt.ylabel("Intention")
    plt.tight_layout()

    output_path = (
        FIGURES_DIR
        / f"{dataset_name}_intent_distribution.png"
    )

    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_text_length_distribution(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> None:
    """Crée un histogramme de la longueur des textes."""
    word_lengths = (
        dataframe["text"]
        .fillna("")
        .astype(str)
        .str.split()
        .str.len()
    )

    plt.figure(figsize=(10, 6))
    plt.hist(word_lengths, bins=30)

    plt.title(
        f"Distribution de la longueur des textes — {dataset_name}"
    )
    plt.xlabel("Nombre de mots")
    plt.ylabel("Nombre de textes")
    plt.tight_layout()

    output_path = (
        FIGURES_DIR
        / f"{dataset_name}_text_length_distribution.png"
    )

    plt.savefig(output_path, dpi=200)
    plt.close()


def analyze_dataset(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """Exécute toutes les analyses sur un dataset."""
    validate_columns(dataframe, dataset_name)

    cleaned_dataframe = clean_text_columns(dataframe)

    analyze_missing_values(cleaned_dataframe, dataset_name)
    analyze_duplicates(cleaned_dataframe, dataset_name)
    analyze_domains(cleaned_dataframe, dataset_name)
    analyze_intents(cleaned_dataframe, dataset_name)
    analyze_text_lengths(cleaned_dataframe, dataset_name)
    analyze_intent_domain_consistency(
        cleaned_dataframe,
        dataset_name,
    )
    analyze_conflicting_labels(
        cleaned_dataframe,
        dataset_name,
    )

    plot_domain_distribution(
        cleaned_dataframe,
        dataset_name,
    )
    plot_intent_distribution(
        cleaned_dataframe,
        dataset_name,
    )
    plot_text_length_distribution(
        cleaned_dataframe,
        dataset_name,
    )

    return cleaned_dataframe


def main() -> None:
    """Point d'entrée principal du script."""
    create_output_directories()

    train_dataframe = load_dataset(
        TRAIN_PATH,
        "train",
    )

    test_dataframe = load_dataset(
        TEST_PATH,
        "test",
    )

    train_cleaned = analyze_dataset(
        train_dataframe,
        "train",
    )

    test_cleaned = analyze_dataset(
        test_dataframe,
        "test",
    )

    compare_train_and_test(
        train_cleaned,
        test_cleaned,
    )

    save_summary(
        train_cleaned,
        test_cleaned,
    )

    print(f"\n{'=' * 70}")
    print("Analyse terminée avec succès")
    print(f"{'=' * 70}")
    print(f"Résultats CSV : {RESULTS_DIR}")
    print(f"Graphiques : {FIGURES_DIR}")


if __name__ == "__main__":
    main()