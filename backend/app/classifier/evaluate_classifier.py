from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

TEST_PATH = PROJECT_ROOT / "backend" / "data" / "test.csv"

MODELS_DIR = PROJECT_ROOT / "backend" / "models"
RESULTS_DIR = PROJECT_ROOT / "reports" / "results"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

CLASSIFIER_PATH = MODELS_DIR / "intent_classifier.joblib"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"
INTENT_DOMAIN_PATH = MODELS_DIR / "intent_to_domain.json"
TEST_EMBEDDINGS_PATH = MODELS_DIR / "test_embeddings.npy"

SUMMARY_PATH = RESULTS_DIR / "classifier_evaluation_summary.json"
CLASSIFICATION_REPORT_PATH = (
    RESULTS_DIR / "classification_report.csv"
)
ERRORS_PATH = RESULTS_DIR / "classification_errors.csv"
PREDICTIONS_PATH = RESULTS_DIR / "test_predictions.csv"
TOP_CONFUSIONS_PATH = RESULTS_DIR / "top_confusions.csv"
DOMAIN_METRICS_PATH = RESULTS_DIR / "domain_metrics.csv"
CONFIDENCE_ANALYSIS_PATH = (
    RESULTS_DIR / "confidence_analysis.csv"
)

CONFUSION_MATRIX_PATH = (
    FIGURES_DIR / "confusion_matrix_counts.png"
)
NORMALIZED_CONFUSION_MATRIX_PATH = (
    FIGURES_DIR / "confusion_matrix_normalized.png"
)
TOP_CONFUSIONS_FIGURE_PATH = (
    FIGURES_DIR / "top_confusions.png"
)
CONFIDENCE_HISTOGRAM_PATH = (
    FIGURES_DIR / "confidence_distribution.png"
)

CONFIDENCE_THRESHOLD = 0.60


def create_output_directories() -> None:
    """Crée les dossiers de sortie."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def validate_files() -> None:
    """Vérifie que tous les fichiers nécessaires existent."""
    required_files = [
        TEST_PATH,
        CLASSIFIER_PATH,
        LABEL_ENCODER_PATH,
        INTENT_DOMAIN_PATH,
        TEST_EMBEDDINGS_PATH,
    ]

    missing_files = [
        path for path in required_files if not path.exists()
    ]

    if missing_files:
        formatted = "\n".join(str(path) for path in missing_files)
        raise FileNotFoundError(
            "Certains fichiers sont introuvables :\n"
            f"{formatted}"
        )


def load_artifacts():
    """Charge le dataset, le modèle et les fichiers associés."""
    test_dataframe = pd.read_csv(TEST_PATH)

    classifier = joblib.load(CLASSIFIER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)

    with INTENT_DOMAIN_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        intent_to_domain = json.load(file)

    test_embeddings = np.load(TEST_EMBEDDINGS_PATH)

    print("Fichiers chargés avec succès")
    print("Test :", test_dataframe.shape)
    print("Embeddings :", test_embeddings.shape)
    print("Classes :", len(label_encoder.classes_))

    if len(test_dataframe) != len(test_embeddings):
        raise ValueError(
            "Le nombre de lignes du test ne correspond pas "
            "au nombre d'embeddings."
        )

    return (
        test_dataframe,
        classifier,
        label_encoder,
        intent_to_domain,
        test_embeddings,
    )


def calculate_global_metrics(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
) -> dict[str, float]:
    """Calcule les métriques globales."""
    metrics = {
        "accuracy": float(
            accuracy_score(true_labels, predicted_labels)
        ),
        "macro_precision": float(
            precision_score(
                true_labels,
                predicted_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                true_labels,
                predicted_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                true_labels,
                predicted_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                true_labels,
                predicted_labels,
                average="weighted",
                zero_division=0,
            )
        ),
    }

    return metrics


def create_classification_report(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    class_names: list[str],
) -> pd.DataFrame:
    """Crée le rapport détaillé par intention."""
    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    report_dataframe = (
        pd.DataFrame(report)
        .transpose()
        .reset_index()
        .rename(columns={"index": "intent"})
    )

    report_dataframe.to_csv(
        CLASSIFICATION_REPORT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    return report_dataframe


def build_predictions_dataframe(
    test_dataframe: pd.DataFrame,
    classifier,
    label_encoder,
    intent_to_domain: dict[str, str],
    test_embeddings: np.ndarray,
) -> pd.DataFrame:
    """Construit le tableau complet des prédictions."""
    encoded_true_labels = label_encoder.transform(
        test_dataframe["intent"]
    )

    encoded_predictions = classifier.predict(
        test_embeddings
    )

    probabilities = classifier.predict_proba(
        test_embeddings
    )

    confidences = probabilities.max(axis=1)

    predicted_intents = label_encoder.inverse_transform(
        encoded_predictions
    )

    predicted_domains = [
        intent_to_domain[intent]
        for intent in predicted_intents
    ]

    top_2_indices = np.argsort(
        probabilities,
        axis=1,
    )[:, -2:]

    second_best_indices = top_2_indices[:, -2]

    second_best_intents = label_encoder.inverse_transform(
        second_best_indices
    )

    second_best_confidences = probabilities[
        np.arange(len(probabilities)),
        second_best_indices,
    ]

    prediction_dataframe = test_dataframe.copy()

    prediction_dataframe["predicted_intent"] = (
        predicted_intents
    )

    prediction_dataframe["predicted_domain"] = (
        predicted_domains
    )

    prediction_dataframe["confidence"] = confidences

    prediction_dataframe["second_best_intent"] = (
        second_best_intents
    )

    prediction_dataframe["second_best_confidence"] = (
        second_best_confidences
    )

    prediction_dataframe["confidence_margin"] = (
        prediction_dataframe["confidence"]
        - prediction_dataframe["second_best_confidence"]
    )

    prediction_dataframe["correct"] = (
        prediction_dataframe["intent"]
        == prediction_dataframe["predicted_intent"]
    )

    prediction_dataframe["domain_correct"] = (
        prediction_dataframe["domain"]
        == prediction_dataframe["predicted_domain"]
    )

    prediction_dataframe["below_threshold"] = (
        prediction_dataframe["confidence"]
        < CONFIDENCE_THRESHOLD
    )

    prediction_dataframe.to_csv(
        PREDICTIONS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    errors_dataframe = prediction_dataframe[
        ~prediction_dataframe["correct"]
    ].copy()

    errors_dataframe = errors_dataframe.sort_values(
        by="confidence",
        ascending=False,
    )

    errors_dataframe.to_csv(
        ERRORS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    return prediction_dataframe


def create_confusion_matrices(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    class_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Crée les matrices de confusion brute et normalisée."""
    raw_matrix = confusion_matrix(
        true_labels,
        predicted_labels,
    )

    normalized_matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        normalize="true",
    )

    figure_size = (22, 20)

    plt.figure(figsize=figure_size)
    plt.imshow(raw_matrix, interpolation="nearest")
    plt.title("Matrice de confusion — valeurs brutes")
    plt.xlabel("Intention prédite")
    plt.ylabel("Intention réelle")
    plt.xticks(
        range(len(class_names)),
        class_names,
        rotation=90,
        fontsize=6,
    )
    plt.yticks(
        range(len(class_names)),
        class_names,
        fontsize=6,
    )
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(
        CONFUSION_MATRIX_PATH,
        dpi=200,
    )
    plt.close()

    plt.figure(figsize=figure_size)
    plt.imshow(
        normalized_matrix,
        interpolation="nearest",
    )
    plt.title("Matrice de confusion normalisée")
    plt.xlabel("Intention prédite")
    plt.ylabel("Intention réelle")
    plt.xticks(
        range(len(class_names)),
        class_names,
        rotation=90,
        fontsize=6,
    )
    plt.yticks(
        range(len(class_names)),
        class_names,
        fontsize=6,
    )
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(
        NORMALIZED_CONFUSION_MATRIX_PATH,
        dpi=200,
    )
    plt.close()

    return raw_matrix, normalized_matrix


def extract_top_confusions(
    confusion: np.ndarray,
    class_names: list[str],
    top_n: int = 20,
) -> pd.DataFrame:
    """Extrait les erreurs inter-classes les plus fréquentes."""
    confusion_copy = confusion.copy()

    np.fill_diagonal(confusion_copy, 0)

    rows: list[dict[str, object]] = []

    for true_index in range(confusion_copy.shape[0]):
        for predicted_index in range(
            confusion_copy.shape[1]
        ):
            error_count = int(
                confusion_copy[
                    true_index,
                    predicted_index,
                ]
            )

            if error_count > 0:
                rows.append(
                    {
                        "true_intent": class_names[
                            true_index
                        ],
                        "predicted_intent": class_names[
                            predicted_index
                        ],
                        "error_count": error_count,
                    }
                )

    top_confusions = pd.DataFrame(rows)

    if not top_confusions.empty:
        top_confusions = (
            top_confusions
            .sort_values(
                by="error_count",
                ascending=False,
            )
            .head(top_n)
            .reset_index(drop=True)
        )

    top_confusions.to_csv(
        TOP_CONFUSIONS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    if not top_confusions.empty:
        chart_labels = (
            top_confusions["true_intent"]
            + " → "
            + top_confusions["predicted_intent"]
        )

        plt.figure(figsize=(12, 9))
        plt.barh(
            chart_labels[::-1],
            top_confusions["error_count"][::-1],
        )
        plt.title("Top erreurs inter-classes")
        plt.xlabel("Nombre d'erreurs")
        plt.ylabel("Confusion")
        plt.tight_layout()
        plt.savefig(
            TOP_CONFUSIONS_FIGURE_PATH,
            dpi=200,
        )
        plt.close()

    return top_confusions


def evaluate_by_domain(
    prediction_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Calcule les performances pour chacun des 7 domaines."""
    domain_rows = []

    for domain, group in prediction_dataframe.groupby(
        "domain"
    ):
        domain_rows.append(
            {
                "domain": domain,
                "examples": len(group),
                "intent_accuracy": float(
                    group["correct"].mean()
                ),
                "domain_accuracy": float(
                    group["domain_correct"].mean()
                ),
                "mean_confidence": float(
                    group["confidence"].mean()
                ),
                "error_count": int(
                    (~group["correct"]).sum()
                ),
                "low_confidence_count": int(
                    group["below_threshold"].sum()
                ),
            }
        )

    domain_metrics = pd.DataFrame(domain_rows)

    domain_metrics = domain_metrics.sort_values(
        by="intent_accuracy",
        ascending=False,
    )

    domain_metrics.to_csv(
        DOMAIN_METRICS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    return domain_metrics


def analyze_confidence(
    prediction_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Analyse les scores de confiance."""
    correct_predictions = prediction_dataframe[
        prediction_dataframe["correct"]
    ]

    incorrect_predictions = prediction_dataframe[
        ~prediction_dataframe["correct"]
    ]

    high_confidence_predictions = prediction_dataframe[
        prediction_dataframe["confidence"]
        >= CONFIDENCE_THRESHOLD
    ]

    low_confidence_predictions = prediction_dataframe[
        prediction_dataframe["confidence"]
        < CONFIDENCE_THRESHOLD
    ]

    rows = [
        {
            "indicator": "mean_confidence_all",
            "value": prediction_dataframe[
                "confidence"
            ].mean(),
        },
        {
            "indicator": "mean_confidence_correct",
            "value": correct_predictions[
                "confidence"
            ].mean(),
        },
        {
            "indicator": "mean_confidence_errors",
            "value": incorrect_predictions[
                "confidence"
            ].mean()
            if len(incorrect_predictions) > 0
            else 0.0,
        },
        {
            "indicator": "low_confidence_count",
            "value": len(low_confidence_predictions),
        },
        {
            "indicator": "low_confidence_rate",
            "value": len(low_confidence_predictions)
            / len(prediction_dataframe),
        },
        {
            "indicator": "high_confidence_count",
            "value": len(high_confidence_predictions),
        },
        {
            "indicator": "high_confidence_accuracy",
            "value": high_confidence_predictions[
                "correct"
            ].mean()
            if len(high_confidence_predictions) > 0
            else 0.0,
        },
        {
            "indicator": "low_confidence_accuracy",
            "value": low_confidence_predictions[
                "correct"
            ].mean()
            if len(low_confidence_predictions) > 0
            else 0.0,
        },
        {
            "indicator": "mean_confidence_margin",
            "value": prediction_dataframe[
                "confidence_margin"
            ].mean(),
        },
    ]

    confidence_analysis = pd.DataFrame(rows)

    confidence_analysis.to_csv(
        CONFIDENCE_ANALYSIS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    plt.figure(figsize=(10, 6))
    plt.hist(
        correct_predictions["confidence"],
        bins=20,
        alpha=0.7,
        label="Prédictions correctes",
    )

    if len(incorrect_predictions) > 0:
        plt.hist(
            incorrect_predictions["confidence"],
            bins=20,
            alpha=0.7,
            label="Erreurs",
        )

    plt.axvline(
        CONFIDENCE_THRESHOLD,
        linestyle="--",
        label=f"Seuil {CONFIDENCE_THRESHOLD}",
    )

    plt.title("Distribution des scores de confiance")
    plt.xlabel("Score de confiance")
    plt.ylabel("Nombre de prédictions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        CONFIDENCE_HISTOGRAM_PATH,
        dpi=200,
    )
    plt.close()

    return confidence_analysis


def save_summary(
    global_metrics: dict[str, float],
    prediction_dataframe: pd.DataFrame,
    top_confusions: pd.DataFrame,
) -> None:
    """Sauvegarde le résumé de l'évaluation."""
    error_count = int(
        (~prediction_dataframe["correct"]).sum()
    )

    domain_error_count = int(
        (~prediction_dataframe["domain_correct"]).sum()
    )

    summary = {
        **global_metrics,
        "test_examples": int(
            len(prediction_dataframe)
        ),
        "correct_predictions": int(
            prediction_dataframe["correct"].sum()
        ),
        "incorrect_predictions": error_count,
        "domain_errors": domain_error_count,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "low_confidence_predictions": int(
            prediction_dataframe[
                "below_threshold"
            ].sum()
        ),
        "mean_confidence": float(
            prediction_dataframe[
                "confidence"
            ].mean()
        ),
        "number_of_confusion_pairs": int(
            len(top_confusions)
        ),
    }

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )


def main() -> None:
    """Point d'entrée principal."""
    create_output_directories()
    validate_files()

    print("=" * 70)
    print("ÉVALUATION COMPLÈTE DU CLASSIFIEUR")
    print("=" * 70)

    (
        test_dataframe,
        classifier,
        label_encoder,
        intent_to_domain,
        test_embeddings,
    ) = load_artifacts()

    true_encoded_labels = label_encoder.transform(
        test_dataframe["intent"]
    )

    predicted_encoded_labels = classifier.predict(
        test_embeddings
    )

    class_names = label_encoder.classes_.tolist()

    global_metrics = calculate_global_metrics(
        true_encoded_labels,
        predicted_encoded_labels,
    )

    print("\nMétriques globales")
    print(
        f"Accuracy : "
        f"{global_metrics['accuracy']:.4f}"
    )
    print(
        f"Macro precision : "
        f"{global_metrics['macro_precision']:.4f}"
    )
    print(
        f"Macro recall : "
        f"{global_metrics['macro_recall']:.4f}"
    )
    print(
        f"Macro F1 : "
        f"{global_metrics['macro_f1']:.4f}"
    )
    print(
        f"Weighted F1 : "
        f"{global_metrics['weighted_f1']:.4f}"
    )

    report_dataframe = create_classification_report(
        true_encoded_labels,
        predicted_encoded_labels,
        class_names,
    )

    prediction_dataframe = build_predictions_dataframe(
        test_dataframe,
        classifier,
        label_encoder,
        intent_to_domain,
        test_embeddings,
    )

    raw_confusion, _ = create_confusion_matrices(
        true_encoded_labels,
        predicted_encoded_labels,
        class_names,
    )

    top_confusions = extract_top_confusions(
        raw_confusion,
        class_names,
        top_n=20,
    )

    domain_metrics = evaluate_by_domain(
        prediction_dataframe
    )

    confidence_analysis = analyze_confidence(
        prediction_dataframe
    )

    save_summary(
        global_metrics,
        prediction_dataframe,
        top_confusions,
    )

    error_count = int(
        (~prediction_dataframe["correct"]).sum()
    )

    print("\nRésumé")
    print(
        "Prédictions correctes :",
        int(prediction_dataframe["correct"].sum()),
    )
    print("Erreurs :", error_count)
    print(
        "Accuracy domaine :",
        f"{prediction_dataframe['domain_correct'].mean():.4f}",
    )

    print("\nTop confusions")
    if top_confusions.empty:
        print("Aucune confusion détectée.")
    else:
        print(top_confusions.head(10).to_string(index=False))

    print("\nPerformance par domaine")
    print(domain_metrics.to_string(index=False))

    print("\nAnalyse de confiance")
    print(confidence_analysis.to_string(index=False))

    print("\nRapport par intention")
    print(
        report_dataframe[
            [
                "intent",
                "precision",
                "recall",
                "f1-score",
                "support",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print("\nFichiers créés")
    print(f"- {SUMMARY_PATH}")
    print(f"- {CLASSIFICATION_REPORT_PATH}")
    print(f"- {PREDICTIONS_PATH}")
    print(f"- {ERRORS_PATH}")
    print(f"- {TOP_CONFUSIONS_PATH}")
    print(f"- {DOMAIN_METRICS_PATH}")
    print(f"- {CONFIDENCE_ANALYSIS_PATH}")
    print(f"- {CONFUSION_MATRIX_PATH}")
    print(f"- {NORMALIZED_CONFUSION_MATRIX_PATH}")
    print(f"- {TOP_CONFUSIONS_FIGURE_PATH}")
    print(f"- {CONFIDENCE_HISTOGRAM_PATH}")

    print("\n" + "=" * 70)
    print("ÉVALUATION TERMINÉE AVEC SUCCÈS")
    print("=" * 70)


if __name__ == "__main__":
    main()