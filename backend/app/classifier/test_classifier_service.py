from __future__ import annotations

from classifier_service import predict_intent


TEST_QUESTIONS = [
    "J'ai perdu mon permis de conduire.",
    "Comment payer une amende ?",
    "Je souhaite prendre un rendez-vous.",
    "Quels documents faut-il pour la visite technique ?",
    "J'ai eu un accident avec un blessé.",
    "j perdu mon permi",
    "day3 lia permis chno ndir",
]


def main() -> None:
    for question in TEST_QUESTIONS:
        print("=" * 70)
        print("Question :", question)

        try:
            result = predict_intent(
                question,
                top_k=3,
            )

            print("Domaine :", result["domain"])
            print("Intention :", result["intent"])
            print("Confiance :", result["confidence"])
            print(
                "Clarification nécessaire :",
                result["needs_clarification"],
            )

            print("Alternatives :")

            for alternative in result["alternatives"]:
                print(
                    f"- {alternative['intent']} "
                    f"({alternative['domain']}) : "
                    f"{alternative['confidence']}"
                )

        except Exception as error:
            print("Erreur :", error)


if __name__ == "__main__":
    main()