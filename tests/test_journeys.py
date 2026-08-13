from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.app.journeys.catalogue import JOURNEYS
from backend.app.journeys.service import detect_journey
from backend.app.main import app
from backend.app.routes.forms import ALLOWED_FORMS


client = TestClient(app)


def classification(
    intent: str,
    confidence: float,
) -> dict[str, object]:
    return {
        "domain": "vehicule",
        "intent": intent,
        "confidence": confidence,
        "needs_clarification": confidence < 0.60,
        "alternatives": [],
    }


@pytest.mark.parametrize(
    "goal,expected",
    [
        (
            "J'ai acheté une voiture d'occasion",
            "achat_vehicule_occasion",
        ),
        (
            "j'ai perdu mon permis de conduire hier",
            "permis_perdu",
        ),
        (
            "je veux contester une contravention",
            "contester_contravention",
        ),
        (
            "je dois faire la visite technique",
            "visite_technique",
        ),
        (
            "je viens d'importer un véhicule",
            "immatriculer_vehicule",
        ),
    ],
)
def test_goals_are_mapped_to_the_right_journey(
    goal: str,
    expected: str,
) -> None:
    key, confidence, reason = detect_journey(goal)

    assert key == expected
    assert confidence > 0
    assert reason


def test_unrelated_goal_is_not_forced_into_a_journey() -> None:
    key, confidence, _ = detect_journey(
        "je voudrais connaitre la meteo de demain"
    )

    assert key is None
    assert confidence == 0.0


def test_classification_supports_detection() -> None:
    key, _confidence, reason = detect_journey(
        "je dois refaire les papiers du vehicule",
        classification=classification(
            "mutation_vehicule",
            0.91,
        ),
    )

    assert key == "achat_vehicule_occasion"
    assert "mutation_vehicule" in reason


def test_low_confidence_classification_is_ignored() -> None:
    key, _confidence, _reason = detect_journey(
        "bonjour",
        classification=classification(
            "mutation_vehicule",
            0.21,
        ),
    )

    assert key is None


def test_plan_creates_a_journey_with_progress() -> None:
    response = client.post(
        "/journeys/plan",
        json={
            "goal": "J'ai acheté une voiture d'occasion",
            "session_id": "test-session",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["detected"] is True

    journey = data["journey"]

    assert journey["reference"].startswith("PARC-")
    assert journey["total_steps"] == 6
    assert journey["completed_count"] == 0
    assert journey["progress"] == 0
    assert journey["next_step"]["index"] == 0
    assert "30 jours" in journey["deadline"]


def test_steps_can_be_completed_and_reopened() -> None:
    journey = client.post(
        "/journeys/plan",
        json={"goal": "J'ai perdu mon permis de conduire"},
    ).json()["journey"]

    reference = journey["reference"]

    client.post(
        f"/journeys/{reference}/steps/0",
        json={"done": True},
    )
    updated = client.post(
        f"/journeys/{reference}/steps/1",
        json={"done": True},
    ).json()

    assert updated["completed_count"] == 2
    assert updated["total_steps"] == 5
    assert updated["progress"] == 40
    assert updated["next_step"]["index"] == 2

    reopened = client.post(
        f"/journeys/{reference}/steps/0",
        json={"done": False},
    ).json()

    assert reopened["completed_count"] == 1
    assert reopened["next_step"]["index"] == 0


def test_progress_is_persisted() -> None:
    journey = client.post(
        "/journeys/plan",
        json={"goal": "je veux contester une contravention"},
    ).json()["journey"]

    client.post(
        f"/journeys/{journey['reference']}/steps/0",
        json={"done": True},
    )

    reloaded = client.get(
        f"/journeys/{journey['reference']}"
    )

    assert reloaded.status_code == 200
    assert reloaded.json()["completed_count"] == 1


def test_unknown_goal_returns_the_catalogue() -> None:
    response = client.post(
        "/journeys/plan",
        json={"goal": "je cherche un restaurant a rabat"},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["detected"] is False
    assert data["journey"] is None
    assert len(data["suggestions"]) == len(JOURNEYS)


def test_unknown_reference_and_step_are_rejected() -> None:
    assert (
        client.get("/journeys/PARC-2026-INCONNU").status_code
        == 404
    )

    journey = client.post(
        "/journeys/plan",
        json={"goal": "je dois faire la visite technique"},
    ).json()["journey"]

    response = client.post(
        f"/journeys/{journey['reference']}/steps/99",
        json={"done": True},
    )

    assert response.status_code == 400


def test_catalogue_lists_every_journey() -> None:
    response = client.get("/journeys/catalogue")

    assert response.status_code == 200
    assert len(response.json()["journeys"]) == len(JOURNEYS)


def test_referenced_forms_and_sources_exist() -> None:
    from pathlib import Path

    documents_dir = (
        Path(__file__).resolve().parents[1]
        / "backend"
        / "data"
        / "documents"
    )

    for journey in JOURNEYS.values():
        for step in journey["steps"]:
            for filename in step.get("forms", []):
                assert filename in ALLOWED_FORMS

            source = step.get("source")

            if source:
                assert (documents_dir / source).is_file(), source


def test_document_content_is_served_for_a_source() -> None:
    response = client.get(
        "/documents/content",
        params={
            "path": "vehicule/mutation_vehicule_narsa.md"
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["domain"] == "vehicule"
    assert "mutation" in data["content"].lower()
    assert "Métadonnées" not in data["content"]


@pytest.mark.parametrize(
    "path",
    [
        "../../.env",
        "vehicule/inexistant.md",
        "/etc/passwd",
    ],
)
def test_document_route_rejects_unknown_paths(
    path: str,
) -> None:
    response = client.get(
        "/documents/content",
        params={"path": path},
    )

    assert response.status_code == 404
