from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from backend.app.database.models import (
    Complaint,
    ConversationLog,
)


# Deux questions sont considérées comme portant sur le même
# sujet au-delà de ce niveau de similarité cosinus.
SIMILARITY_THRESHOLD = 0.72

MAX_ANALYZED_QUESTIONS = 200
MAX_THEMES = 6

STALE_COMPLAINT_DAYS = 7
CLARIFICATION_RATE_ALERT = 0.25
NO_CONTEXT_RATE_ALERT = 0.15
CONFIDENCE_ALERT = 0.70


def _unanswered_questions(
    session: Session,
    since: datetime,
) -> list[ConversationLog]:
    """Questions restées sans réponse documentaire fiable."""
    statement = (
        select(ConversationLog)
        .where(ConversationLog.created_at >= since)
        .where(
            or_(
                ConversationLog.status == "no_context_found",
                ConversationLog.needs_clarification.is_(True),
            )
        )
        .order_by(ConversationLog.created_at.desc())
        .limit(MAX_ANALYZED_QUESTIONS)
    )

    return list(session.scalars(statement))


def _cluster_questions(
    questions: list[str],
) -> list[list[int]]:
    """
    Regroupe les questions par proximité sémantique.

    Le modèle d'embedding déjà chargé pour la classification
    est réutilisé : aucun appel réseau n'est effectué.
    """
    if len(questions) < 2:
        return [[index] for index in range(len(questions))]

    try:
        import numpy as np

        from backend.app.classifier.classifier_service import (
            get_classifier_service,
        )

        embeddings = get_classifier_service().embedding_model.encode(
            questions,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    except Exception as error:  # pragma: no cover
        print(
            "Regroupement sémantique indisponible :",
            f"{type(error).__name__}: {error}",
        )
        return [[index] for index in range(len(questions))]

    clusters: list[list[int]] = []
    centroids: list[Any] = []

    for index, embedding in enumerate(embeddings):
        best_cluster = -1
        best_similarity = 0.0

        for cluster_index, centroid in enumerate(centroids):
            similarity = float(np.dot(embedding, centroid))

            if (
                similarity > best_similarity
                and similarity >= SIMILARITY_THRESHOLD
            ):
                best_similarity = similarity
                best_cluster = cluster_index

        if best_cluster == -1:
            clusters.append([index])
            centroids.append(embedding)
            continue

        clusters[best_cluster].append(index)

        members = embeddings[clusters[best_cluster]]
        centroid = members.mean(axis=0)
        norm = float(np.linalg.norm(centroid)) or 1.0
        centroids[best_cluster] = centroid / norm

    return clusters


def build_blind_spots(
    session: Session,
    since: datetime,
) -> list[dict[str, Any]]:
    """
    Thèmes sur lesquels l'assistant n'a pas su répondre.

    Ces regroupements indiquent les documents à écrire ou à
    compléter en priorité.
    """
    logs = _unanswered_questions(
        session=session,
        since=since,
    )

    if not logs:
        return []

    questions = [log.question for log in logs]
    clusters = _cluster_questions(questions)

    themes: list[dict[str, Any]] = []

    for cluster in clusters:
        members = [logs[index] for index in cluster]

        # La question la plus complète représente le thème.
        representative = max(
            members,
            key=lambda log: len(log.question),
        )

        domains = sorted(
            {
                log.domain
                for log in members
                if log.domain
            }
        )

        themes.append(
            {
                "question": representative.question,
                "occurrences": len(members),
                "domains": domains,
                "last_seen": max(
                    log.created_at for log in members
                ),
                "examples": [
                    log.question
                    for log in members[:3]
                    if log.question != representative.question
                ],
            }
        )

    themes.sort(
        key=lambda theme: (
            theme["occurrences"],
            theme["last_seen"],
        ),
        reverse=True,
    )

    return themes[:MAX_THEMES]


def build_alerts(
    session: Session,
    since: datetime,
) -> list[dict[str, str]]:
    """Signaux de qualité destinés à l'agent."""
    alerts: list[dict[str, str]] = []

    total = int(
        session.scalar(
            select(func.count(ConversationLog.id)).where(
                ConversationLog.created_at >= since
            )
        )
        or 0
    )

    if total:
        clarifications = int(
            session.scalar(
                select(func.count(ConversationLog.id))
                .where(ConversationLog.created_at >= since)
                .where(
                    ConversationLog.needs_clarification.is_(
                        True
                    )
                )
            )
            or 0
        )

        no_context = int(
            session.scalar(
                select(func.count(ConversationLog.id))
                .where(ConversationLog.created_at >= since)
                .where(
                    ConversationLog.status
                    == "no_context_found"
                )
            )
            or 0
        )

        average_confidence = float(
            session.scalar(
                select(
                    func.avg(ConversationLog.confidence)
                ).where(
                    ConversationLog.created_at >= since
                )
            )
            or 0.0
        )

        clarification_rate = clarifications / total
        no_context_rate = no_context / total

        if clarification_rate > CLARIFICATION_RATE_ALERT:
            alerts.append(
                {
                    "level": "attention",
                    "title": "Taux de clarification élevé",
                    "detail": (
                        f"{round(clarification_rate * 100)} % des "
                        "échanges se terminent par une demande de "
                        "reformulation. Vérifiez les formulations "
                        "mal classées."
                    ),
                }
            )

        if no_context_rate > NO_CONTEXT_RATE_ALERT:
            alerts.append(
                {
                    "level": "critique",
                    "title": "Questions sans document",
                    "detail": (
                        f"{round(no_context_rate * 100)} % des "
                        "questions ne trouvent aucun document "
                        "fiable. Consultez les angles morts "
                        "ci-dessous."
                    ),
                }
            )

        if 0 < average_confidence < CONFIDENCE_ALERT:
            alerts.append(
                {
                    "level": "attention",
                    "title": "Confiance moyenne faible",
                    "detail": (
                        "La confiance moyenne du classifieur est de "
                        f"{average_confidence:.2f}. Un réentraînement "
                        "peut être nécessaire."
                    ),
                }
            )

    stale_threshold = datetime.now(timezone.utc) - timedelta(
        days=STALE_COMPLAINT_DAYS
    )

    stale_complaints = int(
        session.scalar(
            select(func.count(Complaint.id))
            .where(Complaint.status == "recue")
            .where(Complaint.created_at < stale_threshold)
        )
        or 0
    )

    if stale_complaints:
        alerts.append(
            {
                "level": "critique",
                "title": "Réclamations non traitées",
                "detail": (
                    f"{stale_complaints} réclamation(s) sont "
                    f"en attente depuis plus de "
                    f"{STALE_COMPLAINT_DAYS} jours."
                ),
            }
        )

    urgent_complaints = int(
        session.scalar(
            select(func.count(Complaint.id))
            .where(Complaint.priority == "urgente")
            .where(Complaint.status.in_(["recue", "en_cours"]))
        )
        or 0
    )

    if urgent_complaints:
        alerts.append(
            {
                "level": "attention",
                "title": "Réclamations urgentes ouvertes",
                "detail": (
                    f"{urgent_complaints} réclamation(s) classée(s) "
                    "urgente(s) sont encore ouvertes."
                ),
            }
        )

    if not alerts:
        alerts.append(
            {
                "level": "ok",
                "title": "Aucune anomalie détectée",
                "detail": (
                    "Les indicateurs de la période sont dans "
                    "les valeurs attendues."
                ),
            }
        )

    return alerts


def build_insights(
    session: Session,
    days: int = 7,
) -> dict[str, Any]:
    """Alertes qualité et angles morts documentaires."""
    days = max(1, min(days, 90))
    since = datetime.now(timezone.utc) - timedelta(days=days)

    return {
        "period_days": days,
        "alerts": build_alerts(
            session=session,
            since=since,
        ),
        "blind_spots": build_blind_spots(
            session=session,
            since=since,
        ),
    }
