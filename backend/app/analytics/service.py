from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.database.models import (
    COMPLAINT_STATUS_LABELS,
    Complaint,
    ConversationLog,
)
from backend.app.schemas.complaints import (
    COMPLAINT_CATEGORY_LABELS,
)


LOW_CONFIDENCE_THRESHOLD = 0.60


def _percent_change(
    current: float,
    previous: float,
) -> float | None:
    """Variation en pourcentage entre deux périodes."""
    if previous <= 0:
        return None

    return round(
        ((current - previous) / previous) * 100,
        1,
    )


def _window_bounds(
    days: int,
) -> tuple[datetime, datetime]:
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days)
    previous_start = start - timedelta(days=days)

    return previous_start, start


def _log_metrics(
    session: Session,
    since: datetime | None = None,
    until: datetime | None = None,
) -> dict[str, Any]:
    statement = select(
        func.count(ConversationLog.id),
        func.avg(ConversationLog.confidence),
        func.avg(ConversationLog.total_time_ms),
        func.count(func.distinct(ConversationLog.session_id)),
    )

    if since is not None:
        statement = statement.where(
            ConversationLog.created_at >= since
        )

    if until is not None:
        statement = statement.where(
            ConversationLog.created_at < until
        )

    row = session.execute(statement).one()

    return {
        "interactions": int(row[0] or 0),
        "avg_confidence": round(float(row[1] or 0.0), 2),
        "avg_time_ms": round(float(row[2] or 0.0), 1),
        "conversations": int(row[3] or 0),
    }


def _count_where(
    session: Session,
    condition: Any,
    since: datetime | None = None,
    until: datetime | None = None,
) -> int:
    statement = select(
        func.count(ConversationLog.id)
    ).where(condition)

    if since is not None:
        statement = statement.where(
            ConversationLog.created_at >= since
        )

    if until is not None:
        statement = statement.where(
            ConversationLog.created_at < until
        )

    return int(session.scalar(statement) or 0)


def build_dashboard(
    session: Session,
    days: int = 7,
) -> dict[str, Any]:
    """Agrège les indicateurs affichés dans le Dashboard IA."""
    days = max(1, min(days, 90))
    previous_start, current_start = _window_bounds(days)

    current = _log_metrics(
        session,
        since=current_start,
    )
    previous = _log_metrics(
        session,
        since=previous_start,
        until=current_start,
    )

    clarifications = _count_where(
        session,
        ConversationLog.needs_clarification.is_(True),
        since=current_start,
    )
    previous_clarifications = _count_where(
        session,
        ConversationLog.needs_clarification.is_(True),
        since=previous_start,
        until=current_start,
    )

    low_confidence = _count_where(
        session,
        ConversationLog.confidence
        < LOW_CONFIDENCE_THRESHOLD,
        since=current_start,
    )
    previous_low_confidence = _count_where(
        session,
        ConversationLog.confidence
        < LOW_CONFIDENCE_THRESHOLD,
        since=previous_start,
        until=current_start,
    )

    return {
        "period_days": days,
        "kpis": {
            "conversations": {
                "value": current["conversations"],
                "change": _percent_change(
                    current["conversations"],
                    previous["conversations"],
                ),
            },
            "interactions": {
                "value": current["interactions"],
                "change": _percent_change(
                    current["interactions"],
                    previous["interactions"],
                ),
            },
            "avg_confidence": {
                "value": current["avg_confidence"],
                "change": _percent_change(
                    current["avg_confidence"],
                    previous["avg_confidence"],
                ),
            },
            "avg_time_ms": {
                "value": current["avg_time_ms"],
                "change": _percent_change(
                    current["avg_time_ms"],
                    previous["avg_time_ms"],
                ),
            },
            "clarifications": {
                "value": clarifications,
                "change": _percent_change(
                    clarifications,
                    previous_clarifications,
                ),
            },
            "low_confidence": {
                "value": low_confidence,
                "change": _percent_change(
                    low_confidence,
                    previous_low_confidence,
                ),
            },
        },
        "timeseries": build_timeseries(
            session,
            days=days,
        ),
        "domains": build_domain_distribution(
            session,
            since=current_start,
        ),
        "top_intents": build_top_intents(
            session,
            since=current_start,
        ),
        "answer_types": build_answer_types(
            session,
            since=current_start,
        ),
        "recent": build_recent_conversations(
            session,
            limit=10,
        ),
        "complaints": build_complaint_stats(session),
        "feedback": build_feedback_stats(
            session,
            since=current_start,
        ),
    }


def build_timeseries(
    session: Session,
    days: int = 7,
) -> list[dict[str, Any]]:
    """Nombre d'interactions par jour sur la période."""
    start = datetime.now(timezone.utc) - timedelta(
        days=days
    )

    rows = session.execute(
        select(
            func.date(ConversationLog.created_at),
            func.count(ConversationLog.id),
        )
        .where(ConversationLog.created_at >= start)
        .group_by(func.date(ConversationLog.created_at))
    ).all()

    counts = {
        str(row[0]): int(row[1] or 0)
        for row in rows
    }

    series: list[dict[str, Any]] = []
    today = datetime.now(timezone.utc).date()

    for offset in range(days - 1, -1, -1):
        day = today - timedelta(days=offset)
        key = day.isoformat()

        series.append(
            {
                "date": key,
                "count": counts.get(key, 0),
            }
        )

    return series


def build_domain_distribution(
    session: Session,
    since: datetime | None = None,
) -> list[dict[str, Any]]:
    statement = select(
        ConversationLog.domain,
        func.count(ConversationLog.id),
    ).group_by(ConversationLog.domain)

    if since is not None:
        statement = statement.where(
            ConversationLog.created_at >= since
        )

    rows = session.execute(statement).all()
    total = sum(int(row[1] or 0) for row in rows) or 1

    distribution = [
        {
            "domain": row[0] or "inconnu",
            "label": COMPLAINT_CATEGORY_LABELS.get(
                row[0],
                (row[0] or "inconnu").replace("_", " "),
            ),
            "count": int(row[1] or 0),
            "percentage": round(
                int(row[1] or 0) / total * 100,
                1,
            ),
        }
        for row in rows
    ]

    distribution.sort(
        key=lambda item: item["count"],
        reverse=True,
    )

    return distribution


def build_top_intents(
    session: Session,
    since: datetime | None = None,
    limit: int = 6,
) -> list[dict[str, Any]]:
    statement = (
        select(
            ConversationLog.intent,
            func.count(ConversationLog.id),
        )
        .group_by(ConversationLog.intent)
        .order_by(func.count(ConversationLog.id).desc())
        .limit(limit)
    )

    if since is not None:
        statement = statement.where(
            ConversationLog.created_at >= since
        )

    rows = session.execute(statement).all()
    total = int(
        session.scalar(
            select(func.count(ConversationLog.id)).where(
                ConversationLog.created_at >= since
            )
            if since is not None
            else select(func.count(ConversationLog.id))
        )
        or 0
    ) or 1

    return [
        {
            "intent": row[0] or "inconnu",
            "count": int(row[1] or 0),
            "percentage": round(
                int(row[1] or 0) / total * 100,
                1,
            ),
        }
        for row in rows
    ]


def build_answer_types(
    session: Session,
    since: datetime | None = None,
) -> list[dict[str, Any]]:
    """Répartition entre génération, extraction et clarification."""
    definitions = [
        (
            "Réponses générées (Gemini)",
            ConversationLog.generated.is_(True),
        ),
        (
            "Réponses extractives (RAG)",
            ConversationLog.extractive.is_(True),
        ),
        (
            "Clarifications",
            ConversationLog.needs_clarification.is_(True),
        ),
        (
            "Sans contexte",
            ConversationLog.status == "no_context_found",
        ),
    ]

    results = [
        {
            "label": label,
            "count": _count_where(
                session,
                condition,
                since=since,
            ),
        }
        for label, condition in definitions
    ]

    total = sum(item["count"] for item in results) or 1

    for item in results:
        item["percentage"] = round(
            item["count"] / total * 100,
            1,
        )

    return results


def build_recent_conversations(
    session: Session,
    limit: int = 10,
) -> list[dict[str, Any]]:
    rows = session.scalars(
        select(ConversationLog)
        .order_by(ConversationLog.created_at.desc())
        .limit(max(1, min(limit, 100)))
    ).all()

    return [
        {
            "question": row.question,
            "domain": row.domain,
            "intent": row.intent,
            "confidence": round(
                float(row.confidence or 0.0),
                2,
            ),
            "time_ms": round(
                float(row.total_time_ms or 0.0),
                1,
            ),
            "status": row.status,
            "created_at": row.created_at,
        }
        for row in rows
    ]


def build_complaint_stats(
    session: Session,
) -> dict[str, Any]:
    rows = session.execute(
        select(
            Complaint.status,
            func.count(Complaint.id),
        ).group_by(Complaint.status)
    ).all()

    by_status = {
        row[0]: int(row[1] or 0)
        for row in rows
    }

    return {
        "total": sum(by_status.values()),
        "by_status": [
            {
                "status": status,
                "label": COMPLAINT_STATUS_LABELS.get(
                    status,
                    status,
                ),
                "count": by_status.get(status, 0),
            }
            for status in COMPLAINT_STATUS_LABELS
        ],
    }


def log_conversation(
    session: Session,
    question: str,
    answer: str,
    domain: str,
    intent: str,
    confidence: float,
    status: str,
    generated: bool,
    extractive: bool,
    needs_clarification: bool,
    total_time_ms: float,
    session_id: str | None = None,
    language: str = "fr",
) -> int:
    """
    Journalise un échange pour alimenter le dashboard.

    Retourne l'identifiant de l'interaction : il permet de
    rattacher ensuite l'avis du citoyen (utile / pas utile).
    """
    entry = ConversationLog(
        session_id=session_id,
        question=question[:2000],
        answer=answer[:4000],
        domain=domain or "inconnu",
        intent=intent or "inconnu",
        confidence=float(confidence or 0.0),
        status=status or "",
        generated=bool(generated),
        extractive=bool(extractive),
        needs_clarification=bool(needs_clarification),
        total_time_ms=float(total_time_ms or 0.0),
        language=language or "fr",
    )

    session.add(entry)
    session.commit()

    return int(entry.id)


def record_feedback(
    session: Session,
    interaction_id: int,
    helpful: bool,
) -> bool:
    """Rattache l'avis du citoyen à une interaction."""
    entry = session.get(ConversationLog, interaction_id)

    if entry is None:
        return False

    entry.feedback = "useful" if helpful else "not_useful"
    entry.feedback_at = datetime.now(timezone.utc)

    session.commit()

    return True


def build_feedback_stats(
    session: Session,
    since: datetime | None = None,
) -> dict[str, Any]:
    """Satisfaction déclarée par les citoyens."""
    statement = select(
        ConversationLog.feedback,
        func.count(ConversationLog.id),
    ).where(ConversationLog.feedback.is_not(None))

    if since is not None:
        statement = statement.where(
            ConversationLog.created_at >= since
        )

    rows = session.execute(
        statement.group_by(ConversationLog.feedback)
    ).all()

    counts = {row[0]: int(row[1] or 0) for row in rows}
    useful = counts.get("useful", 0)
    not_useful = counts.get("not_useful", 0)
    total = useful + not_useful

    return {
        "useful": useful,
        "not_useful": not_useful,
        "total": total,
        "satisfaction": (
            round(useful / total * 100) if total else None
        ),
    }
