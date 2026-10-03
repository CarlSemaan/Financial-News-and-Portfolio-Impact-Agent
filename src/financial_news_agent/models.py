from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def isoformat(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_datetime(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True)
class Company:
    ticker: str
    name: str
    query_terms: tuple[str, ...]
    official_domains: tuple[str, ...] = ()
    cik: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Company":
        ticker = str(data["ticker"]).strip().upper()
        if not ticker.isalnum() or len(ticker) > 10:
            raise ValueError(f"Invalid ticker: {ticker!r}")
        return cls(
            ticker=ticker,
            name=str(data["name"]).strip(),
            query_terms=tuple(str(x).strip() for x in data.get("query_terms", [ticker])),
            official_domains=tuple(str(x).lower().strip() for x in data.get("official_domains", [])),
            cik=str(data["cik"]).zfill(10) if data.get("cik") else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SourceEvidence:
    evidence_id: str
    source_type: Literal["news", "official"]
    source_name: str
    title: str
    url: str
    published_at: datetime
    event_date: datetime
    summary: str
    credible: bool
    event_group: str
    event_type: str
    injection_flag: bool = False

    def __post_init__(self) -> None:
        if not self.url.startswith(("http://", "https://")):
            raise ValueError("Evidence URL must use http or https")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SourceEvidence":
        return cls(
            evidence_id=str(data["evidence_id"]),
            source_type=str(data.get("source_type", "news")),
            source_name=str(data.get("source_name", "Unknown source")),
            title=str(data["title"]),
            url=str(data["url"]),
            published_at=parse_datetime(data["published_at"]),
            event_date=parse_datetime(data.get("event_date", data["published_at"])),
            summary=str(data.get("summary", "")),
            credible=bool(data.get("credible", False)),
            event_group=str(data.get("event_group", data["evidence_id"])),
            event_type=str(data.get("event_type", "other")),
            injection_flag=bool(data.get("injection_flag", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["published_at"] = isoformat(self.published_at)
        data["event_date"] = isoformat(self.event_date)
        return data


@dataclass(frozen=True)
class CandidateEvent:
    ticker: str
    company_name: str
    event_group: str
    event_type: str
    headline: str
    event_date: datetime
    factual_summary: str
    evidence: tuple[SourceEvidence, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "company_name": self.company_name,
            "event_group": self.event_group,
            "event_type": self.event_type,
            "headline": self.headline,
            "event_date": isoformat(self.event_date),
            "factual_summary": self.factual_summary,
            "evidence": [item.to_dict() for item in self.evidence],
        }


@dataclass(frozen=True)
class VerifiedAssessment:
    event_key: str
    candidate: CandidateEvent
    verified: bool
    significant: bool
    significance_score: int
    confidence: Literal["low", "medium", "high"]
    affected_drivers: tuple[str, ...]
    impact_explanation: str
    uncertainty: str
    supporting_evidence_ids: tuple[str, ...]
    suppression_reason: str | None = None
    safeguards_triggered: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["candidate"] = self.candidate.to_dict()
        return data


@dataclass(frozen=True)
class Alert:
    event_key: str
    ticker: str
    company_name: str
    headline: str
    event_date: datetime
    facts: str
    impact: str
    uncertainty: str
    confidence: str
    significance_score: int
    sources: tuple[SourceEvidence, ...]
    created_at: datetime = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_key": self.event_key,
            "ticker": self.ticker,
            "company_name": self.company_name,
            "headline": self.headline,
            "event_date": isoformat(self.event_date),
            "facts": self.facts,
            "impact": self.impact,
            "uncertainty": self.uncertainty,
            "confidence": self.confidence,
            "significance_score": self.significance_score,
            "sources": [item.to_dict() for item in self.sources],
            "created_at": isoformat(self.created_at),
        }


@dataclass(frozen=True)
class Decision:
    ticker: str
    event_group: str
    event_key: str
    outcome: Literal["alerted", "suppressed"]
    reason: str
    significance_score: int
    alert: Alert | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["alert"] = self.alert.to_dict() if self.alert else None
        return data


@dataclass(frozen=True)
class RunReport:
    run_id: str
    mode: str
    started_at: datetime
    completed_at: datetime
    decisions: tuple[Decision, ...]
    errors: tuple[str, ...] = ()

    @property
    def alerts(self) -> tuple[Alert, ...]:
        return tuple(d.alert for d in self.decisions if d.alert is not None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "mode": self.mode,
            "started_at": isoformat(self.started_at),
            "completed_at": isoformat(self.completed_at),
            "decisions": [item.to_dict() for item in self.decisions],
            "errors": list(self.errors),
            "summary": {
                "candidates": len(self.decisions),
                "alerts": len(self.alerts),
                "suppressed": len(self.decisions) - len(self.alerts),
            },
        }
