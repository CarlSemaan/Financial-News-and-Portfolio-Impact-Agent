from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from .config import Settings, load_watchlist
from .memory import AlertMemory, JsonlNotificationSink
from .models import Alert, CandidateEvent, Company, parse_datetime
from .providers import FixtureProvider, LiveProvider
from .safeguards import validate_output_text, validate_source_text
from .scoring import EVENT_BASE_SCORES, stable_event_key


class ToolService:
    """Controlled tool implementations shared by the app and the MCP server."""

    def __init__(self, settings: Settings):
        self.settings = settings
        settings.ensure_runtime_dirs()
        self.companies = {item.ticker: item for item in load_watchlist(settings.watchlist_path)}
        self.provider = (
            FixtureProvider()
            if settings.mode == "demo"
            else LiveProvider(user_agent=settings.sec_user_agent)
        )
        self.memory = AlertMemory(settings.data_dir / "alerts.db")
        self.notifications = JsonlNotificationSink(settings.data_dir / "notifications.jsonl")

    def _company(self, ticker: str) -> Company:
        normalized = ticker.strip().upper()
        if normalized not in self.companies:
            raise ValueError(f"Ticker {normalized!r} is outside the approved watchlist")
        return self.companies[normalized]

    def read_watchlist(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self.companies.values()]

    def search_news(
        self, ticker: str, since: str, until: str, query_hint: str | None = None
    ) -> list[dict[str, Any]]:
        return self.provider.search_news(
            self._company(ticker), parse_datetime(since), parse_datetime(until), query_hint
        )

    def lookup_official_sources(self, ticker: str, since: str, until: str) -> list[dict[str, Any]]:
        return self.provider.official_sources(
            self._company(ticker), parse_datetime(since), parse_datetime(until)
        )

    def check_alert_memory(
        self, event_key: str, ticker: str, event_type: str, headline: str, event_date: str
    ) -> dict[str, Any]:
        self._company(ticker)
        result = self.memory.check_duplicate(
            event_key=event_key,
            ticker=ticker.upper(),
            event_type=event_type,
            headline=headline,
            event_date=event_date,
        )
        matched_event_key = result.get("matched_event_key")
        if result.get("duplicate") and matched_event_key:
            pending_alert = self.memory.get_alert(str(matched_event_key))
            if pending_alert is not None and not self.notifications.contains(str(matched_event_key)):
                return {
                    **result,
                    "duplicate": False,
                    "pending_delivery": True,
                    "pending_alert": pending_alert,
                }
        return {**result, "pending_delivery": False}

    def _validated_alert(self, alert_payload: dict[str, Any]) -> Alert:
        alert = Alert.from_dict(alert_payload)
        company = self._company(alert.ticker)
        if alert.company_name.casefold() != company.name.casefold():
            raise ValueError("Alert company name does not match the approved watchlist")
        if not alert.event_key.strip():
            raise ValueError("Alert event key must not be empty")
        if not all((alert.headline, alert.facts, alert.impact, alert.uncertainty)):
            raise ValueError("Alert headline, facts, impact, and uncertainty are required")
        if alert.confidence not in {"low", "medium", "high"}:
            raise ValueError("Alert confidence must be low, medium, or high")
        if not self.settings.significance_threshold <= alert.significance_score <= 100:
            raise ValueError("Alert significance score does not meet the publication threshold")
        if not alert.sources:
            raise ValueError("Alert must contain source evidence")
        evidence_ids = [item.evidence_id for item in alert.sources]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("Alert source evidence identifiers must be unique")
        if any(item.injection_flag for item in alert.sources):
            raise ValueError("Alert cannot contain prompt-injection-flagged evidence")
        has_official = any(item.source_type == "official" for item in alert.sources)
        credible_sources = {item.source_name.casefold() for item in alert.sources if item.credible}
        if not (has_official or len(credible_sources) >= 2):
            raise ValueError("Alert lacks an official source or two credible independent sources")
        validate_source_text(" ".join([alert.headline, alert.facts]))
        validate_output_text(" ".join([alert.impact, alert.uncertainty]))
        return alert

    def save_alert(self, alert_payload: dict[str, Any], event_type: str) -> dict[str, Any]:
        alert = self._validated_alert(alert_payload)
        normalized_event_type = event_type.strip()
        if normalized_event_type not in EVENT_BASE_SCORES:
            raise ValueError("Event type is not supported")
        if any(item.event_type != normalized_event_type for item in alert.sources):
            raise ValueError("Alert event type does not match its source evidence")
        candidate = CandidateEvent(
            ticker=alert.ticker,
            company_name=alert.company_name,
            event_group=alert.sources[0].event_group,
            event_type=normalized_event_type,
            headline=alert.headline,
            event_date=alert.event_date,
            factual_summary=alert.facts,
            evidence=alert.sources,
        )
        if stable_event_key(candidate) != alert.event_key:
            raise ValueError("Alert event key does not match the validated event fields")
        self.memory.save(alert, normalized_event_type)
        return {"saved": True, "event_key": alert.event_key}

    def notify_alert(self, alert_payload: dict[str, Any]) -> dict[str, Any]:
        alert = self._validated_alert(alert_payload)
        stored = self.memory.get_alert(alert.event_key)
        if stored is None:
            raise ValueError("Alert must pass validation and be saved before notification")
        if stored != alert.to_dict():
            raise ValueError("Notification payload does not match the validated stored alert")
        return self.notifications.send(alert)

    def reset_demo_state(self) -> None:
        self.memory.reset()
        notification_path = self.settings.data_dir / "notifications.jsonl"
        if notification_path.exists():
            notification_path.unlink()


class ToolGateway(Protocol):
    async def read_watchlist(self) -> list[dict[str, Any]]: ...
    async def search_news(self, ticker: str, since: str, until: str, query_hint: str | None = None) -> list[dict[str, Any]]: ...
    async def lookup_official_sources(self, ticker: str, since: str, until: str) -> list[dict[str, Any]]: ...
    async def check_alert_memory(self, event_key: str, ticker: str, event_type: str, headline: str, event_date: str) -> dict[str, Any]: ...
    async def save_alert(self, alert_payload: dict[str, Any], event_type: str) -> dict[str, Any]: ...
    async def notify_alert(self, alert_payload: dict[str, Any]) -> dict[str, Any]: ...


class InProcessToolGateway:
    def __init__(self, service: ToolService):
        self.service = service

    async def read_watchlist(self) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self.service.read_watchlist)

    async def search_news(self, ticker: str, since: str, until: str, query_hint: str | None = None) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self.service.search_news, ticker, since, until, query_hint)

    async def lookup_official_sources(self, ticker: str, since: str, until: str) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self.service.lookup_official_sources, ticker, since, until)

    async def check_alert_memory(self, event_key: str, ticker: str, event_type: str, headline: str, event_date: str) -> dict[str, Any]:
        return await asyncio.to_thread(
            self.service.check_alert_memory, event_key, ticker, event_type, headline, event_date
        )

    async def save_alert(self, alert_payload: dict[str, Any], event_type: str) -> dict[str, Any]:
        return await asyncio.to_thread(self.service.save_alert, alert_payload, event_type)

    async def notify_alert(self, alert_payload: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(self.service.notify_alert, alert_payload)
