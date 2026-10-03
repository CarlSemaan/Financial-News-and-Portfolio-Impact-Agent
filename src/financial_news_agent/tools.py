from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from .config import Settings, load_watchlist
from .memory import AlertMemory, JsonlNotificationSink
from .models import Alert, Company, SourceEvidence, parse_datetime
from .providers import FixtureProvider, LiveProvider


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
        return self.memory.check_duplicate(
            event_key=event_key,
            ticker=ticker.upper(),
            event_type=event_type,
            headline=headline,
            event_date=event_date,
        )

    def save_alert(self, alert_payload: dict[str, Any], event_type: str) -> dict[str, Any]:
        self._company(str(alert_payload["ticker"]))
        alert = Alert(
            event_key=str(alert_payload["event_key"]),
            ticker=str(alert_payload["ticker"]),
            company_name=str(alert_payload["company_name"]),
            headline=str(alert_payload["headline"]),
            event_date=parse_datetime(alert_payload["event_date"]),
            facts=str(alert_payload["facts"]),
            impact=str(alert_payload["impact"]),
            uncertainty=str(alert_payload["uncertainty"]),
            confidence=str(alert_payload["confidence"]),
            significance_score=int(alert_payload["significance_score"]),
            sources=tuple(SourceEvidence.from_dict(item) for item in alert_payload["sources"]),
            created_at=parse_datetime(alert_payload["created_at"]),
        )
        self.memory.save(alert, event_type)
        return {"saved": True, "event_key": alert.event_key}

    def notify_alert(self, alert_payload: dict[str, Any]) -> dict[str, Any]:
        alert = Alert(
            event_key=str(alert_payload["event_key"]),
            ticker=str(alert_payload["ticker"]),
            company_name=str(alert_payload["company_name"]),
            headline=str(alert_payload["headline"]),
            event_date=parse_datetime(alert_payload["event_date"]),
            facts=str(alert_payload["facts"]),
            impact=str(alert_payload["impact"]),
            uncertainty=str(alert_payload["uncertainty"]),
            confidence=str(alert_payload["confidence"]),
            significance_score=int(alert_payload["significance_score"]),
            sources=tuple(SourceEvidence.from_dict(item) for item in alert_payload["sources"]),
            created_at=parse_datetime(alert_payload["created_at"]),
        )
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
