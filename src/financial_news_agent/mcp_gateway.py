from __future__ import annotations

import json
import os
import sys
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .config import Settings


class MCPToolGateway:
    """Calls the project tools over a real MCP stdio session."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None

    async def __aenter__(self) -> "MCPToolGateway":
        environment = os.environ.copy()
        environment.update(
            {
                "NEWS_AGENT_MODE": self.settings.mode,
                "NEWS_AGENT_WATCHLIST": str(self.settings.watchlist_path),
                "NEWS_AGENT_DATA_DIR": str(self.settings.data_dir),
                "NEWS_AGENT_MODEL": self.settings.model,
                "NEWS_AGENT_OLLAMA_URL": self.settings.ollama_url,
                "NEWS_AGENT_USE_OLLAMA": "true" if self.settings.use_ollama else "false",
                "SEC_USER_AGENT": self.settings.sec_user_agent,
            }
        )
        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-m", "financial_news_agent.mcp_server"],
            env=environment,
        )
        self._stack = AsyncExitStack()
        read_stream, write_stream = await self._stack.enter_async_context(stdio_client(parameters))
        self._session = await self._stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await self._session.initialize()
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        if self._stack:
            await self._stack.aclose()
        self._stack = None
        self._session = None

    async def _call(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        if self._session is None:
            raise RuntimeError("MCPToolGateway must be used as an async context manager")
        result = await self._session.call_tool(name, arguments or {})
        if result.isError:
            message = " ".join(getattr(item, "text", "") for item in result.content)
            raise RuntimeError(f"MCP tool {name} failed: {message}")
        def unwrap(value: Any) -> Any:
            if isinstance(value, dict) and set(value) == {"result"}:
                return value["result"]
            return value

        if result.structuredContent is not None:
            return unwrap(result.structuredContent)
        for item in result.content:
            if getattr(item, "type", None) == "text":
                return unwrap(json.loads(item.text))
        raise RuntimeError(f"MCP tool {name} returned no JSON content")

    async def read_watchlist(self) -> list[dict[str, Any]]:
        return await self._call("read_watchlist")

    async def search_news(self, ticker: str, since: str, until: str, query_hint: str | None = None) -> list[dict[str, Any]]:
        return await self._call(
            "search_news",
            {"ticker": ticker, "since": since, "until": until, "query_hint": query_hint},
        )

    async def lookup_official_sources(self, ticker: str, since: str, until: str) -> list[dict[str, Any]]:
        return await self._call(
            "lookup_official_sources", {"ticker": ticker, "since": since, "until": until}
        )

    async def check_alert_memory(self, event_key: str, ticker: str, event_type: str, headline: str, event_date: str) -> dict[str, Any]:
        return await self._call(
            "check_alert_memory",
            {
                "event_key": event_key,
                "ticker": ticker,
                "event_type": event_type,
                "headline": headline,
                "event_date": event_date,
            },
        )

    async def save_alert(self, alert_payload: dict[str, Any], event_type: str) -> dict[str, Any]:
        return await self._call(
            "save_alert", {"alert_payload": alert_payload, "event_type": event_type}
        )

    async def notify_alert(self, alert_payload: dict[str, Any]) -> dict[str, Any]:
        return await self._call("notify_alert", {"alert_payload": alert_payload})
