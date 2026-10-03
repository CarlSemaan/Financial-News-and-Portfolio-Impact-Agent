from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from financial_news_agent.config import Settings
from financial_news_agent.mcp_gateway import MCPToolGateway


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class MCPTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_stdio_server_exposes_watchlist_and_news_tools(self) -> None:
        as_of = datetime(2026, 10, 1, 12, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings.from_env(
                project_root=PROJECT_ROOT,
                mode="demo",
                use_ollama=False,
                data_dir=Path(temp_dir),
            )
            async with MCPToolGateway(settings) as tools:
                watchlist = await tools.read_watchlist()
                news = await tools.search_news(
                    "AAPL",
                    (as_of - timedelta(days=7)).isoformat(),
                    as_of.isoformat(),
                )
            self.assertEqual({"AAPL", "MSFT", "TSLA"}, {item["ticker"] for item in watchlist})
            self.assertTrue(any(item["event_group"] == "aapl-demo-earnings" for item in news))


if __name__ == "__main__":
    unittest.main()
