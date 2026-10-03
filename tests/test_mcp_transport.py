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

    async def test_stdio_server_rejects_unapproved_notification(self) -> None:
        as_of = datetime(2026, 10, 1, 12, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings.from_env(
                project_root=PROJECT_ROOT,
                mode="demo",
                use_ollama=False,
                data_dir=Path(temp_dir),
            )
            payload = {
                "event_key": "unapproved-event",
                "ticker": "NVDA",
                "company_name": "NVIDIA Corporation",
                "headline": "NVIDIA reports quarterly results",
                "event_date": as_of.isoformat(),
                "facts": "NVIDIA reports quarterly results.",
                "impact": "The results may affect revenue expectations.",
                "uncertainty": "The size of the effect remains uncertain.",
                "confidence": "high",
                "significance_score": 90,
                "sources": [
                    {
                        "evidence_id": "nvda-official",
                        "source_type": "official",
                        "source_name": "Company filing",
                        "title": "NVIDIA reports quarterly results",
                        "url": "https://example.com/nvda-filing",
                        "published_at": as_of.isoformat(),
                        "event_date": as_of.isoformat(),
                        "summary": "NVIDIA reports quarterly results.",
                        "credible": True,
                        "event_group": "nvda-event",
                        "event_type": "earnings",
                        "injection_flag": False,
                    }
                ],
                "created_at": as_of.isoformat(),
            }
            async with MCPToolGateway(settings) as tools:
                with self.assertRaises(Exception) as raised:
                    await tools.notify_alert(payload)
            self.assertIn("outside the approved watchlist", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
