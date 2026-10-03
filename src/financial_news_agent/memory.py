from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections.abc import Iterator
from typing import Any

from .models import Alert
from .scoring import jaccard_similarity


SCHEMA = """
CREATE TABLE IF NOT EXISTS alerts (
    event_key TEXT PRIMARY KEY,
    ticker TEXT NOT NULL,
    event_date TEXT NOT NULL,
    event_type TEXT NOT NULL,
    headline TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_alerts_ticker_date
ON alerts(ticker, event_date);
"""


class AlertMemory:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._session() as connection:
            connection.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _session(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def reset(self) -> None:
        with self._session() as connection:
            connection.execute("DELETE FROM alerts")

    def check_duplicate(
        self,
        *,
        event_key: str,
        ticker: str,
        event_type: str,
        headline: str,
        event_date: str,
        similarity_threshold: float = 0.55,
        similarity_window_days: int = 7,
    ) -> dict[str, Any]:
        with self._session() as connection:
            exact = connection.execute(
                "SELECT event_key, headline FROM alerts WHERE event_key = ?", (event_key,)
            ).fetchone()
            if exact:
                return {
                    "duplicate": True,
                    "matched_event_key": exact["event_key"],
                    "match_type": "exact",
                    "similarity": 1.0,
                }
            rows = connection.execute(
                "SELECT event_key, headline, event_date FROM alerts WHERE ticker = ? AND event_type = ?",
                (ticker, event_type),
            ).fetchall()
        target_date = datetime.fromisoformat(event_date.replace("Z", "+00:00"))
        if target_date.tzinfo is None:
            target_date = target_date.replace(tzinfo=timezone.utc)
        best_key, best_score = None, 0.0
        for row in rows:
            stored_date = datetime.fromisoformat(row["event_date"].replace("Z", "+00:00"))
            if stored_date.tzinfo is None:
                stored_date = stored_date.replace(tzinfo=timezone.utc)
            if abs(target_date - stored_date) > timedelta(days=similarity_window_days):
                continue
            score = jaccard_similarity(headline, row["headline"])
            if score > best_score:
                best_key, best_score = row["event_key"], score
        return {
            "duplicate": best_score >= similarity_threshold,
            "matched_event_key": best_key if best_score >= similarity_threshold else None,
            "match_type": "similar" if best_score >= similarity_threshold else None,
            "similarity": round(best_score, 3),
        }

    def save(self, alert: Alert, event_type: str) -> None:
        payload = json.dumps(alert.to_dict(), ensure_ascii=False, sort_keys=True)
        with self._session() as connection:
            connection.execute(
                """
                INSERT INTO alerts(event_key, ticker, event_date, event_type, headline, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    alert.event_key,
                    alert.ticker,
                    alert.event_date.isoformat(),
                    event_type,
                    alert.headline,
                    payload,
                    alert.created_at.isoformat(),
                ),
            )

    def get_alert(self, event_key: str) -> dict[str, Any] | None:
        with self._session() as connection:
            row = connection.execute(
                "SELECT payload_json FROM alerts WHERE event_key = ?", (event_key,)
            ).fetchone()
        return json.loads(row["payload_json"]) if row else None

    def list_alerts(self) -> list[dict[str, Any]]:
        with self._session() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM alerts ORDER BY created_at DESC"
            ).fetchall()
        return [json.loads(row["payload_json"]) for row in rows]


class JsonlNotificationSink:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def send(self, alert: Alert) -> dict[str, Any]:
        with self._lock:
            if self._contains_unlocked(alert.event_key):
                return {
                    "delivered": True,
                    "channel": "jsonl",
                    "path": str(self.path),
                    "already_delivered": True,
                }
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(alert.to_dict(), ensure_ascii=False) + "\n")
        return {
            "delivered": True,
            "channel": "jsonl",
            "path": str(self.path),
            "already_delivered": False,
        }

    def contains(self, event_key: str) -> bool:
        with self._lock:
            return self._contains_unlocked(event_key)

    def _contains_unlocked(self, event_key: str) -> bool:
        if not self.path.exists():
            return False
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                if json.loads(line).get("event_key") == event_key:
                    return True
            except json.JSONDecodeError:
                continue
        return False
