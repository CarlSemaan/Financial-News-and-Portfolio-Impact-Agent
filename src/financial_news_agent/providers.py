from __future__ import annotations

import hashlib
import html
import json
import re
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from importlib import resources
from typing import Any
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from .models import Company, isoformat
from .safeguards import sanitize_untrusted_text
from .scoring import canonical_tokens


CREDIBLE_SOURCE_NAMES = {
    "associated press",
    "the associated press",
    "bloomberg",
    "cnbc",
    "financial times",
    "reuters",
    "the wall street journal",
}


def _strip_html(value: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", html.unescape(value or ""))
    return " ".join(without_tags.split())


def infer_event_type(text: str) -> str:
    lowered = text.lower()
    rules = (
        ("earnings", ("earnings", "quarterly results", "revenue", "profit")),
        ("guidance", ("guidance", "forecast", "outlook")),
        ("merger_acquisition", ("acquisition", "acquire", "merger", "takeover")),
        ("regulatory", ("regulator", "regulatory", "investigation", "antitrust", "fine", "sec filing")),
        ("operations", ("outage", "shutdown", "factory", "plant", "disruption", "recall")),
        ("product", ("launch", "product", "service", "release")),
        ("leadership", ("ceo", "cfo", "resigns", "appointed", "leadership")),
        ("capital_allocation", ("dividend", "buyback", "debt", "offering")),
    )
    for event_type, terms in rules:
        if any(term in lowered for term in terms):
            return event_type
    return "other"


def _event_group(ticker: str, title: str, event_date: datetime, event_type: str) -> str:
    tokens = sorted(canonical_tokens(title))[:8]
    raw = f"{ticker}|{event_date.date()}|{event_type}|{' '.join(tokens)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


class FixtureProvider:
    def __init__(self) -> None:
        fixture = resources.files("financial_news_agent").joinpath("resources/demo_news.json")
        self.records = json.loads(fixture.read_text(encoding="utf-8"))["records"]

    def _materialize(self, record: dict[str, Any], until: datetime) -> dict[str, Any]:
        published = until - timedelta(hours=float(record.get("published_offset_hours", 0)))
        event_date = until - timedelta(days=float(record.get("event_offset_days", 0)))
        text, flagged = sanitize_untrusted_text(str(record.get("summary", "")))
        return {
            **record,
            "evidence_id": record["evidence_id"],
            "published_at": isoformat(published),
            "event_date": isoformat(event_date),
            "summary": text,
            "injection_flag": flagged,
        }

    def search_news(
        self, company: Company, since: datetime, until: datetime, query_hint: str | None = None
    ) -> list[dict[str, Any]]:
        del query_hint
        items = [self._materialize(x, until) for x in self.records if x["ticker"] == company.ticker]
        return [
            item
            for item in items
            if item["source_type"] == "news" and datetime.fromisoformat(item["published_at"].replace("Z", "+00:00")) >= since
        ]

    def official_sources(
        self, company: Company, since: datetime, until: datetime
    ) -> list[dict[str, Any]]:
        items = [self._materialize(x, until) for x in self.records if x["ticker"] == company.ticker]
        return [
            item
            for item in items
            if item["source_type"] == "official" and datetime.fromisoformat(item["published_at"].replace("Z", "+00:00")) >= since
        ]


class LiveProvider:
    def __init__(self, *, user_agent: str, timeout_seconds: int = 15):
        self.user_agent = user_agent
        self.timeout_seconds = timeout_seconds

    def _get(self, url: str) -> bytes:
        request = Request(url, headers={"User-Agent": self.user_agent, "Accept": "application/rss+xml, application/atom+xml, application/xml"})
        with urlopen(request, timeout=self.timeout_seconds) as response:
            return response.read()

    def search_news(
        self, company: Company, since: datetime, until: datetime, query_hint: str | None = None
    ) -> list[dict[str, Any]]:
        days = max(1, min(30, (until - since).days + 1))
        core_query = f'("{company.name}" OR {company.ticker})'
        if query_hint:
            core_query += f" {query_hint}"
        url = (
            "https://news.google.com/rss/search?q="
            + quote_plus(f"{core_query} when:{days}d")
            + "&hl=en-US&gl=US&ceid=US:en"
        )
        root = ElementTree.fromstring(self._get(url))
        records: list[dict[str, Any]] = []
        for index, item in enumerate(root.findall("./channel/item")[:20]):
            title = _strip_html(item.findtext("title") or "")
            link = (item.findtext("link") or "").strip()
            published_raw = item.findtext("pubDate") or ""
            try:
                published = parsedate_to_datetime(published_raw).astimezone(timezone.utc)
            except (TypeError, ValueError):
                published = until
            if published < since or published > until + timedelta(hours=1):
                continue
            description = _strip_html(item.findtext("description") or "")
            source_node = item.find("source")
            source_name = _strip_html(source_node.text if source_node is not None else "Unknown source")
            event_type = infer_event_type(f"{title} {description}")
            group = _event_group(company.ticker, title, published, event_type)
            summary, flagged = sanitize_untrusted_text(description)
            records.append(
                {
                    "evidence_id": f"news-{company.ticker.lower()}-{index}-{group[:6]}",
                    "ticker": company.ticker,
                    "source_type": "news",
                    "source_name": source_name,
                    "title": title,
                    "url": link,
                    "published_at": isoformat(published),
                    "event_date": isoformat(published),
                    "summary": summary,
                    "credible": source_name.lower() in CREDIBLE_SOURCE_NAMES,
                    "event_group": group,
                    "event_type": event_type,
                    "injection_flag": flagged,
                }
            )
        return records

    def official_sources(
        self, company: Company, since: datetime, until: datetime
    ) -> list[dict[str, Any]]:
        if not company.cik:
            return []
        url = (
            "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
            f"&CIK={quote_plus(company.cik)}&type=8-K&owner=exclude&count=20&output=atom"
        )
        root = ElementTree.fromstring(self._get(url))
        namespace = {"atom": "http://www.w3.org/2005/Atom"}
        records: list[dict[str, Any]] = []
        for index, entry in enumerate(root.findall("atom:entry", namespace)):
            title = _strip_html(entry.findtext("atom:title", default="", namespaces=namespace))
            updated_raw = entry.findtext("atom:updated", default="", namespaces=namespace)
            try:
                updated = datetime.fromisoformat(updated_raw.replace("Z", "+00:00")).astimezone(timezone.utc)
            except ValueError:
                continue
            if updated < since or updated > until + timedelta(hours=1):
                continue
            link_node = entry.find("atom:link", namespace)
            link = link_node.attrib.get("href", "") if link_node is not None else ""
            summary = _strip_html(entry.findtext("atom:summary", default="", namespaces=namespace))
            event_type = infer_event_type(f"{title} {summary}")
            group = _event_group(company.ticker, title, updated, event_type)
            safe_summary, flagged = sanitize_untrusted_text(summary)
            records.append(
                {
                    "evidence_id": f"sec-{company.ticker.lower()}-{index}-{group[:6]}",
                    "ticker": company.ticker,
                    "source_type": "official",
                    "source_name": "U.S. Securities and Exchange Commission",
                    "title": title,
                    "url": link,
                    "published_at": isoformat(updated),
                    "event_date": isoformat(updated),
                    "summary": safe_summary,
                    "credible": True,
                    "event_group": group,
                    "event_type": event_type,
                    "injection_flag": flagged,
                }
            )
        return records
