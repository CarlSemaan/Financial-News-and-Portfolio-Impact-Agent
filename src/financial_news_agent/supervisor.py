from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone

from .agents import AlertMemoryAgent, NewsRetrievalAgent, VerificationImpactAgent
from .models import Company, Decision, RunReport
from .tools import ToolGateway


class SupervisorAgent:
    def __init__(
        self,
        tools: ToolGateway,
        retrieval_agent: NewsRetrievalAgent,
        verification_agent: VerificationImpactAgent,
        alert_agent: AlertMemoryAgent,
        *,
        max_parallel_companies: int = 4,
    ):
        self.tools = tools
        self.retrieval_agent = retrieval_agent
        self.verification_agent = verification_agent
        self.alert_agent = alert_agent
        self.semaphore = asyncio.Semaphore(max_parallel_companies)

    async def _monitor_company(
        self, company: Company, since: datetime, until: datetime
    ) -> tuple[list[Decision], list[str]]:
        decisions: list[Decision] = []
        errors: list[str] = []
        async with self.semaphore:
            try:
                candidates = await self.retrieval_agent.run(company, since, until)
                if not candidates:
                    candidates = await self.retrieval_agent.run(
                        company,
                        since,
                        until,
                        query_hint="earnings filing investigation acquisition outage",
                    )
                for candidate in candidates:
                    assessment = await self.verification_agent.run(candidate, company, since)
                    decisions.append(await self.alert_agent.run(assessment))
            except Exception as exc:
                errors.append(f"{company.ticker}: {type(exc).__name__}: {exc}")
        return decisions, errors

    async def run(
        self,
        *,
        time_window_hours: int = 72,
        as_of: datetime | None = None,
        mode: str = "live",
    ) -> RunReport:
        if not 1 <= time_window_hours <= 24 * 30:
            raise ValueError("time_window_hours must be between 1 and 720")
        started = datetime.now(timezone.utc)
        until = (as_of or started).astimezone(timezone.utc)
        since = until - timedelta(hours=time_window_hours)
        companies = tuple(Company.from_dict(item) for item in await self.tools.read_watchlist())
        results = await asyncio.gather(
            *(self._monitor_company(company, since, until) for company in companies)
        )
        decisions = tuple(item for group, _ in results for item in group)
        errors = tuple(error for _, group_errors in results for error in group_errors)
        return RunReport(
            run_id=str(uuid.uuid4()),
            mode=mode,
            started_at=started,
            completed_at=datetime.now(timezone.utc),
            decisions=decisions,
            errors=errors,
        )
