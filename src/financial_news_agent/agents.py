from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from .llm import DeterministicImpactModel, ImpactModel
from .models import (
    Alert,
    CandidateEvent,
    Company,
    Decision,
    SourceEvidence,
    VerifiedAssessment,
)
from .safeguards import validate_output_text
from .scoring import (
    confidence_for,
    evidence_ids,
    jaccard_similarity,
    score_candidate,
    stable_event_key,
)
from .tools import ToolGateway


class NewsRetrievalAgent:
    def __init__(self, tools: ToolGateway):
        self.tools = tools

    async def run(
        self,
        company: Company,
        since: datetime,
        until: datetime,
        *,
        query_hint: str | None = None,
    ) -> tuple[CandidateEvent, ...]:
        news_task = self.tools.search_news(
            company.ticker, since.isoformat(), until.isoformat(), query_hint
        )
        official_task = self.tools.lookup_official_sources(
            company.ticker, since.isoformat(), until.isoformat()
        )
        news, official = await asyncio.gather(news_task, official_task)
        evidence = [SourceEvidence.from_dict(item) for item in [*news, *official]]
        clusters: list[list[SourceEvidence]] = []
        for item in sorted(evidence, key=lambda source: source.published_at, reverse=True):
            matching_cluster = None
            for cluster in clusters:
                representative = cluster[0]
                same_fixture_group = item.event_group == representative.event_group
                likely_same_live_event = (
                    item.event_type == representative.event_type
                    and abs((item.event_date - representative.event_date).total_seconds()) <= 2 * 86400
                    and jaccard_similarity(item.title, representative.title) >= 0.28
                )
                if same_fixture_group or likely_same_live_event:
                    matching_cluster = cluster
                    break
            if matching_cluster is None:
                clusters.append([item])
            else:
                matching_cluster.append(item)

        candidates: list[CandidateEvent] = []
        for items in clusters:
            ranked = sorted(
                items,
                key=lambda item: (
                    item.source_type != "official",
                    not item.credible,
                    -item.published_at.timestamp(),
                ),
            )
            primary = ranked[0]
            group = primary.event_group
            summary = primary.summary or primary.title
            candidates.append(
                CandidateEvent(
                    ticker=company.ticker,
                    company_name=company.name,
                    event_group=group,
                    event_type=primary.event_type,
                    headline=primary.title,
                    event_date=min(item.event_date for item in items),
                    factual_summary=summary,
                    evidence=tuple(sorted(items, key=lambda item: item.published_at, reverse=True)),
                )
            )
        return tuple(sorted(candidates, key=lambda item: item.event_date, reverse=True))


class VerificationImpactAgent:
    def __init__(
        self,
        impact_model: ImpactModel,
        *,
        significance_threshold: int,
        fallback_model: ImpactModel | None = None,
    ):
        self.impact_model = impact_model
        self.significance_threshold = significance_threshold
        self.fallback_model = fallback_model or DeterministicImpactModel()

    def _is_relevant(self, candidate: CandidateEvent, company: Company) -> bool:
        combined = " ".join(
            [candidate.headline, candidate.factual_summary, *(item.title for item in candidate.evidence)]
        ).lower()
        return any(term.lower() in combined for term in (company.ticker, company.name, *company.query_terms))

    async def run(
        self,
        candidate: CandidateEvent,
        company: Company,
        since: datetime,
    ) -> VerifiedAssessment:
        fresh = candidate.event_date >= since
        relevant = self._is_relevant(candidate, company)
        usable_evidence = tuple(item for item in candidate.evidence if not item.injection_flag)
        has_official = any(item.source_type == "official" for item in usable_evidence)
        credible_sources = {item.source_name.lower() for item in usable_evidence if item.credible}
        verified = bool(usable_evidence) and (has_official or len(credible_sources) >= 2)
        score = score_candidate(candidate)
        significant = score >= self.significance_threshold
        safeguards: list[str] = []
        if any(item.injection_flag for item in candidate.evidence):
            safeguards.append("prompt_injection_removed")

        payload = {
            "ticker": candidate.ticker,
            "company": candidate.company_name,
            "event_type": candidate.event_type,
            "headline": candidate.headline,
            "factual_summary": candidate.factual_summary,
            "evidence": [
                {
                    "evidence_id": item.evidence_id,
                    "source_name": item.source_name,
                    "source_type": item.source_type,
                    "title": item.title,
                    "summary": item.summary,
                }
                for item in usable_evidence
            ],
        }
        try:
            draft = await asyncio.to_thread(self.impact_model.analyze, payload)
            validate_output_text(" ".join([draft.impact_explanation, draft.uncertainty]))
        except Exception:
            draft = self.fallback_model.analyze(payload)
            safeguards.append("model_output_replaced_with_safe_fallback")

        reason = None
        if not fresh:
            reason = "event is outside the requested time window"
        elif not relevant:
            reason = "event is not relevant to the approved company"
        elif not verified:
            reason = "material claim lacks an official source or two credible independent sources"
        elif not significant:
            reason = f"significance score {score} is below threshold {self.significance_threshold}"

        return VerifiedAssessment(
            event_key=stable_event_key(candidate),
            candidate=candidate,
            verified=verified and fresh and relevant,
            significant=significant,
            significance_score=score,
            confidence=confidence_for(
                verified=verified,
                evidence_count=len(usable_evidence),
                has_official=has_official,
            ),
            affected_drivers=draft.affected_drivers,
            impact_explanation=draft.impact_explanation,
            uncertainty=draft.uncertainty,
            supporting_evidence_ids=evidence_ids(usable_evidence),
            suppression_reason=reason,
            safeguards_triggered=tuple(safeguards),
        )


class AlertMemoryAgent:
    def __init__(self, tools: ToolGateway):
        self.tools = tools

    async def run(self, assessment: VerifiedAssessment) -> Decision:
        candidate = assessment.candidate
        if assessment.suppression_reason:
            return Decision(
                ticker=candidate.ticker,
                event_group=candidate.event_group,
                event_key=assessment.event_key,
                outcome="suppressed",
                reason=assessment.suppression_reason,
                significance_score=assessment.significance_score,
            )

        duplicate = await self.tools.check_alert_memory(
            assessment.event_key,
            candidate.ticker,
            candidate.event_type,
            candidate.headline,
            candidate.event_date.isoformat(),
        )
        if duplicate["duplicate"]:
            return Decision(
                ticker=candidate.ticker,
                event_group=candidate.event_group,
                event_key=assessment.event_key,
                outcome="suppressed",
                reason=f"duplicate of {duplicate['matched_event_key']} ({duplicate['match_type']} match)",
                significance_score=assessment.significance_score,
            )

        allowed_ids = set(assessment.supporting_evidence_ids)
        sources = tuple(item for item in candidate.evidence if item.evidence_id in allowed_ids)
        alert = Alert(
            event_key=assessment.event_key,
            ticker=candidate.ticker,
            company_name=candidate.company_name,
            headline=candidate.headline,
            event_date=candidate.event_date,
            facts=candidate.factual_summary,
            impact=assessment.impact_explanation,
            uncertainty=assessment.uncertainty,
            confidence=assessment.confidence,
            significance_score=assessment.significance_score,
            sources=sources,
        )
        validate_output_text(" ".join([alert.headline, alert.facts, alert.impact, alert.uncertainty]))
        await self.tools.save_alert(alert.to_dict(), candidate.event_type)
        await self.tools.notify_alert(alert.to_dict())
        return Decision(
            ticker=candidate.ticker,
            event_group=candidate.event_group,
            event_key=assessment.event_key,
            outcome="alerted",
            reason="new, significant, and adequately supported event",
            significance_score=assessment.significance_score,
            alert=alert,
        )
