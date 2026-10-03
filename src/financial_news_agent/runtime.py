from __future__ import annotations

from .agents import AlertMemoryAgent, NewsRetrievalAgent, VerificationImpactAgent
from .config import Settings
from .llm import DeterministicImpactModel, OllamaImpactModel
from .supervisor import SupervisorAgent
from .tools import ToolGateway


def build_supervisor(settings: Settings, tools: ToolGateway) -> SupervisorAgent:
    impact_model = (
        OllamaImpactModel(settings.ollama_url, settings.model)
        if settings.use_ollama
        else DeterministicImpactModel()
    )
    return SupervisorAgent(
        tools,
        NewsRetrievalAgent(tools),
        VerificationImpactAgent(
            impact_model,
            significance_threshold=settings.significance_threshold,
        ),
        AlertMemoryAgent(tools),
        max_parallel_companies=settings.max_parallel_companies,
    )
