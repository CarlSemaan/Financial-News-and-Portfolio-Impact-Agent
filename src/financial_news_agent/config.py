from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from .models import Company


@dataclass(frozen=True)
class Settings:
    project_root: Path
    mode: str
    watchlist_path: Path
    data_dir: Path
    ollama_url: str
    model: str
    use_ollama: bool
    sec_user_agent: str
    significance_threshold: int = 55
    max_parallel_companies: int = 4

    @classmethod
    def from_env(
        cls,
        *,
        project_root: Path | None = None,
        mode: str | None = None,
        use_ollama: bool | None = None,
        data_dir: Path | None = None,
    ) -> "Settings":
        root = (project_root or Path.cwd()).resolve()
        selected_mode = (mode or os.getenv("NEWS_AGENT_MODE", "live")).lower()
        if selected_mode not in {"demo", "live"}:
            raise ValueError("NEWS_AGENT_MODE must be 'demo' or 'live'")
        watchlist = Path(os.getenv("NEWS_AGENT_WATCHLIST", "config/watchlist.json"))
        if not watchlist.is_absolute():
            watchlist = root / watchlist
        chosen_data_dir = data_dir or Path(os.getenv("NEWS_AGENT_DATA_DIR", "data"))
        if not chosen_data_dir.is_absolute():
            chosen_data_dir = root / chosen_data_dir
        env_ollama = os.getenv("NEWS_AGENT_USE_OLLAMA", "true").lower() in {"1", "true", "yes"}
        return cls(
            project_root=root,
            mode=selected_mode,
            watchlist_path=watchlist,
            data_dir=chosen_data_dir,
            ollama_url=os.getenv("NEWS_AGENT_OLLAMA_URL", "http://localhost:11434").rstrip("/"),
            model=os.getenv("NEWS_AGENT_MODEL", "qwen2.5:7b"),
            use_ollama=env_ollama if use_ollama is None else use_ollama,
            sec_user_agent=os.getenv(
                "SEC_USER_AGENT", "FinancialNewsAgent/1.0 coursework@example.com"
            ),
        )

    def ensure_runtime_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.project_root / "outputs").mkdir(parents=True, exist_ok=True)


def load_watchlist(path: Path) -> tuple[Company, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    companies = tuple(Company.from_dict(item) for item in payload.get("companies", []))
    if not companies:
        raise ValueError(f"Watchlist is empty: {path}")
    tickers = [item.ticker for item in companies]
    if len(tickers) != len(set(tickers)):
        raise ValueError("Watchlist contains duplicate tickers")
    return companies
