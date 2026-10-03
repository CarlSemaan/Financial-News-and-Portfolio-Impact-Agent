# Financial News and Portfolio Impact Agent

This repository contains a working multi-agent decision-support system for a small, approved equity
watchlist. It retrieves recent news and official filings, verifies material claims, explains possible
business impact, suppresses duplicate or weak events, and writes source-linked alerts to a local
notification channel. It never places trades or produces buy, sell, or price-target recommendations.

Repository: <https://github.com/CarlSemaan/Financial-News-and-Portfolio-Impact-Agent>

## System architecture

The runtime follows the supervisor pattern developed for the project:

```text
                         +------------------+
Scheduled or CLI request | Supervisor Agent |
------------------------>|                  |
                         +--------+---------+
                                  |
                  one task per approved company
                                  |
                         +--------v---------+
                         | News Retrieval   |----> NewsSearch and OfficialSourceLookup
                         | Agent            |      tools
                         +--------+---------+
                                  |
                         +--------v---------+
                         | Verification and|
                         | Impact Agent    |----> deterministic score + optional Ollama
                         +--------+---------+
                                  |
                         +--------v---------+
                         | Alert and Memory|----> SQLite duplicate check + JSONL notification
                         | Agent           |
                         +-----------------+
```

- **Supervisor Agent:** validates the time window, loads the approved watchlist, runs company tasks
  concurrently, and records every release or suppression decision.
- **News Retrieval Agent:** calls the news and official-source tools and returns structured candidate
  events with dates, links, summaries, and source metadata.
- **Verification and Impact Agent:** rejects stale and irrelevant records, requires an official source
  or two credible independent sources for a material claim, applies a transparent significance score,
  and produces a cautious impact explanation.
- **Alert and Memory Agent:** checks exact and similar prior events, releases only validated alerts,
  saves their evidence in SQLite, and sends them to a local JSONL notification channel.

The same controlled tool implementations are available through a Model Context Protocol (MCP) stdio
server. The normal in-process mode is convenient for development; `--tool-mode mcp` exercises the
same workflow through an actual MCP client/server connection.

## Requirements

- Python 3.11 or newer
- [Ollama](https://ollama.com/) for model-assisted impact interpretation in live mode
- Internet access for Google News RSS and SEC filing feeds in live mode

No API key is required by the included connectors. SEC requests must identify the client with a real
contact address through `SEC_USER_AGENT`.

## Install

Clone the public repository and enter it:

```bash
git clone https://github.com/CarlSemaan/Financial-News-and-Portfolio-Impact-Agent.git
cd Financial-News-and-Portfolio-Impact-Agent
```

Create and activate a virtual environment.

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Pull the tested local model:

```bash
ollama pull qwen2.5:7b
```

Copy `.env.example` to `.env` if you want a reference file, then set the variables in your shell. The
application reads environment variables directly; it does not load or commit secrets.

Windows PowerShell example:

```powershell
$env:SEC_USER_AGENT = "FinancialNewsAgent/1.0 your-real-email@example.com"
$env:NEWS_AGENT_MODEL = "qwen2.5:7b"
```

macOS or Linux example:

```bash
export SEC_USER_AGENT="FinancialNewsAgent/1.0 your-real-email@example.com"
export NEWS_AGENT_MODEL="qwen2.5:7b"
```

## Verify the installation offline

The repository contains a clearly labeled synthetic fixture. It exercises retrieval, verification,
impact analysis, duplicate suppression, prompt-injection handling, persistent memory, and delivery
without making a network request:

```bash
financial-news-agent demo --reset
```

Expected first-run summary:

```text
DEMO DATA ONLY - the events below are synthetic and are not current financial news.
Run ... completed: 3 alert(s), 4 suppressed, 0 error(s).
```

Run the demonstration again without `--reset`; the three previously released events should be
suppressed as duplicates. To exercise both Ollama and the MCP transport with the fixture:

```bash
financial-news-agent demo --reset --with-ollama --tool-mode mcp
```

## Run with live sources

Start Ollama in its normal local service mode, then run:

```bash
financial-news-agent run --hours 72 --tool-mode mcp
```

For a deterministic no-model fallback while testing live connectors:

```bash
financial-news-agent run --hours 72 --no-llm
```

Each run writes a structured audit record to `outputs/run-<uuid>.json`. Released alerts are persisted
in `data/alerts.db`, and the notification channel appends them to `data/notifications.jsonl`. These
runtime files are excluded from Git.

The default watchlist is `config/watchlist.json`. Keep it small and include the ticker, legal company
name, search terms, official domains, and zero-padded SEC CIK. A ticker not present in this file is
rejected before any search or memory operation.

## MCP server

Start the stdio server directly with:

```bash
financial-news-mcp
```

It exposes six tools:

| Tool | Purpose |
| --- | --- |
| `read_watchlist` | Read approved companies and monitoring metadata |
| `search_news` | Search recent Google News RSS records within a time window |
| `lookup_official_sources` | Retrieve recent SEC 8-K feed entries |
| `check_alert_memory` | Check exact and similar prior events |
| `save_alert` | Store a validated alert and evidence |
| `notify_alert` | Deliver a validated alert to the JSONL channel |

All tool arguments and returns are JSON-compatible. The application invokes these tools through MCP
when `--tool-mode mcp` is selected.

## Evaluation and tests

Run the unit and end-to-end test suite:

```bash
python -m unittest discover -s tests -v
```

Run the repeatable evaluation:

```bash
financial-news-agent evaluate
```

The evaluator uses seven synthetic event groups: three should alert and four should be suppressed.
It then repeats the same monitoring run to measure duplicate suppression. It reports alert precision,
alert recall, source-citation coverage, unsafe-output rate, prompt-injection handling, duplicate
suppression, latency, and errors to `outputs/evaluation_results.json`. These fixtures verify software
behavior; they do not estimate accuracy on the full distribution of live financial news.

## Safeguards

- A fixed watchlist bounds the system's authority and rejects unapproved tickers.
- News text is labeled and processed as untrusted data. Common prompt-injection phrases are removed
  before model use and flagged in the assessment trace.
- An official source or two credible independent sources are required for a material claim.
- A deterministic significance threshold controls publication; the language model cannot lower it.
- Output validation blocks direct trading instructions, price targets, guaranteed returns, and
  deterministic price predictions. A safe template replaces invalid model output.
- Stable event keys and headline similarity suppress repeated alerts.
- Every released alert contains source records and an explicit uncertainty statement.
- The only delivery action is a local file append. No brokerage or transaction capability exists.

## Data and privacy

The system stores event history locally in SQLite. It does not need portfolio holdings to monitor a
watchlist, so the default configuration avoids storing position sizes, account identifiers, or user
credentials. Delete `data/alerts.db` and `data/notifications.jsonl` to remove local run history. Use the
CLI `--reset` option only with demo data.

## Limitations

Google News RSS can omit sources, delay records, or return links through an aggregator. SEC 8-K feeds
cover only part of the information a company may publish. Source-name allowlists are imperfect, event
grouping is lexical, and deterministic significance rules can miss novel events or overvalue familiar
keywords. The local model can still generate a weak interpretation, although it cannot override the
verification and publication controls. The system requires human review before any investment action.

## Project layout

```text
config/                 approved watchlist
data/                   created at runtime for local memory and notifications
outputs/                created at runtime for run and evaluation records
src/financial_news_agent/
  agents.py             retrieval, verification, impact, alert and memory agents
  supervisor.py         orchestration and parallel company tasks
  providers.py          fixture, Google News RSS, and SEC connectors
  tools.py              controlled tool service and in-process gateway
  mcp_server.py         MCP tool exposure
  mcp_gateway.py        MCP client used by the agent runtime
  memory.py             SQLite event memory and notification sink
  safeguards.py         input and output controls
  evaluate.py           reproducible evaluation harness
tests/                   unit and end-to-end tests
```

## License

MIT. See `LICENSE`.
