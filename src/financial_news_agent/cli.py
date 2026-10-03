from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import Settings
from .evaluate import run_evaluation
from .mcp_gateway import MCPToolGateway
from .runtime import build_supervisor
from .tools import InProcessToolGateway, ToolService


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _print_report(payload: dict[str, Any], *, demo: bool) -> None:
    if demo:
        print("DEMO DATA ONLY - the events below are synthetic and are not current financial news.\n")
    summary = payload["summary"]
    print(
        f"Run {payload['run_id']} completed: {summary['alerts']} alert(s), "
        f"{summary['suppressed']} suppressed, {len(payload['errors'])} error(s)."
    )
    for decision in payload["decisions"]:
        marker = "ALERT" if decision["outcome"] == "alerted" else "SUPPRESS"
        print(
            f"[{marker}] {decision['ticker']} / {decision['event_group']} / "
            f"score {decision['significance_score']}: {decision['reason']}"
        )
        alert = decision.get("alert")
        if alert:
            print(f"  {alert['headline']}")
            print(f"  What happened: {alert['facts']}")
            print(f"  Why it may matter: {alert['impact']}")
            print(f"  Uncertainty: {alert['uncertainty']}")
            for source in alert["sources"]:
                print(f"  Source: {source['source_name']} - {source['url']}")


async def _execute(args: argparse.Namespace, settings: Settings) -> int:
    service = ToolService(settings)
    if args.reset:
        service.reset_demo_state()
    as_of = datetime.fromisoformat(args.as_of.replace("Z", "+00:00")) if args.as_of else None

    if args.tool_mode == "mcp":
        async with MCPToolGateway(settings) as tools:
            report = await build_supervisor(settings, tools).run(
                time_window_hours=args.hours, as_of=as_of, mode=settings.mode
            )
    else:
        tools = InProcessToolGateway(service)
        report = await build_supervisor(settings, tools).run(
            time_window_hours=args.hours, as_of=as_of, mode=settings.mode
        )

    payload = report.to_dict()
    output = PROJECT_ROOT / "outputs" / f"run-{report.run_id}.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _print_report(payload, demo=settings.mode == "demo")
    print(f"\nStructured run record: {output}")
    return 1 if report.errors else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="financial-news-agent",
        description="Run the Financial News and Portfolio Impact multi-agent system.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name, help_text in (
        ("demo", "Run the deterministic offline demonstration."),
        ("run", "Run against live Google News RSS and SEC filing feeds."),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("--hours", type=int, default=168 if name == "demo" else 72)
        command.add_argument("--as-of", help="UTC ISO timestamp used as the end of the window.")
        command.add_argument("--tool-mode", choices=("in-process", "mcp"), default="in-process")
        if name == "demo":
            command.add_argument("--reset", action="store_true", help="Clear prior demo alerts before the run.")
            command.add_argument("--with-ollama", action="store_true")
        else:
            command.set_defaults(reset=False)
            command.add_argument("--no-llm", action="store_true")

    subparsers.add_parser("evaluate", help="Run the repeatable synthetic evaluation suite.")
    subparsers.add_parser("list-alerts", help="List alerts stored in the local SQLite memory.")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "evaluate":
        output = PROJECT_ROOT / "outputs" / "evaluation_results.json"
        metrics = run_evaluation(PROJECT_ROOT, output)
        print(json.dumps(metrics, indent=2))
        print(f"\nEvaluation record: {output}")
        return 0
    if args.command == "list-alerts":
        settings = Settings.from_env(project_root=PROJECT_ROOT)
        print(json.dumps(ToolService(settings).memory.list_alerts(), indent=2))
        return 0

    mode = "demo" if args.command == "demo" else "live"
    use_ollama = args.with_ollama if mode == "demo" else not args.no_llm
    settings = Settings.from_env(
        project_root=PROJECT_ROOT,
        mode=mode,
        use_ollama=use_ollama,
    )
    return asyncio.run(_execute(args, settings))


if __name__ == "__main__":
    sys.exit(main())
