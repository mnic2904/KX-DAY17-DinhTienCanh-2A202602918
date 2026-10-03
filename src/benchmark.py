from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Load and minimally validate a benchmark dataset."""

    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Benchmark dataset must be a JSON list: {path}")
    required = {"id", "user_id", "turns", "recall_questions"}
    for index, conversation in enumerate(data):
        if not isinstance(conversation, dict) or not required.issubset(conversation):
            raise ValueError(f"Invalid conversation at index {index} in {path}")
        if not isinstance(conversation["turns"], list) or not isinstance(conversation["recall_questions"], list):
            raise ValueError(f"turns and recall_questions must be lists at index {index}")
    return data


def recall_points(answer: str, expected: list[str]) -> float:
    """Return 0, 0.5, or 1 for none, partial, or complete recall."""

    if not expected:
        return 1.0
    normalized = " ".join(answer.casefold().split())
    found = sum(" ".join(fact.casefold().split()) in normalized for fact in expected)
    if found == 0:
        return 0.0
    return 1.0 if found == len(expected) else 0.5


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Score fact coverage (80%) and concise, non-empty output (20%)."""

    if not answer.strip():
        return 0.0
    normalized = answer.casefold()
    coverage = (
        sum(fact.casefold() in normalized for fact in expected) / len(expected)
        if expected
        else 1.0
    )
    clarity = 1.0 if len(answer.split()) <= 120 else 0.5
    return round(0.8 * coverage + 0.2 * clarity, 2)


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    """Run identical turns and fresh-thread recall questions through one agent."""

    del config  # The agent already owns its fully resolved configuration.
    user_ids = {str(item["user_id"]) for item in conversations}
    memory_size = getattr(agent, "memory_file_size", None)
    before_bytes = sum(memory_size(user_id) for user_id in user_ids) if memory_size else 0

    thread_ids: set[str] = set()
    recall_scores: list[float] = []
    quality_scores: list[float] = []

    for conversation in conversations:
        user_id = str(conversation["user_id"])
        thread_id = str(conversation["id"])
        thread_ids.add(thread_id)
        for turn in conversation["turns"]:
            agent.reply(user_id, thread_id, str(turn))

        for index, recall in enumerate(conversation["recall_questions"]):
            recall_thread = f"{thread_id}-recall-{index}"
            thread_ids.add(recall_thread)
            result = agent.reply(user_id, recall_thread, str(recall["question"]))
            answer = result["response"] if isinstance(result, dict) else str(result)
            expected = [str(item) for item in recall.get("expected_contains", [])]
            recall_scores.append(recall_points(answer, expected))
            quality_scores.append(heuristic_quality(answer, expected))

    after_bytes = sum(memory_size(user_id) for user_id in user_ids) if memory_size else 0
    average = lambda values: sum(values) / len(values) if values else 0.0
    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=sum(agent.token_usage(thread_id) for thread_id in thread_ids),
        prompt_tokens_processed=sum(agent.prompt_token_usage(thread_id) for thread_id in thread_ids),
        recall_score=average(recall_scores),
        response_quality=average(quality_scores),
        memory_growth_bytes=max(0, after_bytes - before_bytes),
        compactions=sum(agent.compaction_count(thread_id) for thread_id in thread_ids),
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Render benchmark rows as a dependency-free Markdown table."""

    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    values = [
        [
            row.agent_name,
            str(row.agent_tokens_only),
            str(row.prompt_tokens_processed),
            f"{row.recall_score:.2f}",
            f"{row.response_quality:.2f}",
            str(row.memory_growth_bytes),
            str(row.compactions),
        ]
        for row in rows
    ]
    widths = [max(len(headers[i]), *(len(row[i]) for row in values)) for i in range(len(headers))]

    def line(items: list[str]) -> str:
        return "| " + " | ".join(item.ljust(widths[i]) for i, item in enumerate(items)) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([line(headers), separator, *(line(row) for row in values)])


def main() -> None:
    """Run the standard and long-context suites in deterministic offline mode."""

    config = load_config(Path(__file__).resolve().parent.parent)
    suites = [
        ("Standard Benchmark", config.data_dir / "conversations.json"),
        ("Long-Context Stress Benchmark", config.data_dir / "advanced_long_context.json"),
    ]
    for title, path in suites:
        conversations = load_conversations(path)
        with tempfile.TemporaryDirectory(prefix="memory-lab-benchmark-") as temp_dir:
            isolated_config = replace(config, state_dir=Path(temp_dir))
            rows = [
                run_agent_benchmark(
                    "Baseline",
                    BaselineAgent(isolated_config, force_offline=True),
                    conversations,
                    isolated_config,
                ),
                run_agent_benchmark(
                    "Advanced",
                    AdvancedAgent(isolated_config, force_offline=True),
                    conversations,
                    isolated_config,
                ),
            ]
        print(f"\n## {title}\n")
        print(format_rows(rows))


if __name__ == "__main__":
    main()
