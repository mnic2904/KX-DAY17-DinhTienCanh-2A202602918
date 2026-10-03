from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config
from memory_store import UserProfileStore


def make_config(tmp_path: Path):
    """Build a fast, isolated configuration for one test."""

    config = load_config(tmp_path)
    return replace(
        config,
        state_dir=tmp_path / "state",
        compact_threshold_tokens=100,
        compact_keep_messages=2,
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """`User.md` supports creation, reading, editing, and fact correction."""

    config = make_config(tmp_path)
    store = UserProfileStore(config.state_dir / "profiles")

    assert store.read_text("dungct") == "# User Profile\n"
    path = store.write_text("dungct", "# User Profile\n\n- **location**: Huế")
    assert path == config.state_dir / "profiles" / "dungct" / "User.md"
    assert "Huế" in store.read_text("dungct")
    assert store.edit_text("dungct", "Huế", "Đà Nẵng") is True
    assert store.edit_text("dungct", "không tồn tại", "x") is False
    assert "Đà Nẵng" in store.read_text("dungct")

    store.upsert_fact("dungct", "profession", "backend engineer")
    store.upsert_fact("dungct", "profession", "MLOps engineer")
    assert store.facts("dungct")["profession"] == "MLOps engineer"
    assert store.read_text("dungct").count("**profession**") == 1
    assert store.file_size("dungct") > 0


def test_compact_trigger(tmp_path: Path) -> None:
    """Long threads move old messages into a summary."""

    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    thread_id = "compact-thread"
    for index in range(8):
        agent.reply(
            "dungct",
            thread_id,
            f"Lượt {index}: " + "đây là nội dung dài để kích hoạt compact memory " * 8,
        )

    context = agent.compact_memory.context(thread_id)
    assert agent.compaction_count(thread_id) > 0
    assert context["summary"]
    assert len(context["messages"]) <= agent.config.compact_keep_messages


def test_cross_session_recall(tmp_path: Path) -> None:
    """Advanced persists facts across sessions while baseline forgets new threads."""

    config = make_config(tmp_path)
    fact = "Mình tên là DũngCT. Đồ uống yêu thích là cà phê sữa đá."
    question = "Mình tên gì và đồ uống yêu thích là gì?"

    first_advanced = AdvancedAgent(config, force_offline=True)
    first_advanced.reply("dungct", "advanced-old", fact)
    restarted_advanced = AdvancedAgent(config, force_offline=True)
    advanced_answer = restarted_advanced.reply("dungct", "advanced-new", question)["response"]

    baseline = BaselineAgent(config, force_offline=True)
    baseline.reply("dungct", "baseline-old", fact)
    baseline_answer = baseline.reply("dungct", "baseline-new", question)["response"]

    assert "DũngCT" in advanced_answer
    assert "cà phê sữa đá" in advanced_answer
    assert "DũngCT" not in baseline_answer
    assert "cà phê sữa đá" not in baseline_answer


def test_conflict_handling_keeps_latest_stable_facts(tmp_path: Path) -> None:
    """Corrections replace old facts while obvious location/job noise is ignored."""

    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    user_id = "correction-user"
    agent.reply(
        user_id,
        "old-thread",
        "Mình hiện ở Huế và đang làm backend engineer cho một nhóm AI.",
    )
    agent.reply(
        user_id,
        "old-thread",
        "Nơi ở hiện tại là Đà Nẵng. Giờ mình chuyển sang MLOps engineer.",
    )
    agent.reply(
        user_id,
        "old-thread",
        "Hà Nội chỉ là nơi mình họp chứ không phải nơi ở; product manager chỉ là câu đùa.",
    )

    answer = agent.reply(
        user_id,
        "new-thread",
        "Nghề nghiệp và nơi ở hiện tại của mình là gì?",
    )["response"]

    assert "MLOps engineer" in answer
    assert "Đà Nẵng" in answer
    assert "backend engineer" not in answer
    assert "Huế" not in answer
    assert "Hà Nội" not in answer
    assert "product manager" not in answer


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Compaction lowers cumulative prompt processing for identical long input."""

    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    thread_id = "long-thread"

    for index in range(12):
        message = f"Lượt benchmark {index}: " + "ngữ cảnh kỹ thuật dài và lặp lại " * 15
        baseline.reply("stress-user", thread_id, message)
        advanced.reply("stress-user", thread_id, message)

    assert advanced.compaction_count(thread_id) > 0
    assert advanced.prompt_token_usage(thread_id) < baseline.prompt_token_usage(thread_id)
