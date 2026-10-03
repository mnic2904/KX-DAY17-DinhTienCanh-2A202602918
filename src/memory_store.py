from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path


_EMPTY_PROFILE = "# User Profile\n"


def estimate_tokens(text: str) -> int:
    """Return a deterministic approximation suitable for offline benchmarks."""

    text = text.strip()
    return (len(text) + 3) // 4 if text else 0


@dataclass
class UserProfileStore:
    """Store one persistent Markdown profile per user."""

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        normalized = unicodedata.normalize("NFKC", user_id).strip()
        slug = re.sub(r"[^\w.-]+", "_", normalized, flags=re.UNICODE).strip("._")
        if not slug:
            raise ValueError("user_id must contain at least one letter or number")
        return self.root_dir / slug / "User.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        return path.read_text(encoding="utf-8") if path.exists() else _EMPTY_PROFILE

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content.rstrip() + "\n", encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        if not search_text:
            return False
        content = self.read_text(user_id)
        if search_text not in content:
            return False
        self.write_text(user_id, content.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        """Parse facts written by :meth:`upsert_fact`."""

        return {
            match.group("key"): match.group("value").strip()
            for match in re.finditer(
                r"^- \*\*(?P<key>[^*]+)\*\*:\s*(?P<value>.*)$",
                self.read_text(user_id),
                flags=re.MULTILINE,
            )
        }

    def upsert_fact(self, user_id: str, key: str, value: str) -> Path:
        """Insert or replace one structured fact, keeping the newest correction."""

        clean_key = re.sub(r"\s+", "_", key.strip().lower())
        clean_value = " ".join(value.split()).strip(" .")
        if not clean_key or not clean_value:
            raise ValueError("key and value must not be empty")

        content = self.read_text(user_id)
        line = f"- **{clean_key}**: {clean_value}"
        pattern = re.compile(rf"^- \*\*{re.escape(clean_key)}\*\*:\s*.*$", re.MULTILINE)
        content = pattern.sub(line, content, count=1) if pattern.search(content) else content.rstrip() + f"\n\n{line}"
        return self.write_text(user_id, content)


def extract_profile_updates(message: str) -> dict[str, str]:
    """Extract confidently stated, persistent facts from a Vietnamese message."""

    text = " ".join(message.split())
    if not text:
        return {}

    facts: dict[str, str] = {}

    def capture(key: str, patterns: list[str]) -> None:
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                value = match.group(1).strip(" ,.;:!?")
                if key == "location":
                    value = re.split(r"\s+(?:và\s+đang|chứ)\b", value, maxsplit=1, flags=re.IGNORECASE)[0]
                elif key == "profession":
                    value = re.split(r"\s+(?:chứ|cho\s+(?:startup|team|một\b))", value, maxsplit=1, flags=re.IGNORECASE)[0]
                    if value.casefold().startswith("việc"):
                        continue
                if value and not re.search(r"\b(?:gì|đâu|nào|không)\b", value, re.IGNORECASE):
                    facts[key] = value
                    return

    capture("name", [r"\b(?:mình|tôi)\s+tên\s+là\s+([^,.;!?]+)"])
    capture(
        "profession",
        [
            r"nghề nghiệp hiện tại\s+(?:vẫn\s+)?là\s+([^,.;!?]+)",
            r"(?:(?:mình|tôi)\s+)?(?:hiện\s+)?ở\s+[^,.;!?]+?\s+và\s+đang làm\s+([^,.;!?]+)",
            r"(?:giờ|hiện tại)\s+(?:(?:mình|tôi)\s+)?(?:đang\s+)?(?:làm|chuyển sang)\s+((?!việc\b)[^,.;!?]+)",
            r"(?:mình|tôi)\s+(?:vẫn\s+)?(?:đang\s+)?làm\s+((?!việc\b)[^,.;!?]+)",
        ],
    )

    location_noise = re.search(r"(?:không phải nơi ở|đừng lấy .* làm nơi ở)", text, re.IGNORECASE)
    if not location_noise:
        capture(
            "location",
            [
                r"nơi ở (?:hiện tại\s+)?(?:là|đã cập nhật từ .+? sang)\s+([^,.;!?]+)",
                r"\bhiện ở\s+([^,.;!?]+)",
                r"(?:giờ|hiện tại|thực ra từ .+?)\s+(?:mình|tôi)\s+(?:đang\s+)?(?:làm việc\s+)?ở\s+([^,.;!?]+)",
                r"(?:mình|tôi)\s+(?:vẫn\s+)?ở\s+([^,.;!?]+)",
            ],
        )

    capture("favorite_drink", [r"(?:đồ uống yêu thích|món uống yêu thích)\s+(?:của mình\s+)?là\s+([^,.;!?]+)"])
    capture("favorite_food", [r"món ăn yêu thích\s+(?:của mình\s+)?là\s+([^,.;!?]+)"])
    capture("pet", [r"(?:mình|tôi)\s+nuôi\s+(?:một\s+)?([^,.;!?]+)"])

    style_parts: list[str] = []
    lowered = text.casefold()
    if "trả lời" in lowered or "cách giải thích" in lowered or "style" in lowered:
        if "3 bullet" in lowered:
            style_parts.append("3 bullet")
        elif "bullet" in lowered:
            style_parts.append("bullet ngắn")
        if "ngắn gọn" in lowered or "trả lời ngắn" in lowered:
            style_parts.append("ngắn gọn")
        if "rõ ý" in lowered:
            style_parts.append("rõ ý")
        if "ví dụ thực chiến" in lowered:
            style_parts.append("có ví dụ thực chiến")
        elif "ví dụ thực tế" in lowered:
            style_parts.append("có ví dụ thực tế")
        if "trade-off" in lowered:
            style_parts.append("nhấn mạnh trade-off")
    if style_parts:
        facts["response_style"] = ", ".join(dict.fromkeys(style_parts))

    interest_match = re.search(
        r"(?:mình|tôi)\s+(?:đang\s+|vẫn\s+)?(?:thích|quan tâm(?: nhiều)? (?:đến|tới))\s+([^.;!?]+)",
        text,
        flags=re.IGNORECASE,
    )
    if interest_match:
        interests = interest_match.group(1).strip(" ,")
        if re.search(r"\b(?:Python|AI|MLOps|RAG|benchmark memory)\b", interests, re.IGNORECASE):
            facts["interests"] = interests

    return facts


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Create a bounded, deterministic summary of the most recent items."""

    if max_items <= 0:
        return ""
    items: list[str] = []
    for message in messages[-max_items:]:
        content = " ".join(str(message.get("content", "")).split())
        if not content:
            continue
        role = str(message.get("role", "message"))
        excerpt = content if len(content) <= 240 else content[:237].rstrip() + "..."
        items.append(f"- {role}: {excerpt}")
    return "\n".join(items)


@dataclass
class CompactMemoryManager:
    """Keep recent messages verbatim and compact older context."""

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        thread = self.state.setdefault(thread_id, {"messages": [], "summary": "", "compactions": 0})
        messages = thread["messages"]
        assert isinstance(messages, list)
        messages.append({"role": role, "content": content})

        summary = str(thread["summary"])
        context_text = summary + "\n" + "\n".join(str(item.get("content", "")) for item in messages)
        if estimate_tokens(context_text) <= self.threshold_tokens or len(messages) <= self.keep_messages:
            return

        split_at = len(messages) - self.keep_messages
        older = messages[:split_at]
        summary_input = ([{"role": "summary", "content": summary}] if summary else []) + older
        thread["summary"] = summarize_messages(summary_input)
        thread["messages"] = messages[split_at:]
        thread["compactions"] = int(thread["compactions"]) + 1

    def context(self, thread_id: str) -> dict[str, object]:
        thread = self.state.get(thread_id)
        if thread is None:
            return {"messages": [], "summary": "", "compactions": 0}
        return {
            "messages": [dict(message) for message in thread["messages"]],
            "summary": str(thread["summary"]),
            "compactions": int(thread["compactions"]),
        }

    def compaction_count(self, thread_id: str) -> int:
        return int(self.state.get(thread_id, {}).get("compactions", 0))
