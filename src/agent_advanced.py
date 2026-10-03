from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent B with persistent profile facts and compact thread memory."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}

        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Route to deterministic offline mode or an optional live model."""

        if self.langchain_agent is None:
            return self._reply_offline(user_id, thread_id, message)

        self._remember_profile_updates(user_id, message)
        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        profile = self.profile_store.read_text(user_id)
        context = self.compact_memory.context(thread_id)
        result = self.langchain_agent.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "Bạn là trợ lý có bộ nhớ người dùng. Ưu tiên fact mới nhất, "
                        "không suy diễn fact không có trong ngữ cảnh.\n\n"
                        f"Hồ sơ người dùng:\n{profile}\n\n"
                        f"Tóm tắt hội thoại cũ:\n{context['summary']}"
                    ),
                },
                *context["messages"],
            ]
        )
        response = self._result_text(result)
        return self._record_response(thread_id, response, prompt_tokens)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Persist stable facts and reply from profile plus compact context."""

        self._remember_profile_updates(user_id, message)
        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        response = self._offline_response(user_id, thread_id, message)
        return self._record_response(thread_id, response, prompt_tokens)

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Estimate profile, summary, and recent-message context for one turn."""

        context = self.compact_memory.context(thread_id)
        parts = [self.profile_store.read_text(user_id), str(context["summary"])]
        parts.extend(str(item.get("content", "")) for item in context["messages"])
        return estimate_tokens("\n".join(parts))

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        """Answer recall questions from the latest persisted facts."""

        facts = self.profile_store.facts(user_id)
        lowered = message.casefold()
        requested = {
            "name": any(term in lowered for term in ("tên", "là ai")),
            "profession": any(term in lowered for term in ("nghề", "công việc", "product manager")),
            "location": any(term in lowered for term in ("ở đâu", "nơi ở", "còn ở", "huế", "hà nội")),
            "favorite_drink": "đồ uống" in lowered,
            "favorite_food": "món ăn" in lowered,
            "pet": any(term in lowered for term in ("nuôi con", "thú cưng", "corgi")),
            "response_style": any(term in lowered for term in ("style", "kiểu trả lời", "cách trả lời")),
            "interests": any(term in lowered for term in ("mối quan tâm", "quan tâm", "tóm tắt")),
        }
        labels = {
            "name": "Tên",
            "profession": "Nghề nghiệp hiện tại",
            "location": "Nơi ở hiện tại",
            "favorite_drink": "Đồ uống yêu thích",
            "favorite_food": "Món ăn yêu thích",
            "pet": "Thú cưng",
            "response_style": "Kiểu trả lời",
            "interests": "Mối quan tâm",
        }
        answers = [f"{labels[key]}: {facts[key]}" for key, wanted in requested.items() if wanted and key in facts]
        if any(requested.values()):
            return "; ".join(answers) + "." if answers else "Mình chưa có thông tin đó trong hồ sơ."
        return "Mình đã cập nhật các thông tin ổn định vào hồ sơ."

    def _maybe_build_langchain_agent(self):
        """Build an optional live model; memory is supplied explicitly per turn."""

        try:
            return build_chat_model(self.config.model)
        except (ImportError, ValueError):
            return None

    def _remember_profile_updates(self, user_id: str, message: str) -> None:
        existing = self.profile_store.facts(user_id)
        for key, value in extract_profile_updates(message).items():
            if key == "response_style" and key in existing:
                parts = [part.strip() for part in f"{existing[key]}, {value}".split(",")]
                value = ", ".join(dict.fromkeys(part for part in parts if part))
            self.profile_store.upsert_fact(user_id, key, value)
            existing[key] = value

    def _record_response(self, thread_id: str, response: str, prompt_tokens: int) -> dict[str, Any]:
        response_tokens = estimate_tokens(response)
        self.thread_prompt_tokens[thread_id] = self.prompt_token_usage(thread_id) + prompt_tokens
        self.thread_tokens[thread_id] = self.token_usage(thread_id) + response_tokens
        self.compact_memory.append(thread_id, "assistant", response)
        return {
            "response": response,
            "tokens": response_tokens,
            "prompt_tokens": prompt_tokens,
        }

    @staticmethod
    def _result_text(result: Any) -> str:
        messages = result.get("messages", []) if isinstance(result, dict) else []
        last = messages[-1] if messages else result
        content = last.get("content", "") if isinstance(last, dict) else getattr(last, "content", last)
        if isinstance(content, list):
            return "".join(
                str(part.get("text", "")) if isinstance(part, dict) else str(part)
                for part in content
            )
        return str(content)
