from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A: conversation history is isolated to each thread."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}

        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Reply using live mode when available, otherwise deterministic offline mode."""

        if self.langchain_agent is None:
            return self._reply_offline(thread_id, message)

        state = self.sessions.setdefault(thread_id, SessionState())
        state.messages.append({"role": "user", "content": message})
        prompt_tokens = sum(estimate_tokens(item["content"]) for item in state.messages)
        result = self.langchain_agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": thread_id}},
        )
        response = self._result_text(result)
        return self._record_response(state, response, prompt_tokens)

    def token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Use only messages already present in this thread."""

        state = self.sessions.setdefault(thread_id, SessionState())
        state.messages.append({"role": "user", "content": message})
        prompt_tokens = sum(estimate_tokens(item["content"]) for item in state.messages)
        facts: dict[str, str] = {}
        for item in state.messages:
            if item["role"] == "user":
                facts.update(extract_profile_updates(item["content"]))

        response = self._offline_response(message, facts)
        return self._record_response(state, response, prompt_tokens)

    def _maybe_build_langchain_agent(self):
        """Build an optional live agent; missing SDK/config falls back offline."""

        try:
            from langchain.agents import create_agent
            from langgraph.checkpoint.memory import InMemorySaver

            return create_agent(
                model=build_chat_model(self.config.model),
                checkpointer=InMemorySaver(),
                system_prompt="Bạn là trợ lý hữu ích. Chỉ dùng lịch sử trong thread hiện tại.",
            )
        except (ImportError, ValueError):
            return None

    @staticmethod
    def _offline_response(message: str, facts: dict[str, str]) -> str:
        lowered = message.casefold()
        requested = {
            "name": any(term in lowered for term in ("tên", "là ai")),
            "profession": any(term in lowered for term in ("nghề", "công việc")),
            "location": any(term in lowered for term in ("ở đâu", "nơi ở")),
            "favorite_drink": "đồ uống" in lowered,
            "favorite_food": "món ăn" in lowered,
            "pet": any(term in lowered for term in ("nuôi con", "thú cưng", "corgi")),
            "response_style": any(term in lowered for term in ("style", "kiểu trả lời", "cách trả lời")),
            "interests": any(term in lowered for term in ("mối quan tâm", "quan tâm", "tóm tắt")),
        }
        labels = {
            "name": "Tên",
            "profession": "Nghề nghiệp",
            "location": "Nơi ở",
            "favorite_drink": "Đồ uống yêu thích",
            "favorite_food": "Món ăn yêu thích",
            "pet": "Thú cưng",
            "response_style": "Kiểu trả lời",
            "interests": "Mối quan tâm",
        }
        answers = [f"{labels[key]}: {facts[key]}" for key, wanted in requested.items() if wanted and key in facts]
        if any(requested.values()):
            return "; ".join(answers) + "." if answers else "Mình chưa có thông tin đó trong thread này."
        return "Mình đã ghi nhận trong thread hiện tại."

    @staticmethod
    def _record_response(state: SessionState, response: str, prompt_tokens: int) -> dict[str, Any]:
        response_tokens = estimate_tokens(response)
        state.prompt_tokens_processed += prompt_tokens
        state.token_usage += response_tokens
        state.messages.append({"role": "assistant", "content": response})
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
