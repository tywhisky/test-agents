from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re


DEFAULT_MEMORY_PATH = Path(".agent_memory.json")


@dataclass
class UserMemory:
    preferred_attraction_types: list[str] = field(default_factory=list)
    budget_range: str | None = None
    rejected_recommendations: list[str] = field(default_factory=list)
    accepted_recommendations: list[str] = field(default_factory=list)
    consecutive_rejections: int = 0
    strategy_notes: list[str] = field(default_factory=list)

    @property
    def needs_strategy_reflection(self) -> bool:
        return self.consecutive_rejections >= 3

    def remember_preference(self, preference: str) -> None:
        cleaned = preference.strip()
        if cleaned and cleaned not in self.preferred_attraction_types:
            self.preferred_attraction_types.append(cleaned)

    def remember_budget(self, budget_range: str) -> None:
        cleaned = budget_range.strip()
        if cleaned:
            self.budget_range = cleaned

    def record_rejection(self, recommendation: str) -> None:
        cleaned = recommendation.strip()
        if cleaned:
            self.rejected_recommendations.append(cleaned)
        self.consecutive_rejections += 1

    def record_acceptance(self, recommendation: str) -> None:
        cleaned = recommendation.strip()
        if cleaned:
            self.accepted_recommendations.append(cleaned)
        self.consecutive_rejections = 0

    def add_strategy_note(self, note: str) -> None:
        cleaned = note.strip()
        if cleaned:
            self.strategy_notes.append(cleaned)

    def reflection_hint(self) -> str:
        if not self.needs_strategy_reflection:
            return ""
        rejected = ", ".join(self.rejected_recommendations[-3:])
        return (
            "The user rejected 3 consecutive recommendations"
            f" ({rejected}). Reflect on why they may not fit, then change strategy."
        )

    def to_prompt_context(self) -> str:
        lines = ["Known user memory:"]
        if self.preferred_attraction_types:
            lines.append("- Preferred attraction types: " + ", ".join(self.preferred_attraction_types))
        if self.budget_range:
            lines.append("- Budget range: " + self.budget_range)
        if self.rejected_recommendations:
            lines.append("- Recently rejected: " + ", ".join(self.rejected_recommendations[-3:]))
        if self.accepted_recommendations:
            lines.append("- Previously accepted: " + ", ".join(self.accepted_recommendations[-3:]))
        if self.strategy_notes:
            lines.append("- Strategy notes: " + " | ".join(self.strategy_notes[-3:]))
        reflection = self.reflection_hint()
        if reflection:
            lines.append("- Reflection required: " + reflection)
        if len(lines) == 1:
            lines.append("- No preferences remembered yet.")
        return "\n".join(lines)


class MemoryStore:
    def __init__(self, path: Path = DEFAULT_MEMORY_PATH):
        self.path = path

    def load(self) -> UserMemory:
        if not self.path.exists():
            return UserMemory()

        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return UserMemory()

        return UserMemory(
            preferred_attraction_types=list(data.get("preferred_attraction_types", [])),
            budget_range=data.get("budget_range"),
            rejected_recommendations=list(data.get("rejected_recommendations", [])),
            accepted_recommendations=list(data.get("accepted_recommendations", [])),
            consecutive_rejections=int(data.get("consecutive_rejections", 0)),
            strategy_notes=list(data.get("strategy_notes", [])),
        )

    def save(self, memory: UserMemory) -> None:
        self.path.write_text(
            json.dumps(asdict(memory), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def update_memory_from_user_text(memory: UserMemory, text: str) -> None:
    lowered = text.lower()
    if "historical" in lowered or "history" in lowered:
        memory.remember_preference("historical")
    if "cultural" in lowered or "culture" in lowered:
        memory.remember_preference("cultural")
    if "museum" in lowered:
        memory.remember_preference("museum")
    if "outdoor" in lowered or "nature" in lowered:
        memory.remember_preference("outdoor")

    budget_match = re.search(r"(\d+\s*-\s*\d+\s*(?:rmb|yuan|\u5143))", text, re.IGNORECASE)
    if budget_match:
        memory.remember_budget(_normalize_budget_range(budget_match.group(1)))


def _normalize_budget_range(raw_budget: str) -> str:
    normalized = re.sub(r"\s*-\s*", "-", raw_budget.strip())
    return re.sub(r"\s+", " ", normalized).replace("rmb", "RMB").replace("yuan", "RMB")
