from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class AppConfig:
    api_key: str | None
    base_url: str | None
    model: str | None
    tavily_api_key: str | None
    missing_values: list[str]

    @property
    def is_complete(self) -> bool:
        return not self.missing_values


def load_config(env: Mapping[str, str]) -> AppConfig:
    api_key = env.get("API_KEY") or env.get("LLM_API_KEY")
    base_url = env.get("BASE_URL") or env.get("LLM_BASE_URL")
    model = env.get("MODEL_ID") or env.get("MODEL_NAME") or env.get("LLM_MODEL_ID")
    tavily_api_key = env.get("TAVILY_API_KEY")

    missing_values = [
        name
        for name, value in {
            "API_KEY": api_key,
            "BASE_URL": base_url,
            "MODEL_ID or MODEL_NAME": model,
            "TAVILY_API_KEY": tavily_api_key,
        }.items()
        if not value
    ]

    return AppConfig(
        api_key=api_key,
        base_url=base_url,
        model=model,
        tavily_api_key=tavily_api_key,
        missing_values=missing_values,
    )
