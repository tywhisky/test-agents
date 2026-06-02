from dotenv import load_dotenv
import os

from agent_playground.agents.travel_assistant import DEFAULT_TRAVEL_PROMPT, create_travel_agent
from agent_playground.config import load_config
from agent_playground.llm import OpenAICompatibleClient
from agent_playground.memory import MemoryStore
from agent_playground.runner import run_agent


def execute() -> None:
    """Compatibility wrapper for the original learning-project import path."""
    load_dotenv()
    config = load_config(os.environ)
    if not config.is_complete:
        print("Missing required environment variables: " + ", ".join(config.missing_values))
        print("Add them to your .env file, then run `uv run python main.py` again.")
        return

    llm = OpenAICompatibleClient(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
    )
    agent = create_travel_agent(config.tavily_api_key)
    memory_store = MemoryStore()
    memory = memory_store.load()
    run_agent(agent, llm, DEFAULT_TRAVEL_PROMPT, memory=memory)
    memory_store.save(memory)
