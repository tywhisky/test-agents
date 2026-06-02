from agent_playground.prompts.travel import TRAVEL_AGENT_SYSTEM_PROMPT
from agent_playground.runner import Agent
from agent_playground.tools.attractions import create_attraction_tool
from agent_playground.tools.weather import get_weather


DEFAULT_TRAVEL_PROMPT = "Hello, please help me check today's weather in Beijing, and then recommend a suitable tourist attraction based on the weather."


def create_travel_agent(tavily_api_key: str) -> Agent:
    return Agent(
        name="travel-assistant",
        system_prompt=TRAVEL_AGENT_SYSTEM_PROMPT,
        tools={
            "get_weather": get_weather,
            "get_attraction": create_attraction_tool(tavily_api_key),
        },
    )

