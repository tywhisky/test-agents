import os
import asyncio
import json
from typing import Annotated

from dotenv import load_dotenv
from agent_framework import Agent, tool
from agent_framework.exceptions import ChatClientException
from agent_framework.openai import OpenAIChatCompletionClient
from pydantic import BaseModel

INSTRUCTIONS = """You are a luxury travel concierge named Alex. Your role is to:
1. Understand the traveler's preferences (budget, climate, activities)
2. Check destination availability before making recommendations
3. Provide detailed, personalized travel suggestions
4. Always mention visa requirements and best travel seasons
Be warm, professional, and enthusiastic about travel."""

DEFAULT_REQUEST = (
    "I'd love a week-long vacation somewhere with great food and history. "
    "Budget around $2500."
)

STRUCTURED_INSTRUCTIONS = (
    "You are a travel expert. Recommend destinations based on traveler preferences. "
    "Use the get_destination_details tool before producing the final recommendation. "
    "Return only data that fits the requested structured schema."
)

STRUCTURED_REQUEST = (
    "Recommend 3 destinations for a culture-loving traveler with a $2500 budget"
)

DESTINATIONS_TO_COMPARE = ("Barcelona", "Tokyo", "Cape Town")


class DestinationRecommendation(BaseModel):
    destination: str
    available: bool
    best_season: str
    highlights: list[str]
    estimated_budget_usd: int


class TravelRecommendations(BaseModel):
    recommendations: list[DestinationRecommendation]
    personalized_note: str


@tool(approval_mode="never_require")
def get_destination_details(destination: Annotated[str, "The destination to look up"]) -> str:
    """Get details about a vacation destination."""
    details = {
        "Barcelona": "Available. Best: May-Jun. Beach, architecture, nightlife. ~$2000/week",
        "Tokyo": "Available. Best: Mar-Apr. Culture, food, technology. ~$2500/week",
        "Cape Town": "Not available. Best: Nov-Mar. Nature, wine, adventure. ~$1800/week",
    }
    return details.get(destination, f"{destination}: No information available.")


def create_client() -> OpenAIChatCompletionClient | None:
    load_dotenv()
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("Missing required environment variable: DEEPSEEK_API_KEY")
        print("Add it to your .env file, then run `uv run python agents/travel_concierge.py` again.")
        return None

    return OpenAIChatCompletionClient(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
    )


def create_agent() -> Agent | None:
    client = create_client()
    if client is None:
        return None
    return Agent(client=client, instructions=INSTRUCTIONS)


def create_structured_agent() -> Agent | None:
    client = create_client()
    if client is None:
        return None
    return Agent(
        client=client,
        name="StructuredTravelExpert",
        instructions=STRUCTURED_INSTRUCTIONS,
        tools=[get_destination_details],
    )


def parse_travel_recommendations(text: str) -> TravelRecommendations:
    normalized = text.strip()
    if normalized.startswith("```"):
        lines = normalized.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        normalized = "\n".join(lines).strip()
    else:
        json_start = normalized.find("{")
        json_end = normalized.rfind("}")
        if json_start != -1 and json_end != -1 and json_start < json_end:
            normalized = normalized[json_start : json_end + 1]
    return TravelRecommendations.model_validate_json(normalized)


def destination_details_context(destinations: tuple[str, ...] = DESTINATIONS_TO_COMPARE) -> str:
    return "\n".join(
        f"{destination}: {get_destination_details(destination=destination)}"
        for destination in destinations
    )


def _json_fallback_prompt() -> str:
    schema = json.dumps(TravelRecommendations.model_json_schema(), indent=2)
    destination_context = destination_details_context()
    return (
        f"{STRUCTURED_REQUEST}\n\n"
        "Use only these get_destination_details results to ground each destination:\n"
        f"{destination_context}\n\n"
        "Create exactly one recommendation object for each destination above, including unavailable destinations.\n"
        "Return only valid JSON that matches this schema. Do not wrap it in Markdown.\n\n"
        f"{schema}"
    )


def _is_response_format_unavailable(error: ChatClientException) -> bool:
    return "response_format type is unavailable" in str(error)


async def run_structured_recommendations(agent: Agent) -> TravelRecommendations:
    try:
        response = await agent.run(
            STRUCTURED_REQUEST,
            options={"response_format": TravelRecommendations},
        )
    except ChatClientException as error:
        if not _is_response_format_unavailable(error):
            raise
        response = await agent.run(_json_fallback_prompt())

    if response.value is not None:
        return response.value
    return parse_travel_recommendations(response.text)


async def main():
    agent = create_structured_agent()
    if agent is None:
        return

    recommendations = await run_structured_recommendations(agent)
    print(recommendations.model_dump_json(indent=2))

if __name__ == "__main__":
    asyncio.run(main())
