from agent_playground.prompts.travel import TRAVEL_AGENT_SYSTEM_PROMPT
from agent_playground.runner import Agent
from agent_playground.tools.attractions import create_attraction_tool
from agent_playground.tools.travel_recommendations import (
    TicketAvailability,
    check_ticket_availability,
    recommend_alternatives,
)
from agent_playground.tools.weather import get_weather


DEFAULT_TRAVEL_PROMPT = (
    "Hello, please help me check today's weather in Beijing, and then recommend "
    "a suitable tourist attraction based on the weather. I like historical and "
    "cultural attractions, and my budget is 100-300 RMB."
)


def create_travel_agent(
    tavily_api_key: str,
    ticket_availability: TicketAvailability | None = None,
) -> Agent:
    availability = ticket_availability or TicketAvailability(
        sold_out_attractions={"Forbidden City"}
    )
    return Agent(
        name="travel-assistant",
        system_prompt=TRAVEL_AGENT_SYSTEM_PROMPT,
        tools={
            "get_weather": get_weather,
            "get_attraction": create_attraction_tool(tavily_api_key),
            "check_ticket_availability": lambda attraction: check_ticket_availability(
                attraction,
                availability,
            ),
            "recommend_alternatives": lambda city, weather, unavailable_attraction, preferences="": recommend_alternatives(
                city=city,
                weather=weather,
                unavailable_attraction=unavailable_attraction,
                preferences=preferences,
                ticket_availability=availability,
            ),
        },
    )
