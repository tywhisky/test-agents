from dataclasses import dataclass, field


@dataclass(frozen=True)
class TicketAvailability:
    sold_out_attractions: set[str] = field(default_factory=set)

    def is_sold_out(self, attraction: str) -> bool:
        normalized = attraction.strip().lower()
        return normalized in {item.lower() for item in self.sold_out_attractions}


def check_ticket_availability(
    attraction: str,
    ticket_availability: TicketAvailability | None = None,
) -> str:
    availability = ticket_availability or TicketAvailability()
    if availability.is_sold_out(attraction):
        return f"{attraction} tickets are sold out."
    return f"{attraction} tickets are available."


def recommend_alternatives(
    city: str,
    weather: str,
    unavailable_attraction: str,
    preferences: str = "",
    ticket_availability: TicketAvailability | None = None,
) -> str:
    availability = ticket_availability or TicketAvailability()
    alternatives = _default_alternatives(city, preferences)
    available_options = [
        option
        for option in alternatives
        if option.lower() != unavailable_attraction.lower()
        and not availability.is_sold_out(option)
    ]

    if not available_options:
        return (
            f"{unavailable_attraction} is unavailable, and no default alternatives are available. "
            "Ask customer service to check nearby attractions manually."
        )

    return (
        f"{unavailable_attraction} is unavailable for {city}. "
        f"Based on {weather} and preferences ({preferences or 'not specified'}), "
        "recommend these alternatives: "
        + ", ".join(available_options[:3])
        + "."
    )


def _default_alternatives(city: str, preferences: str) -> list[str]:
    if city.lower() == "beijing":
        if "historical" in preferences.lower() or "cultural" in preferences.lower():
            return ["Temple of Heaven", "Summer Palace", "National Museum of China"]
        return ["Summer Palace", "Beihai Park", "Temple of Heaven"]
    return ["Local museum", "Historic district", "City park"]

