import unittest

from agent_playground.tools.travel_recommendations import (
    TicketAvailability,
    recommend_alternatives,
)


class TravelRecommendationToolTests(unittest.TestCase):
    def test_recommend_alternatives_avoids_sold_out_attraction(self):
        sold_out = TicketAvailability(sold_out_attractions={"Forbidden City"})

        recommendation = recommend_alternatives(
            city="Beijing",
            weather="Sunny, 33 degrees Celsius",
            unavailable_attraction="Forbidden City",
            preferences="historical and cultural attractions, budget 100-300 RMB",
            ticket_availability=sold_out,
        )

        self.assertIn("Forbidden City is unavailable", recommendation)
        self.assertIn("Temple of Heaven", recommendation)
        self.assertIn("Summer Palace", recommendation)


if __name__ == "__main__":
    unittest.main()
