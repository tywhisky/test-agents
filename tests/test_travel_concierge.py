import importlib
import unittest
from unittest.mock import patch

from agent_framework.exceptions import ChatClientException
from agent_framework.openai import OpenAIChatCompletionClient
from pydantic import ValidationError


class TravelConciergeScriptTests(unittest.TestCase):
    def test_module_imports_without_starting_agent_run(self) -> None:
        module = importlib.import_module("agents.travel_concierge")

        self.assertTrue(callable(module.main))

    def test_create_agent_uses_chat_completions_client_for_deepseek(self) -> None:
        module = importlib.import_module("agents.travel_concierge")

        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}):
            agent = module.create_agent()

        self.assertIsInstance(agent.client, OpenAIChatCompletionClient)

    def test_destination_recommendation_schema_validates_structured_output(self) -> None:
        module = importlib.import_module("agents.travel_concierge")

        result = module.TravelRecommendations(
            recommendations=[
                module.DestinationRecommendation(
                    destination="Tokyo",
                    available=True,
                    best_season="Mar-Apr",
                    highlights=["Culture", "food", "technology"],
                    estimated_budget_usd=2500,
                )
            ],
            personalized_note="Tokyo fits the traveler's culture and food interests.",
        )

        self.assertEqual(result.recommendations[0].destination, "Tokyo")
        with self.assertRaises(ValidationError):
            module.DestinationRecommendation(
                destination="Tokyo",
                available=True,
                best_season="Mar-Apr",
                highlights=["Culture"],
                estimated_budget_usd="expensive",
            )

    def test_destination_details_tool_returns_grounding_data(self) -> None:
        module = importlib.import_module("agents.travel_concierge")

        self.assertIn("Best: Mar-Apr", module.get_destination_details(destination="Tokyo"))
        self.assertIn("Not available", module.get_destination_details(destination="Cape Town"))
        self.assertEqual(
            module.get_destination_details(destination="Lisbon"),
            "Lisbon: No information available.",
        )

    def test_create_structured_agent_registers_destination_tool(self) -> None:
        module = importlib.import_module("agents.travel_concierge")

        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}):
            agent = module.create_structured_agent()

        self.assertEqual(agent.name, "StructuredTravelExpert")
        self.assertIn(module.get_destination_details, agent.default_options["tools"])

    def test_parse_travel_recommendations_validates_json_text(self) -> None:
        module = importlib.import_module("agents.travel_concierge")

        parsed = module.parse_travel_recommendations(
            """
            {
              "recommendations": [
                {
                  "destination": "Barcelona",
                  "available": true,
                  "best_season": "May-Jun",
                  "highlights": ["Beach", "architecture", "nightlife"],
                  "estimated_budget_usd": 2000
                }
              ],
              "personalized_note": "Barcelona fits the culture and food request."
            }
            """
        )

        self.assertEqual(parsed.recommendations[0].destination, "Barcelona")

    def test_parse_travel_recommendations_extracts_json_from_model_text(self) -> None:
        module = importlib.import_module("agents.travel_concierge")

        parsed = module.parse_travel_recommendations(
            """
            Let me look up some culture-heavy options.
            ```json
            {
              "recommendations": [
                {
                  "destination": "Tokyo",
                  "available": true,
                  "best_season": "Mar-Apr",
                  "highlights": ["Culture", "food", "technology"],
                  "estimated_budget_usd": 2500
                }
              ],
              "personalized_note": "Tokyo is a strong match."
            }
            ```
            """
        )

        self.assertEqual(parsed.recommendations[0].destination, "Tokyo")

    def test_structured_runner_falls_back_when_provider_rejects_response_format(self) -> None:
        module = importlib.import_module("agents.travel_concierge")

        class FakeResponse:
            value = None
            text = """
            {
              "recommendations": [
                {
                  "destination": "Tokyo",
                  "available": true,
                  "best_season": "Mar-Apr",
                  "highlights": ["Culture", "food", "technology"],
                  "estimated_budget_usd": 2500
                }
              ],
              "personalized_note": "Tokyo matches the budget and culture focus."
            }
            """

        class FakeAgent:
            def __init__(self) -> None:
                self.calls = []

            async def run(self, prompt, *, options=None):
                self.calls.append((prompt, options))
                if len(self.calls) == 1:
                    raise ChatClientException("This response_format type is unavailable now")
                return FakeResponse()

        async def run_test():
            fake_agent = FakeAgent()
            result = await module.run_structured_recommendations(fake_agent)
            self.assertEqual(result.recommendations[0].destination, "Tokyo")
            self.assertEqual(fake_agent.calls[0][1], {"response_format": module.TravelRecommendations})
            self.assertIn("valid JSON", fake_agent.calls[1][0])
            self.assertIn("Barcelona: Available", fake_agent.calls[1][0])
            self.assertIn("Cape Town: Not available", fake_agent.calls[1][0])

        import asyncio

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
