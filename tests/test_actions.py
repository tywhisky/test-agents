import unittest

from agent_playground.actions import FinishAction, ToolAction, parse_action


class ActionParserTests(unittest.TestCase):
    def test_parse_tool_action_with_keyword_argument(self):
        action = parse_action('Action: get_weather(city="Beijing")')

        self.assertEqual(
            action,
            ToolAction(name="get_weather", kwargs={"city": "Beijing"}),
        )

    def test_parse_finish_action_with_square_brackets(self):
        action = parse_action("Action: Finish[It is sunny today.]")

        self.assertEqual(action, FinishAction(answer="It is sunny today."))

    def test_parse_returns_none_when_action_is_missing(self):
        self.assertIsNone(parse_action("Thought: I need weather first."))


if __name__ == "__main__":
    unittest.main()
