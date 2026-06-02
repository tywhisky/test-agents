import unittest

from agent_playground.memory import UserMemory, update_memory_from_user_text


class MemoryTests(unittest.TestCase):
    def test_memory_formats_preferences_for_prompt(self):
        memory = UserMemory(
            preferred_attraction_types=["historical", "cultural"],
            budget_range="100-300 RMB",
        )

        self.assertIn("historical, cultural", memory.to_prompt_context())
        self.assertIn("100-300 RMB", memory.to_prompt_context())

    def test_record_rejection_triggers_reflection_after_three_rejections(self):
        memory = UserMemory()

        memory.record_rejection("Forbidden City")
        memory.record_rejection("Summer Palace")
        memory.record_rejection("Temple of Heaven")

        self.assertTrue(memory.needs_strategy_reflection)
        self.assertIn("3 consecutive recommendations", memory.reflection_hint())

    def test_acceptance_resets_consecutive_rejections(self):
        memory = UserMemory()
        memory.record_rejection("Forbidden City")

        memory.record_acceptance("Summer Palace")

        self.assertEqual(memory.consecutive_rejections, 0)
        self.assertFalse(memory.needs_strategy_reflection)

    def test_update_memory_from_user_text_extracts_budget_range(self):
        memory = UserMemory()

        update_memory_from_user_text(memory, "I like museums and my budget is 100-300 RMB.")

        self.assertEqual(memory.budget_range, "100-300 RMB")
        self.assertIn("museum", memory.preferred_attraction_types)


if __name__ == "__main__":
    unittest.main()
