import unittest

from agent_playground.memory import UserMemory
from agent_playground.runner import Agent, run_agent


class FakeLlm:
    def __init__(self, responses):
        self.responses = list(responses)

    def generate(self, prompt, system_prompt):
        return self.responses.pop(0)


class RunnerTests(unittest.TestCase):
    def test_runner_exposes_memory_tools_when_memory_is_provided(self):
        llm = FakeLlm(
            [
                'Thought: I should remember this preference.\nAction: remember_preference(preference="historical")',
                "Thought: I am done.\nAction: Finish[Preference saved.]",
            ]
        )
        memory = UserMemory()
        agent = Agent(name="test", system_prompt="test", tools={}, max_steps=2)

        answer = run_agent(agent, llm, "I like historical attractions.", memory=memory)

        self.assertEqual(answer, "Preference saved.")
        self.assertIn("historical", memory.preferred_attraction_types)


if __name__ == "__main__":
    unittest.main()
