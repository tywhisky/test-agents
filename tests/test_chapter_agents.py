import unittest

from agent_playground.chapter_agents import (
    PlanAndSolveAgent,
    Planner,
    ReActAgent,
    ReflectionAgent,
    ReflectionMemory,
    ToolExecutor,
)
from agent_playground.tools.search import format_search_response


class FakeLlm:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def generate(self, prompt, system_prompt=""):
        self.prompts.append((prompt, system_prompt))
        return self.responses.pop(0)


class ChapterAgentTests(unittest.TestCase):
    def test_tool_executor_registers_and_runs_named_tool(self):
        executor = ToolExecutor()
        executor.register_tool("Search", "Search the web.", lambda query: f"found {query}")

        self.assertIn("- Search: Search the web.", executor.available_tools_text())
        self.assertEqual(executor.run("Search", "GPU"), "found GPU")

    def test_react_agent_executes_tool_observation_then_finish(self):
        llm = FakeLlm(
            [
                "Thought: Need fresh information.\nAction: Search[Huawei latest phone]",
                "Thought: I can answer now.\nAction: Finish[Huawei Pura 80 is the answer.]",
            ]
        )
        executor = ToolExecutor()
        executor.register_tool("Search", "Search the web.", lambda query: f"result for {query}")

        answer = ReActAgent(llm, executor, max_steps=3).run("Huawei latest phone?")

        self.assertEqual(answer, "Huawei Pura 80 is the answer.")
        self.assertIn("Observation: result for Huawei latest phone", llm.prompts[1][0])

    def test_planner_parses_python_list_from_fenced_response(self):
        llm = FakeLlm(['```python\n["step one", "step two"]\n```'])

        plan = Planner(llm).plan("solve it")

        self.assertEqual(plan, ["step one", "step two"])

    def test_plan_and_solve_passes_history_between_steps(self):
        llm = FakeLlm(
            [
                '```python\n["compute Tuesday", "compute total"]\n```',
                "30",
                "70",
            ]
        )

        answer = PlanAndSolveAgent(llm).run("apple problem")

        self.assertEqual(answer, "70")
        self.assertIn("Result: 30", llm.prompts[2][0])

    def test_reflection_agent_refines_until_feedback_says_no_improvement(self):
        llm = FakeLlm(
            [
                "def find_primes(n):\n    return []",
                "Use the Sieve of Eratosthenes.",
                "def find_primes(n):\n    return [2]",
                "No improvement needed.",
            ]
        )

        code = ReflectionAgent(llm, max_iterations=2).run("write prime finder")

        self.assertEqual(code, "def find_primes(n):\n    return [2]")

    def test_reflection_memory_formats_trajectory(self):
        memory = ReflectionMemory()
        memory.add_record("execution", "code v1")
        memory.add_record("reflection", "feedback")

        self.assertEqual(memory.get_last_execution(), "code v1")
        self.assertIn("Previous attempt", memory.get_trajectory())
        self.assertIn("Reviewer feedback", memory.get_trajectory())

    def test_format_search_response_prefers_answer_then_results(self):
        self.assertEqual(format_search_response({"answer": "direct answer"}), "direct answer")
        self.assertEqual(
            format_search_response(
                {
                    "results": [
                        {"title": "A", "content": "first"},
                        {"title": "B", "content": "second"},
                    ]
                }
            ),
            "[1] A\nfirst\n\n[2] B\nsecond",
        )


if __name__ == "__main__":
    unittest.main()
