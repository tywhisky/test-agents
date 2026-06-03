from agent_playground.chapter_agents import (
    PlanAndSolveAgent,
    ReActAgent,
    ReflectionAgent,
    ToolExecutor,
)


class ScriptedLlm:
    """Return pre-written responses so examples run without API access."""

    def __init__(self, responses):
        self.responses = list(responses)

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        return self.responses.pop(0)


def run_react_example() -> None:
    llm = ScriptedLlm(
        [
            "Thought: I need fresh search information.\nAction: Search[Huawei latest phone]",
            "Thought: I have enough information.\nAction: Finish[Huawei Pura 80 series, with imaging as a main selling point.]",
        ]
    )
    tools = ToolExecutor()
    tools.register_tool(
        "Search",
        "Search the web for up-to-date facts.",
        lambda query: "Huawei Pura 80 series appears in recent Huawei phone results.",
    )

    answer = ReActAgent(llm, tools).run("What is Huawei's latest phone?")
    print("ReAct answer:", answer)


def run_plan_and_solve_example() -> None:
    llm = ScriptedLlm(
        [
            '```python\n["Compute Tuesday apples", "Compute Wednesday apples", "Compute total apples"]\n```',
            "30",
            "25",
            "70",
        ]
    )

    question = (
        "A fruit shop sold 15 apples on Monday. Tuesday sales were twice Monday. "
        "Wednesday sales were 5 fewer than Tuesday. How many apples total?"
    )
    answer = PlanAndSolveAgent(llm).run(question)
    print("Plan-and-Solve answer:", answer)


def run_reflection_example() -> None:
    llm = ScriptedLlm(
        [
            "def find_primes(n):\n    return [x for x in range(2, n + 1)]",
            "Use the Sieve of Eratosthenes to avoid testing every number naively.",
            "def find_primes(n):\n    \"\"\"Return primes from 1 to n.\"\"\"\n    if n < 2:\n        return []\n    is_prime = [True] * (n + 1)\n    is_prime[0] = is_prime[1] = False\n    for p in range(2, int(n ** 0.5) + 1):\n        if is_prime[p]:\n            for multiple in range(p * p, n + 1, p):\n                is_prime[multiple] = False\n    return [number for number in range(2, n + 1) if is_prime[number]]",
            "No improvement needed.",
        ]
    )

    code = ReflectionAgent(llm, max_iterations=2).run(
        "Write a Python function that finds all prime numbers from 1 to n."
    )
    print("Reflection code:\n", code)


if __name__ == "__main__":
    run_react_example()
    run_plan_and_solve_example()
    run_reflection_example()
