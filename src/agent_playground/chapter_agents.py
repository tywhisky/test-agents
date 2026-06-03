from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass
import re
from typing import Any


Tool = Callable[[str], str]


@dataclass(frozen=True)
class RegisteredTool:
    description: str
    func: Tool


class ToolExecutor:
    """Register and dispatch simple string-in, string-out agent tools."""

    def __init__(self) -> None:
        self.tools: dict[str, RegisteredTool] = {}

    def register_tool(self, name: str, description: str, func: Tool) -> None:
        self.tools[name] = RegisteredTool(description=description, func=func)

    def get_tool(self, name: str) -> Tool | None:
        tool = self.tools.get(name)
        return tool.func if tool else None

    def run(self, name: str, tool_input: str) -> str:
        tool = self.get_tool(name)
        if tool is None:
            return f"Error: Undefined tool '{name}'"
        try:
            return tool(tool_input)
        except Exception as error:
            return f"Error: Tool '{name}' failed - {error}"

    def available_tools_text(self) -> str:
        return "\n".join(
            f"- {name}: {tool.description}" for name, tool in self.tools.items()
        )

    registerTool = register_tool
    getTool = get_tool
    getAvailableTools = available_tools_text


REACT_PROMPT_TEMPLATE = """
You are an intelligent assistant that can call external tools.

Available tools:
{tools}

Always respond in this exact format:

Thought: your reasoning about the current situation.
Action: one of these forms:
- ToolName[tool input]
- Finish[final answer]

Question: {question}
History:
{history}
"""


class ReActAgent:
    """A teaching implementation of the Thought -> Action -> Observation loop."""

    def __init__(
        self,
        llm_client: Any,
        tool_executor: ToolExecutor,
        max_steps: int = 5,
    ) -> None:
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.history: list[str] = []

    def run(self, question: str) -> str | None:
        self.history = []

        for _ in range(self.max_steps):
            prompt = REACT_PROMPT_TEMPLATE.format(
                tools=self.tool_executor.available_tools_text(),
                question=question,
                history="\n".join(self.history) or "None",
            )
            response_text = _call_llm(self.llm_client, prompt)
            thought, action = self._parse_output(response_text)

            if thought:
                self.history.append(f"Thought: {thought}")
            if not action:
                self.history.append("Observation: Error: No Action found.")
                continue

            if action.startswith("Finish"):
                finish_match = re.match(r"Finish\[(.*)\]", action, re.DOTALL)
                return finish_match.group(1).strip() if finish_match else action

            tool_name, tool_input = self._parse_action(action)
            if not tool_name:
                self.history.append(
                    "Observation: Error: Invalid Action format. Use ToolName[input]."
                )
                continue

            observation = self.tool_executor.run(tool_name, tool_input)
            self.history.append(f"Action: {action}")
            self.history.append(f"Observation: {observation}")

        return None

    def _parse_output(self, text: str) -> tuple[str | None, str | None]:
        thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.DOTALL)
        action_match = re.search(r"Action:\s*(.*?)$", text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action

    def _parse_action(self, action_text: str) -> tuple[str | None, str | None]:
        match = re.match(r"(\w+)\[(.*)\]", action_text, re.DOTALL)
        if not match:
            return None, None
        return match.group(1), match.group(2).strip()


PLANNER_PROMPT_TEMPLATE = """
You are an expert AI planner. Break the user's problem into ordered,
independently executable steps.

Problem: {question}

Return only a fenced Python list:
```python
["step 1", "step 2", "step 3"]
```
"""


EXECUTOR_PROMPT_TEMPLATE = """
You are an expert AI executor. Follow the plan one step at a time.
Return only the result for the current step.

# Original question
{question}

# Full plan
{plan}

# History
{history}

# Current step
{current_step}
"""


class Planner:
    def __init__(self, llm_client: Any) -> None:
        self.llm_client = llm_client

    def plan(self, question: str) -> list[str]:
        response_text = _call_llm(
            self.llm_client,
            PLANNER_PROMPT_TEMPLATE.format(question=question),
        )
        plan_text = _extract_fenced_python(response_text) or response_text.strip()

        try:
            plan = ast.literal_eval(plan_text)
        except (ValueError, SyntaxError):
            return []
        if not isinstance(plan, list):
            return []
        return [str(step) for step in plan]


class Executor:
    def __init__(self, llm_client: Any) -> None:
        self.llm_client = llm_client

    def execute(self, question: str, plan: list[str]) -> str:
        history = ""
        last_result = ""

        for index, step in enumerate(plan, start=1):
            prompt = EXECUTOR_PROMPT_TEMPLATE.format(
                question=question,
                plan=plan,
                history=history or "None",
                current_step=step,
            )
            last_result = _call_llm(self.llm_client, prompt)
            history += f"Step {index}: {step}\nResult: {last_result}\n\n"

        return last_result


class PlanAndSolveAgent:
    def __init__(self, llm_client: Any) -> None:
        self.planner = Planner(llm_client)
        self.executor = Executor(llm_client)

    def run(self, question: str) -> str | None:
        plan = self.planner.plan(question)
        if not plan:
            return None
        return self.executor.execute(question, plan)


class ReflectionMemory:
    """Short-term memory for execution and reflection records."""

    def __init__(self) -> None:
        self.records: list[dict[str, str]] = []

    def add_record(self, record_type: str, content: str) -> None:
        self.records.append({"type": record_type, "content": content})

    def get_trajectory(self) -> str:
        parts = []
        for record in self.records:
            if record["type"] == "execution":
                parts.append(f"--- Previous attempt ---\n{record['content']}")
            elif record["type"] == "reflection":
                parts.append(f"--- Reviewer feedback ---\n{record['content']}")
        return "\n\n".join(parts)

    def get_last_execution(self) -> str | None:
        for record in reversed(self.records):
            if record["type"] == "execution":
                return record["content"]
        return None


INITIAL_PROMPT_TEMPLATE = """
You are a senior Python programmer. Write Python code for this task.
Include a full function signature, a docstring, and PEP 8 style.

Task: {task}

Return only code.
"""


REFLECT_PROMPT_TEMPLATE = """
You are a strict code reviewer and senior algorithm engineer.
Review the Python code for algorithmic efficiency.

# Original task
{task}

# Code to review
```python
{code}
```

If there is a meaningfully better algorithm, explain the problem and give a
specific improvement. If it is already good enough, answer "No improvement
needed."
"""


REFINE_PROMPT_TEMPLATE = """
You are a senior Python programmer improving code from review feedback.

# Original task
{task}

# Previous code
{last_code_attempt}

# Reviewer feedback
{feedback}

Return only the improved code.
"""


class ReflectionAgent:
    def __init__(self, llm_client: Any, max_iterations: int = 3) -> None:
        self.llm_client = llm_client
        self.memory = ReflectionMemory()
        self.max_iterations = max_iterations

    def run(self, task: str) -> str:
        initial_code = _call_llm(
            self.llm_client,
            INITIAL_PROMPT_TEMPLATE.format(task=task),
        )
        self.memory.add_record("execution", initial_code)

        for _ in range(self.max_iterations):
            last_code = self.memory.get_last_execution() or ""
            feedback = _call_llm(
                self.llm_client,
                REFLECT_PROMPT_TEMPLATE.format(task=task, code=last_code),
            )
            self.memory.add_record("reflection", feedback)

            if _means_no_improvement(feedback):
                break

            refined_code = _call_llm(
                self.llm_client,
                REFINE_PROMPT_TEMPLATE.format(
                    task=task,
                    last_code_attempt=last_code,
                    feedback=feedback,
                ),
            )
            self.memory.add_record("execution", refined_code)

        return self.memory.get_last_execution() or ""


def _call_llm(llm_client: Any, prompt: str) -> str:
    if hasattr(llm_client, "generate"):
        return llm_client.generate(prompt, system_prompt="")
    if hasattr(llm_client, "think"):
        return llm_client.think([{"role": "user", "content": prompt}]) or ""
    raise TypeError("LLM client must provide generate(...) or think(...).")


def _extract_fenced_python(text: str) -> str | None:
    match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else None


def _means_no_improvement(feedback: str) -> bool:
    normalized = feedback.lower()
    return "no improvement needed" in normalized or "\u65e0\u9700\u6539\u8fdb" in feedback
