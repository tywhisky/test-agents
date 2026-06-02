from collections.abc import Callable
from dataclasses import dataclass

from agent_playground.actions import (
    FinishAction,
    ToolAction,
    parse_action,
    truncate_to_first_action,
)
from agent_playground.memory import UserMemory, update_memory_from_user_text


Tool = Callable[..., str]


@dataclass(frozen=True)
class Agent:
    name: str
    system_prompt: str
    tools: dict[str, Tool]
    max_steps: int = 5


def run_agent(
    agent: Agent,
    llm,
    user_prompt: str,
    memory: UserMemory | None = None,
) -> str | None:
    prompt_history = [f"User request: {user_prompt}"]
    tools = dict(agent.tools)
    if memory is not None:
        update_memory_from_user_text(memory, user_prompt)
        tools.update(_create_memory_tools(memory))
        prompt_history.append(memory.to_prompt_context())

    print(f"User input: {user_prompt}\n" + "=" * 40)
    if memory is not None:
        print(memory.to_prompt_context() + "\n" + "=" * 40)

    for step in range(agent.max_steps):
        print(f"--- Loop {step + 1} ---\n")

        full_prompt = "\n".join(prompt_history)
        llm_output = llm.generate(full_prompt, system_prompt=agent.system_prompt)
        print(f"Raw Model output:\n{llm_output}\n")

        truncated_output = truncate_to_first_action(llm_output)
        if truncated_output != llm_output.strip():
            print("-> Truncated extra Thought-Action pairs")
        prompt_history.append(truncated_output)

        try:
            action = parse_action(truncated_output)
        except Exception as error:
            observation = f"Error parsing/executing action: {error}"
            _record_observation(prompt_history, observation)
            continue

        if action is None:
            observation = "Error: No action found. Please explicitly use Action: tool_name(arguments) or Action: Finish[answer]"
            _record_observation(prompt_history, observation)
            continue

        if isinstance(action, FinishAction):
            print(f"Task completed! {action.answer}")
            return action.answer

        observation = _execute_tool(tools, action)
        _record_observation(prompt_history, observation)

    return None


def _execute_tool(tools: dict[str, Tool], action: ToolAction) -> str:
    tool = tools.get(action.name)
    if tool is None:
        return f"Error: Undefined tool '{action.name}'"

    try:
        return tool(**action.kwargs)
    except Exception as error:
        return f"Error parsing/executing action: {error}"


def _record_observation(prompt_history: list[str], observation: str) -> None:
    observation_text = f"Observation: {observation}"
    print(f"{observation_text}\n" + "=" * 40)
    prompt_history.append(observation_text)


def _create_memory_tools(memory: UserMemory) -> dict[str, Tool]:
    def remember_preference(preference: str) -> str:
        memory.remember_preference(preference)
        return f"Remembered preference: {preference}"

    def remember_budget(budget_range: str) -> str:
        memory.remember_budget(budget_range)
        return f"Remembered budget range: {budget_range}"

    def record_rejection(recommendation: str) -> str:
        memory.record_rejection(recommendation)
        return memory.reflection_hint() or f"Recorded rejection: {recommendation}"

    def record_acceptance(recommendation: str) -> str:
        memory.record_acceptance(recommendation)
        return f"Recorded acceptance: {recommendation}"

    return {
        "remember_preference": remember_preference,
        "remember_budget": remember_budget,
        "record_rejection": record_rejection,
        "record_acceptance": record_acceptance,
    }
