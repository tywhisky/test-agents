from dataclasses import dataclass
import re


@dataclass(frozen=True)
class ToolAction:
    name: str
    kwargs: dict[str, str]


@dataclass(frozen=True)
class FinishAction:
    answer: str


def truncate_to_first_action(model_output: str) -> str:
    match = re.search(
        r"(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)",
        model_output,
        re.DOTALL,
    )
    if not match:
        return model_output
    return match.group(1).strip()


def parse_action(model_output: str) -> ToolAction | FinishAction | None:
    action_match = re.search(r"Action:\s*(.*)", model_output, re.IGNORECASE)
    if not action_match:
        return None

    action_text = action_match.group(1).strip()
    if action_text.startswith("Finish"):
        finish_match = re.match(r"Finish(?:\[(.*)\]|\((.*)\))", action_text)
        if not finish_match:
            return FinishAction(answer=action_text)
        return FinishAction(answer=finish_match.group(1) or finish_match.group(2) or "")

    tool_name_match = re.search(r"(\w+)\(", action_text)
    args_match = re.search(r"\((.*)\)", action_text)
    if not tool_name_match or not args_match:
        raise ValueError('Action format invalid. Use tool_name(key="value")')

    kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_match.group(1)))
    return ToolAction(name=tool_name_match.group(1), kwargs=kwargs)

