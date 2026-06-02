# Agent Playground

This is a beginner-friendly playground for learning how Python agents work.

The current demo is a ReAct-style travel assistant. ReAct means the model loops through:

1. The user asks a question.
2. The language model returns a `Thought` and an `Action`.
3. Python parses the action.
4. Python calls the matching tool function.
5. The tool result is added back as an `Observation`.
6. The loop repeats until the model returns `Finish[...]`.

## Project Structure

```text
test-agent/
├── main.py                         # Simple app entry point
├── examples/
│   └── travel_demo.py              # Example script for the travel agent
├── src/
│   └── agent_playground/
│       ├── actions.py              # Parse Action lines from model output
│       ├── config.py               # Read and validate .env values
│       ├── llm.py                  # OpenAI-compatible LLM client wrapper
│       ├── runner.py               # Generic ReAct agent loop
│       ├── agents/
│       │   └── travel_assistant.py # Travel agent definition
│       ├── prompts/
│       │   └── travel.py           # Travel agent system prompt
│       └── tools/
│           ├── weather.py          # Weather tool
│           └── attractions.py      # Tavily attraction search tool
└── tests/
    ├── test_actions.py             # Parser tests
    └── test_config.py              # Config tests
```

The travel assistant currently has two tools:

- `get_weather(city="...")`
- `get_attraction(city="...", weather="...")`

## Setup

Copy `.env.example` to `.env`, then fill in your values:

```bash
TAVILY_API_KEY=your_tavily_key
API_KEY=your_llm_api_key
BASE_URL=your_openai_compatible_base_url
MODEL_NAME=your_model_name
```

`MODEL_ID` also works if your tutorial or provider uses that name instead of `MODEL_NAME`.

## Run

```bash
uv run python main.py
```

You can also run the example directly:

```bash
uv run python examples/travel_demo.py
```

If everything is configured, the program will:

1. Ask the model what to do.
2. Call the weather tool for Beijing.
3. Call the attraction search tool.
4. Print the final travel recommendation.

## Test

```bash
uv run python -m unittest discover -s tests -v
```

## How To Add Your Next Agent

1. Add tool functions in `src/agent_playground/tools/`.
2. Add a system prompt in `src/agent_playground/prompts/`.
3. Add an agent factory in `src/agent_playground/agents/`.
4. Run it through `run_agent(...)` from `src/agent_playground/runner.py`.

This keeps each concept small:

- `tools` do real-world work.
- `prompts` teach the model how to behave.
- `agents` connect prompts and tools.
- `runner` owns the loop.
- `llm` owns model API calls.
- `config` owns environment setup.
