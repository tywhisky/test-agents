# test-agent

This is a beginner ReAct-style Python agent.

The basic loop is:

1. The user asks a question.
2. The language model returns a `Thought` and an `Action`.
3. Python parses the action.
4. Python calls the matching tool function.
5. The tool result is added back as an `Observation`.
6. The loop repeats until the model returns `Finish[...]`.

In this project, the tools are:

- `get_weather(city="...")`
- `get_attraction(city="...", weather="...")`

## Setup

Create a `.env` file with:

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

If everything is configured, the program will:

1. Ask the model what to do.
2. Call the weather tool for Beijing.
3. Call the attraction search tool.
4. Print the final travel recommendation.
