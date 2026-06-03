from __future__ import annotations

from typing import Any

from tavily import TavilyClient


def create_search_tool(api_key: str):
    """Create a generic web search tool backed by Tavily."""

    def search(query: str) -> str:
        try:
            client = TavilyClient(api_key=api_key)
            response = client.search(
                query=query,
                search_depth="basic",
                include_answer=True,
            )
            return format_search_response(response, query=query)
        except Exception as error:
            return f"Error: Problem occurred when executing Tavily search - {error}"

    return search


def format_search_response(
    response: dict[str, Any],
    query: str | None = None,
    max_results: int = 3,
) -> str:
    """Prefer direct answers, then fall back to formatted search snippets."""
    answer = response.get("answer")
    if answer:
        return str(answer)

    results = response.get("results", [])
    snippets = [
        f"[{index}] {result.get('title', '')}\n{result.get('content', '')}".strip()
        for index, result in enumerate(results[:max_results], start=1)
    ]
    snippets = [snippet for snippet in snippets if snippet]
    if snippets:
        return "\n\n".join(snippets)

    if query:
        return f"Sorry, no information found for '{query}'."
    return "Sorry, no search results found."
