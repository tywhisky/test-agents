from tavily import TavilyClient


def create_attraction_tool(api_key: str):
    def get_attraction(city: str, weather: str) -> str:
        """Search for travel attraction recommendations with Tavily."""
        tavily = TavilyClient(api_key=api_key)
        query = f"'{city}' most worthwhile tourist attractions and reasons in '{weather}' weather"

        try:
            response = tavily.search(query=query, search_depth="basic", include_answer=True)

            if response.get("answer"):
                return response["answer"]

            formatted_results = [
                f"- {result['title']}: {result['content']}"
                for result in response.get("results", [])
            ]
            if not formatted_results:
                return "Sorry, no relevant tourist attraction recommendations found."

            return "Based on search, found the following information for you:\n" + "\n".join(formatted_results)
        except Exception as error:
            return f"Error: Problem occurred when executing Tavily search - {error}"

    return get_attraction

