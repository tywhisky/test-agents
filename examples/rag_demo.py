from agent_playground.tools.rag import RAGTool


class ScriptedLlm:
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        if "Python" in prompt:
            return "Python is a high-level language first released in 1991."
        return "The answer is not available in the provided context."


def main() -> None:
    rag_tool = RAGTool(rag_namespace="demo", llm=ScriptedLlm())

    rag_tool.execute(
        "add_text",
        text=(
            "# Python\n"
            "Python is a high-level programming language. "
            "It was first released in 1991 by Guido van Rossum."
        ),
        document_id="python_intro",
    )
    rag_tool.execute(
        "add_text",
        text=(
            "# Machine Learning\n"
            "Machine learning is a branch of artificial intelligence. "
            "It learns patterns from data."
        ),
        document_id="ml_basics",
    )
    rag_tool.execute(
        "add_text",
        text=(
            "# RAG\n"
            "Retrieval-augmented generation retrieves relevant knowledge before "
            "asking a language model to answer."
        ),
        document_id="rag_concept",
    )

    print("Search results:")
    print(rag_tool.execute("search", query="Python programming history", limit=3))

    print("\nQuestion answering:")
    answer = rag_tool.execute("ask", question="When was Python first released?")
    print(answer["answer"])
    print("Sources:", answer["sources"])

    print("\nStats:")
    print(rag_tool.execute("stats"))


if __name__ == "__main__":
    main()
