import tempfile
import unittest
from pathlib import Path

from agent_playground.rag import (
    InMemoryVectorStore,
    chunk_markdown,
    split_paragraphs_with_headings,
)
from agent_playground.tools.rag import RAGTool


class FakeLlm:
    def __init__(self):
        self.prompts = []

    def generate(self, prompt, system_prompt=""):
        self.prompts.append((prompt, system_prompt))
        return "Python was first released in 1991."


class RagTests(unittest.TestCase):
    def test_split_paragraphs_tracks_markdown_heading_path(self):
        paragraphs = split_paragraphs_with_headings(
            "# Python\nPython is a programming language.\n\n## History\nReleased in 1991."
        )

        self.assertEqual(paragraphs[0]["heading_path"], "Python")
        self.assertEqual(paragraphs[1]["heading_path"], "Python > History")

    def test_chunk_markdown_keeps_heading_metadata(self):
        chunks = chunk_markdown(
            "# Python\nPython is readable and concise.\n\n## History\nReleased in 1991.",
            chunk_tokens=5,
            overlap_tokens=0,
        )

        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["heading_path"], "Python")
        self.assertEqual(chunks[-1]["heading_path"], "Python > History")

    def test_chunk_markdown_handles_overlap_larger_than_chunk(self):
        chunks = chunk_markdown(
            "First paragraph has several words.\n\nSecond paragraph also has words.",
            chunk_tokens=3,
            overlap_tokens=10,
        )

        self.assertEqual(len(chunks), 2)

    def test_vector_store_search_ranks_related_chunks(self):
        store = InMemoryVectorStore()
        store.add_chunks(
            [
                {"content": "Python is a programming language.", "document_id": "python"},
                {"content": "Tickets are sold out today.", "document_id": "travel"},
            ],
            namespace="test",
        )

        results = store.search("programming language", limit=1, namespace="test")

        self.assertEqual(results[0]["document_id"], "python")
        self.assertGreater(results[0]["score"], 0)

    def test_rag_tool_add_text_search_and_stats(self):
        tool = RAGTool(rag_namespace="test")

        add_result = tool.execute(
            "add_text",
            text="Python is a high-level programming language.",
            document_id="python_intro",
        )
        search_result = tool.execute("search", query="high-level language", limit=2)
        stats = tool.execute("stats")

        self.assertTrue(add_result["success"])
        self.assertIn("Python", search_result)
        self.assertEqual(stats["documents"], 1)
        self.assertEqual(stats["chunks"], 1)

    def test_rag_tool_add_document_uses_text_reader_fallback(self):
        tool = RAGTool(rag_namespace="docs")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "guide.md"
            path.write_text("# Guide\nUse retrieval before generation.", encoding="utf-8")

            result = tool.execute("add_document", file_path=str(path))

        self.assertTrue(result["success"])
        self.assertEqual(result["documents"], 1)
        self.assertGreaterEqual(result["chunks"], 1)

    def test_rag_tool_ask_uses_retrieved_context_with_llm(self):
        llm = FakeLlm()
        tool = RAGTool(rag_namespace="qa", llm=llm)
        tool.execute(
            "add_text",
            text="Python was first released in 1991 by Guido van Rossum.",
            document_id="python_history",
        )

        answer = tool.execute("ask", question="When was Python first released?")

        self.assertEqual(answer["answer"], "Python was first released in 1991.")
        self.assertIn("Python was first released", llm.prompts[0][0])
        self.assertEqual(answer["sources"][0]["document_id"], "python_history")


if __name__ == "__main__":
    unittest.main()
