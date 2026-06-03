from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_playground.rag import (
    InMemoryVectorStore,
    chunk_markdown,
    convert_to_markdown,
)


class RAGTool:
    """Unified local RAG tool for adding, searching, and answering from knowledge."""

    def __init__(
        self,
        knowledge_base_path: str = "./knowledge_base",
        collection_name: str = "rag_knowledge_base",
        rag_namespace: str = "default",
        store: InMemoryVectorStore | None = None,
        llm: Any | None = None,
    ) -> None:
        self.knowledge_base_path = Path(knowledge_base_path)
        self.collection_name = collection_name
        self.rag_namespace = rag_namespace
        self.store = store or InMemoryVectorStore()
        self.llm = llm

    def execute(self, action: str, **kwargs):
        handlers = {
            "add_text": self.add_text,
            "add_document": self.add_document,
            "search": self.search,
            "ask": self.ask,
            "stats": self.stats,
        }
        handler = handlers.get(action)
        if handler is None:
            return {"success": False, "error": f"Unknown RAG action: {action}"}
        return handler(**kwargs)

    def add_text(
        self,
        text: str,
        document_id: str,
        chunk_size: int = 300,
        chunk_overlap: int = 50,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        chunks = chunk_markdown(
            text,
            chunk_tokens=chunk_size,
            overlap_tokens=chunk_overlap,
        )
        for chunk in chunks:
            chunk["document_id"] = document_id
            chunk["metadata"] = metadata or {}
        stored_chunks = self.store.add_chunks(
            chunks,
            namespace=self.rag_namespace,
            document_id=document_id,
        )
        return {
            "success": True,
            "document_id": document_id,
            "chunks": len(stored_chunks),
        }

    def add_document(
        self,
        file_path: str,
        chunk_size: int = 300,
        chunk_overlap: int = 50,
    ) -> dict[str, Any]:
        markdown = convert_to_markdown(file_path)
        if not markdown.strip():
            return {"success": False, "error": f"Could not read document: {file_path}"}

        document_id = Path(file_path).stem
        result = self.add_text(
            text=markdown,
            document_id=document_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            metadata={"source_path": file_path},
        )
        result["documents"] = self.stats()["documents"]
        return result

    def search(
        self,
        query: str,
        limit: int = 5,
        min_score: float = 0.01,
        enable_mqe: bool = False,
        enable_hyde: bool = False,
        **kwargs,
    ) -> str:
        hits = self.search_hits(
            query=query,
            limit=limit,
            min_score=min_score,
            enable_mqe=enable_mqe,
            enable_hyde=enable_hyde,
        )
        if not hits:
            return "No relevant knowledge found."
        return "\n\n".join(_format_hit(index, hit) for index, hit in enumerate(hits, 1))

    def search_hits(
        self,
        query: str,
        limit: int = 5,
        min_score: float = 0.01,
        enable_mqe: bool = False,
        enable_hyde: bool = False,
    ) -> list[dict[str, Any]]:
        expansions = []
        if enable_mqe:
            expansions.extend(_simple_query_expansions(query))
        if enable_hyde:
            expansions.append(_simple_hypothetical_document(query))
        return self.store.search(
            query=query,
            limit=limit,
            namespace=self.rag_namespace,
            min_score=min_score,
            expansions=expansions,
        )

    def ask(
        self,
        question: str,
        limit: int = 5,
        min_score: float = 0.01,
        enable_advanced_search: bool = False,
        enable_mqe: bool = False,
        enable_hyde: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        hits = self.search_hits(
            query=question,
            limit=limit,
            min_score=min_score,
            enable_mqe=enable_advanced_search or enable_mqe,
            enable_hyde=enable_advanced_search or enable_hyde,
        )
        if not hits:
            return {"answer": "No relevant knowledge found.", "sources": []}

        prompt = _build_answer_prompt(question, hits)
        answer = _call_llm(self.llm, prompt) if self.llm else _fallback_answer(hits)
        return {
            "answer": answer,
            "sources": [
                {
                    "document_id": hit["document_id"],
                    "heading_path": hit.get("heading_path"),
                    "score": hit["score"],
                }
                for hit in hits
            ],
        }

    def stats(self) -> dict[str, int | str]:
        stats = self.store.stats(namespace=self.rag_namespace)
        return {
            "collection": self.collection_name,
            "namespace": self.rag_namespace,
            **stats,
        }


def _format_hit(index: int, hit: dict[str, Any]) -> str:
    heading = f" ({hit['heading_path']})" if hit.get("heading_path") else ""
    return (
        f"[{index}] {hit['document_id']}{heading} "
        f"score={hit['score']:.3f}\n{hit['content']}"
    )


def _build_answer_prompt(question: str, hits: list[dict[str, Any]]) -> str:
    context = "\n\n".join(
        f"Source {index}: {hit['content']}" for index, hit in enumerate(hits, 1)
    )
    return (
        "Answer the question using only the retrieved context. "
        "If the answer is not present, say that the knowledge base does not contain it.\n\n"
        f"Question: {question}\n\nContext:\n{context}"
    )


def _call_llm(llm: Any, prompt: str) -> str:
    if hasattr(llm, "generate"):
        return llm.generate(prompt, system_prompt="")
    if hasattr(llm, "think"):
        return llm.think([{"role": "user", "content": prompt}]) or ""
    raise TypeError("LLM client must provide generate(...) or think(...).")


def _fallback_answer(hits: list[dict[str, Any]]) -> str:
    source_lines = [hit["content"] for hit in hits]
    return "\n\n".join(source_lines)


def _simple_query_expansions(query: str) -> list[str]:
    words = [word for word in query.split() if len(word) > 3]
    if not words:
        return []
    return [" ".join(words), " ".join(reversed(words))]


def _simple_hypothetical_document(query: str) -> str:
    return f"This passage explains {query} with definitions, history, and key facts."
