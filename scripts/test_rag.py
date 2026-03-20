#!/usr/bin/env python
"""Test RAG integration — verify retrieval quality and prompt output.

Usage:
    python scripts/test_rag.py
"""

import io
import os
import sys

# Fix Windows console encoding for Chinese output
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def test_rag_retrieval():
    """Test basic RAG retrieval with several queries."""
    from agent.rag import AbaqusRAG

    rag = AbaqusRAG()
    print(f"=== RAG Index Status ===")
    print(f"Documents in index: {rag.count()}")
    print()

    if not rag.is_indexed():
        print("ERROR: Index is empty. Run 'python scripts/build_rag_index.py' first.")
        return False

    # Test queries covering different categories
    test_queries = [
        "接触分析，两个零件之间有摩擦",
        "热传导分析，稳态温度场",
        "屈曲分析，求临界载荷",
        "螺栓预紧力",
        "弹簧单元连接",
        "cohesive zone delamination",
    ]

    for query in test_queries:
        print(f"--- Query: {query} ---")
        results = rag.retrieve(query, top_k=3)
        for i, doc in enumerate(results):
            dist = f"{doc['distance']:.4f}" if doc['distance'] is not None else "N/A"
            print(f"  [{i+1}] {doc['title']} (distance={dist}, source={doc['source']})")
            # Show first 80 chars of content
            preview = doc['content'][:80].replace('\n', ' ')
            print(f"      {preview}...")
        print()

    return True


def test_prompt_with_rag():
    """Show how RAG context gets injected into the system prompt."""
    from agent.rag import AbaqusRAG

    rag = AbaqusRAG()
    if not rag.is_indexed():
        print("Skipping prompt test — index is empty.")
        return

    # Simulate what build_system_prompt does with RAG
    query = "两个零件之间的接触分析，有摩擦系数0.3"
    results = rag.retrieve(query, top_k=3)

    print("=== RAG Context (injected into system prompt) ===")
    print()
    rag_context = "# Abaqus Native API Reference (from documentation)\n"
    rag_context += (
        "When the simplified API above does not cover the needed functionality, "
        "you may use these native Abaqus Python commands directly.\n"
        "Wrap each native command with m._buf.emit(\"...\") to add it to the code buffer.\n\n"
    )
    for doc in results:
        rag_context += f"## {doc['title']}\n{doc['content']}\n\n"

    print(rag_context)
    print(f"--- RAG context length: {len(rag_context)} chars ---")


def test_agent_init():
    """Test that AbaqusAgent initializes RAG correctly."""
    print("=== Agent RAG Initialization ===")
    try:
        from agent.agent import AbaqusAgent
        # Don't need a real API key for init test
        agent = AbaqusAgent(api_key="test-key")
        if agent.rag is not None:
            print(f"OK: RAG enabled with {agent.rag.count()} documents")
        else:
            print("WARN: RAG not enabled (index may be empty or chromadb issue)")
    except Exception as e:
        print(f"Agent init error: {e}")
    print()


if __name__ == "__main__":
    ok = test_rag_retrieval()
    if ok:
        test_prompt_with_rag()
    print()
    test_agent_init()
