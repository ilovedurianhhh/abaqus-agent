#!/usr/bin/env python
"""Build RAG vector index from Abaqus documentation.

Usage:
    python scripts/build_rag_index.py [--docs-dir PATH] [--db-dir PATH]

Sources:
    1. Abaqus HTML documentation (if available on this machine)
    2. Hand-curated API docs from rag_docs/abaqus_api_manual.json
"""

import argparse
import json
import os
import sys

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(description="Build Abaqus RAG index")
    parser.add_argument(
        "--docs-dir",
        default=r"D:/SIMULIA/EstProducts/2022/CAADoc/win_b64.doc/English",
        help="Path to Abaqus HTML documentation directory",
    )
    parser.add_argument(
        "--db-dir",
        default="rag_db",
        help="ChromaDB persistence directory (relative to project root)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-index even if the database already exists",
    )
    args = parser.parse_args()

    from agent.rag import AbaqusRAG

    db_path = args.db_dir
    if not os.path.isabs(db_path):
        db_path = os.path.join(_PROJECT_ROOT, db_path)

    rag = AbaqusRAG(db_dir=db_path)

    if rag.is_indexed() and not args.force:
        print(f"Index already exists with {rag.count()} documents.")
        print("Use --force to re-index.")
        return

    total = 0

    # 1. Ingest HTML docs (if directory exists)
    if os.path.isdir(args.docs_dir):
        print(f"Ingesting HTML docs from: {args.docs_dir}")
        print("This may take a few minutes...")
        n = rag.ingest_html_docs(args.docs_dir)
        print(f"  -> {n} HTML chunks ingested")
        total += n
    else:
        print(f"HTML docs directory not found: {args.docs_dir}")
        print("Skipping HTML ingestion (will use manual docs only).")

    # 2. Ingest manual JSON docs
    manual_path = os.path.join(_PROJECT_ROOT, "rag_docs", "abaqus_api_manual.json")
    if os.path.isfile(manual_path):
        print(f"Ingesting manual docs from: {manual_path}")
        with open(manual_path, "r", encoding="utf-8") as f:
            docs_list = json.load(f)
        n = rag.ingest_manual_docs(docs_list)
        print(f"  -> {n} manual entries ingested")
        total += n
    else:
        print(f"Manual docs not found: {manual_path}")

    print(f"\nDone! Total documents in index: {rag.count()}")
    print(f"Database stored at: {db_path}")


if __name__ == "__main__":
    main()
