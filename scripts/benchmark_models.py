#!/usr/bin/env python
"""Download candidate embedding models and benchmark retrieval quality.

Usage:
    python scripts/benchmark_models.py
"""

import io
import json
import os
import sys
import time

# Force offline mode BEFORE any HF imports — all models are already cached
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# --- Candidate models ---
CANDIDATE_MODELS = [
    "all-MiniLM-L6-v2",                        # 90MB, English-only, fast
    "paraphrase-multilingual-MiniLM-L12-v2",    # 470MB, 50+ languages
    "distiluse-base-multilingual-cased-v2",     # 520MB, 50+ languages, distilled
]

# --- Test queries with expected best match title ---
TEST_CASES = [
    ("接触分析，两个零件之间有摩擦",       "Surface-to-Surface Contact (Standard)"),
    ("热传导分析，稳态温度场",             "Heat Transfer Analysis (Steady-State)"),
    ("屈曲分析，求临界载荷",               "Buckling Analysis (Linear Eigenvalue)"),
    ("螺栓预紧力",                         "Bolt Load (Pre-tension)"),
    ("弹簧单元连接到地面",                 "Spring/Dashpot Element"),
    ("cohesive zone delamination",          "Cohesive Zone / Cohesive Elements"),
    ("两个零件绑定在一起",                 "Tie Constraint"),
    ("热力耦合分析，考虑热膨胀",           "Thermal-Mechanical Coupled Analysis"),
    ("定义载荷随时间变化的曲线",           "Amplitude Definition"),
    ("刚体约束，用参考点控制",             "Rigid Body Constraint"),
    ("通用接触，显式分析中的碰撞",         "General Contact (Abaqus/Explicit)"),
    ("子模型分析，局部细化",               "Submodeling"),
]


def download_model(model_name):
    """Load a sentence-transformers model from cache."""
    from sentence_transformers import SentenceTransformer
    print(f"  Loading {model_name} from cache ...")
    t0 = time.time()
    model = SentenceTransformer(model_name)
    elapsed = time.time() - t0
    print(f"  Loaded in {elapsed:.1f}s")
    return model


def build_test_index(model_name):
    """Build a temporary ChromaDB index with manual docs + HTML docs, using given model."""
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    db_dir = os.path.join(_PROJECT_ROOT, f"rag_db_bench_{model_name.replace('/', '_')}")
    import shutil
    if os.path.exists(db_dir):
        shutil.rmtree(db_dir)

    ef = SentenceTransformerEmbeddingFunction(model_name=model_name)
    client = chromadb.PersistentClient(path=db_dir)
    collection = client.get_or_create_collection(name="bench", embedding_function=ef)

    # Load manual docs
    manual_path = os.path.join(_PROJECT_ROOT, "rag_docs", "abaqus_api_manual.json")
    with open(manual_path, "r", encoding="utf-8") as f:
        docs_list = json.load(f)

    chunks, ids, metas = [], [], []
    for i, entry in enumerate(docs_list):
        title = entry.get("title", "")
        content = entry.get("content", "")
        example = entry.get("example", "")
        text = f"## {title}\n{content}"
        if example:
            text += f"\n\n### Example\n{example}"
        chunks.append(text[:2000])
        ids.append(f"m_{i}")
        metas.append({"title": title, "source": "manual"})

    # Also load HTML docs using AbaqusRAG's ingestion logic
    sys.path.insert(0, _PROJECT_ROOT)
    from agent.rag import AbaqusRAG
    temp_rag = AbaqusRAG.__new__(AbaqusRAG)
    docs_dir = r"D:/SIMULIA/EstProducts/2022/CAADoc/win_b64.doc/English"
    if os.path.isdir(docs_dir):
        from bs4 import BeautifulSoup
        html_count = 0
        for root, _dirs, files in os.walk(docs_dir):
            for fname in files:
                if not fname.lower().endswith((".htm", ".html")):
                    continue
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        soup = BeautifulSoup(f.read(), "html.parser")
                except Exception:
                    continue
                page_chunks = temp_rag._split_by_headings(soup, fpath)
                for title, text, meta in page_chunks:
                    if len(text.strip()) < 30:
                        continue
                    chunks.append(text[:2000])
                    ids.append(f"h_{html_count}")
                    metas.append(meta)
                    html_count += 1
        print(f"  Index: {len(docs_list)} manual + {html_count} HTML = {len(chunks)} total")
    else:
        print(f"  Index: {len(docs_list)} manual docs only (HTML dir not found)")

    collection.add(documents=chunks, ids=ids, metadatas=metas)
    return collection, db_dir


def benchmark_model(model_name, collection):
    """Run test queries and compute hit rate + avg distance."""
    # Import augment helper for Chinese queries
    from agent.rag import AbaqusRAG
    rag = AbaqusRAG.__new__(AbaqusRAG)  # just to borrow _augment_query

    hits = 0
    total_dist = 0.0
    results_detail = []

    for query, expected_title in TEST_CASES:
        augmented = rag._augment_query(query)
        res = collection.query(query_texts=[augmented], n_results=3)
        top_title = res["metadatas"][0][0]["title"]
        top_dist = res["distances"][0][0]
        is_hit = (top_title == expected_title)
        if is_hit:
            hits += 1
        total_dist += top_dist

        # Also check if expected is in top-3
        top3_titles = [m["title"] for m in res["metadatas"][0]]
        in_top3 = expected_title in top3_titles

        results_detail.append({
            "query": query,
            "expected": expected_title,
            "got": top_title,
            "distance": top_dist,
            "hit": is_hit,
            "in_top3": in_top3,
        })

    return {
        "model": model_name,
        "hit_rate": hits / len(TEST_CASES),
        "top3_rate": sum(1 for r in results_detail if r["in_top3"]) / len(TEST_CASES),
        "avg_distance": total_dist / len(TEST_CASES),
        "details": results_detail,
    }


def cleanup_bench_dbs():
    """Remove temporary benchmark databases."""
    import shutil
    for name in os.listdir(_PROJECT_ROOT):
        if name.startswith("rag_db_bench_"):
            path = os.path.join(_PROJECT_ROOT, name)
            shutil.rmtree(path, ignore_errors=True)


def main():
    print("=" * 70)
    print("Embedding Model Benchmark for Abaqus RAG")
    print("=" * 70)
    print(f"Test queries: {len(TEST_CASES)}")
    print(f"Candidate models: {len(CANDIDATE_MODELS)}")
    print()

    all_results = []

    for model_name in CANDIDATE_MODELS:
        print(f"\n--- Model: {model_name} ---")
        try:
            download_model(model_name)
            collection, db_dir = build_test_index(model_name)
            result = benchmark_model(model_name, collection)
            all_results.append(result)

            print(f"  Hit@1: {result['hit_rate']:.0%} ({int(result['hit_rate'] * len(TEST_CASES))}/{len(TEST_CASES)})")
            print(f"  Hit@3: {result['top3_rate']:.0%}")
            print(f"  Avg distance: {result['avg_distance']:.4f}")

            # Show misses
            misses = [r for r in result["details"] if not r["hit"]]
            if misses:
                print(f"  Misses:")
                for m in misses:
                    t3 = "  (in top3)" if m["in_top3"] else ""
                    print(f"    Q: {m['query']}")
                    print(f"      Expected: {m['expected']}")
                    print(f"      Got:      {m['got']} (dist={m['distance']:.4f}){t3}")
        except Exception as e:
            print(f"  FAILED: {e}")

    # Summary table
    if all_results:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"{'Model':<50} {'Hit@1':>6} {'Hit@3':>6} {'AvgDist':>8}")
        print("-" * 70)
        for r in sorted(all_results, key=lambda x: -x["hit_rate"]):
            print(f"{r['model']:<50} {r['hit_rate']:>5.0%} {r['top3_rate']:>5.0%} {r['avg_distance']:>8.4f}")

        best = max(all_results, key=lambda x: (x["hit_rate"], x["top3_rate"], -x["avg_distance"]))
        print(f"\nBest model: {best['model']}")
        print(f"  Hit@1={best['hit_rate']:.0%}, Hit@3={best['top3_rate']:.0%}, AvgDist={best['avg_distance']:.4f}")

    # Cleanup
    cleanup_bench_dbs()
    print("\nBenchmark databases cleaned up.")


if __name__ == "__main__":
    main()
