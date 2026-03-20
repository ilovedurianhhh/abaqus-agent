#!/usr/bin/env python
"""Compare LLM output quality with and without RAG context.

Sends the same set of test queries to the LLM twice:
1. Without RAG — only simplified API + few-shot examples
2. With RAG — adds native Abaqus API documentation from vector search

Compares whether the LLM can handle tasks beyond the simplified API coverage.

Usage:
    python scripts/test_rag_comparison.py
"""

import io
import json
import os
import sys
import time
import logging

# Offline mode for embedding model
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Suppress noisy logs
logging.basicConfig(level=logging.WARNING)

# ──────────────────────────────────────────────────────────────
# Test cases: queries that REQUIRE native API (not covered by simplified API)
# ──────────────────────────────────────────────────────────────
TEST_QUERIES = [
    {
        "id": "contact",
        "query": "两个钢块叠放在一起，上面的块顶部受10MPa压力，底部块底面固定，两个块之间有接触，摩擦系数0.3",
        "category": "接触分析（简化API不支持）",
        "expect_native": True,
        "key_patterns": ["ContactProperty", "SurfaceToSurfaceContactStd", "TangentialBehavior"],
    },
    {
        "id": "heat",
        "query": "一根200x20x20mm的钢杆，左端温度200度，右端温度25度，求稳态温度分布",
        "category": "热传导分析（简化API不支持）",
        "expect_native": True,
        "key_patterns": ["HeatTransferStep", "Conductivity", "TemperatureBC"],
    },
    {
        "id": "buckling",
        "query": "一根细长钢柱500x10x10mm，底部固定，顶部受轴向压力，求前3阶屈曲临界载荷",
        "category": "屈曲分析（简化API不支持）",
        "expect_native": True,
        "key_patterns": ["BuckleStep", "numEigen"],
    },
    {
        "id": "static_basic",
        "query": "分析一个200x20x20mm的钢制悬臂梁，左端固定，右端面施加5MPa压力",
        "category": "基础静力学（简化API支持）",
        "expect_native": False,
        "key_patterns": ["AbaqusModel", "create_static", "pressure", "submit"],
    },
]


def build_prompt_without_rag(user_input, llm_client):
    """Build system prompt WITHOUT RAG (original behavior)."""
    from agent.prompts import build_system_prompt
    return build_system_prompt(user_input=user_input, llm_client=llm_client, rag=None)


def build_prompt_with_rag(user_input, llm_client, rag):
    """Build system prompt WITH RAG."""
    from agent.prompts import build_system_prompt
    return build_system_prompt(user_input=user_input, llm_client=llm_client, rag=rag)


def call_llm(llm_client, system_prompt, user_input, max_retries=3):
    """Call LLM with retry on 429 errors."""
    messages = [{"role": "user", "content": user_input}]
    for attempt in range(max_retries):
        try:
            response = llm_client.generate(
                system=system_prompt,
                messages=messages,
                max_tokens=4096,
            )
            return response
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = 15 * (attempt + 1)
                print(f"   (429 rate limited, waiting {wait}s before retry...)")
                time.sleep(wait)
            else:
                return f"[LLM ERROR] {e}"


def extract_code(response):
    """Extract <code> block from LLM response."""
    import re
    match = re.search(r"<code>(.*?)</code>", response, re.DOTALL)
    return match.group(1).strip() if match else ""


def check_patterns(code, patterns):
    """Check which expected patterns appear in generated code."""
    found = []
    missing = []
    for p in patterns:
        if p in code:
            found.append(p)
        else:
            missing.append(p)
    return found, missing


def analyze_response(response, test_case):
    """Analyze a single LLM response against expectations."""
    code = extract_code(response)
    has_code = bool(code)
    found, missing = check_patterns(code or response, test_case["key_patterns"])

    # Check if it uses native API or simplified API
    uses_native = any(p in (code or response) for p in [
        "mdb.models", "m._buf.emit", "from abaqus import",
    ])
    uses_simplified = "AbaqusModel" in (code or response)

    # Check if it admits inability
    admits_limitation = any(kw in response for kw in [
        "不支持", "无法", "not supported", "cannot", "没有", "不能",
        "超出", "beyond", "不具备",
    ])

    return {
        "has_code": has_code,
        "code_lines": len(code.splitlines()) if code else 0,
        "patterns_found": found,
        "patterns_missing": missing,
        "pattern_hit_rate": len(found) / len(test_case["key_patterns"]) if test_case["key_patterns"] else 1.0,
        "uses_native": uses_native,
        "uses_simplified": uses_simplified,
        "admits_limitation": admits_limitation,
    }


def print_separator():
    print("=" * 78)


def main():
    from agent.llm import LLMClient
    from agent.rag import AbaqusRAG

    print_separator()
    print("RAG vs No-RAG Comparison Test")
    print_separator()

    # Initialize
    llm = LLMClient()
    print(f"LLM: {llm.model}")

    rag = AbaqusRAG()
    print(f"RAG: {rag.count()} documents indexed")
    print(f"Test queries: {len(TEST_QUERIES)}")
    print()

    results = []

    for tc in TEST_QUERIES:
        print_separator()
        print(f"[{tc['id']}] {tc['category']}")
        print(f"Query: {tc['query']}")
        print_separator()

        # ── Without RAG ──
        print("\n>> WITHOUT RAG:")
        t0 = time.time()
        prompt_no_rag = build_prompt_without_rag(tc["query"], llm)
        resp_no_rag = call_llm(llm, prompt_no_rag, tc["query"])
        time_no_rag = time.time() - t0
        analysis_no_rag = analyze_response(resp_no_rag, tc)

        code_no = extract_code(resp_no_rag)
        if code_no:
            # Show first 15 lines of code
            lines = code_no.splitlines()
            preview = "\n".join(lines[:15])
            if len(lines) > 15:
                preview += f"\n... ({len(lines)} lines total)"
            print(f"   Code: YES ({analysis_no_rag['code_lines']} lines)")
            print(f"   ---\n{preview}\n   ---")
        else:
            # Show first 200 chars of text response
            print(f"   Code: NO")
            print(f"   Response preview: {resp_no_rag[:300]}...")

        print(f"   Patterns found: {analysis_no_rag['patterns_found']}")
        print(f"   Patterns missing: {analysis_no_rag['patterns_missing']}")
        print(f"   Hit rate: {analysis_no_rag['pattern_hit_rate']:.0%}")
        print(f"   Uses native API: {analysis_no_rag['uses_native']}")
        print(f"   Admits limitation: {analysis_no_rag['admits_limitation']}")
        print(f"   Time: {time_no_rag:.1f}s")

        # Delay between LLM calls to avoid rate limiting
        time.sleep(5)

        # ── With RAG ──
        print("\n>> WITH RAG:")
        t0 = time.time()
        prompt_with_rag = build_prompt_with_rag(tc["query"], llm, rag)
        resp_with_rag = call_llm(llm, prompt_with_rag, tc["query"])
        time_with_rag = time.time() - t0
        analysis_with_rag = analyze_response(resp_with_rag, tc)

        # Show RAG context that was retrieved
        rag_results = rag.retrieve(tc["query"], top_k=3)
        rag_titles = [r["title"] for r in rag_results]
        print(f"   RAG retrieved: {rag_titles}")

        code_rag = extract_code(resp_with_rag)
        if code_rag:
            lines = code_rag.splitlines()
            preview = "\n".join(lines[:15])
            if len(lines) > 15:
                preview += f"\n... ({len(lines)} lines total)"
            print(f"   Code: YES ({analysis_with_rag['code_lines']} lines)")
            print(f"   ---\n{preview}\n   ---")
        else:
            print(f"   Code: NO")
            print(f"   Response: {resp_with_rag[:200]}...")

        print(f"   Patterns found: {analysis_with_rag['patterns_found']}")
        print(f"   Patterns missing: {analysis_with_rag['patterns_missing']}")
        print(f"   Hit rate: {analysis_with_rag['pattern_hit_rate']:.0%}")
        print(f"   Uses native API: {analysis_with_rag['uses_native']}")
        print(f"   Time: {time_with_rag:.1f}s")

        results.append({
            "id": tc["id"],
            "category": tc["category"],
            "expect_native": tc["expect_native"],
            "no_rag": analysis_no_rag,
            "with_rag": analysis_with_rag,
            "time_no_rag": time_no_rag,
            "time_with_rag": time_with_rag,
        })

        # Delay between test cases
        if tc != TEST_QUERIES[-1]:
            time.sleep(5)

    # ──────────────────────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────────────────────
    print("\n")
    print_separator()
    print("SUMMARY")
    print_separator()
    print(f"{'Test':<12} {'Category':<28} {'No-RAG Hit':>10} {'RAG Hit':>10} {'Improved':>10}")
    print("-" * 78)

    improved_count = 0
    for r in results:
        nr = r["no_rag"]["pattern_hit_rate"]
        wr = r["with_rag"]["pattern_hit_rate"]
        delta = wr - nr
        marker = ""
        if delta > 0:
            marker = f"+{delta:.0%} ✓"
            improved_count += 1
        elif delta == 0:
            marker = "same"
        else:
            marker = f"{delta:.0%} ✗"
        print(f"{r['id']:<12} {r['category']:<28} {nr:>9.0%} {wr:>9.0%} {marker:>10}")

    print("-" * 78)

    # Aggregate stats
    avg_no_rag = sum(r["no_rag"]["pattern_hit_rate"] for r in results) / len(results)
    avg_with_rag = sum(r["with_rag"]["pattern_hit_rate"] for r in results) / len(results)
    native_queries = [r for r in results if r["expect_native"]]
    avg_native_no = sum(r["no_rag"]["pattern_hit_rate"] for r in native_queries) / len(native_queries) if native_queries else 0
    avg_native_rag = sum(r["with_rag"]["pattern_hit_rate"] for r in native_queries) / len(native_queries) if native_queries else 0

    print(f"\nOverall pattern hit rate:    No-RAG {avg_no_rag:.0%}  →  RAG {avg_with_rag:.0%}")
    print(f"Native-API queries only:     No-RAG {avg_native_no:.0%}  →  RAG {avg_native_rag:.0%}")
    print(f"Queries improved by RAG:     {improved_count}/{len(results)}")

    # Timing
    avg_t_no = sum(r["time_no_rag"] for r in results) / len(results)
    avg_t_rag = sum(r["time_with_rag"] for r in results) / len(results)
    print(f"Avg response time:           No-RAG {avg_t_no:.1f}s  →  RAG {avg_t_rag:.1f}s")

    # Save detailed results
    output_path = os.path.join(_PROJECT_ROOT, "scripts", "rag_comparison_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
