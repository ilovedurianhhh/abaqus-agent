#!/usr/bin/env python
"""End-to-end performance test for the Abaqus Agent pipeline.

Tests the full flow: RAG retrieval → prompt build → LLM generation → code validation
Covers both simplified-API tasks and native-API (RAG-dependent) tasks.

Usage:
    python scripts/test_performance.py
"""

import io
import os
import sys
import time
import logging

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(level=logging.WARNING)

# ─────────────────────────────────────────────
# Test cases
# ─────────────────────────────────────────────
TEST_CASES = [
    {
        "id": "static_basic",
        "query": "分析一个200x20x20mm的钢制悬臂梁，左端固定，右端面施加5MPa压力",
        "category": "基础静力学",
        "type": "simplified",
        "expect_patterns": ["AbaqusModel", "create_static", "pressure", "submit", "fix"],
        "reject_patterns": ["mdb.models", "_buf.emit"],
    },
    {
        "id": "modal",
        "query": "分析一个300x20x10mm钢制悬臂梁的前5阶固有频率",
        "category": "模态分析",
        "type": "simplified",
        "expect_patterns": ["AbaqusModel", "create_frequency", "num_eigen", "create_density"],
        "reject_patterns": [],
    },
    {
        "id": "contact",
        "query": "两个50x50x20mm的钢块叠放，上块顶面10MPa压力，下块底面固定，两块之间有摩擦接触，摩擦系数0.3",
        "category": "接触分析 (RAG)",
        "type": "native",
        "expect_patterns": ["ContactProperty", "TangentialBehavior"],
        "reject_patterns": [],
    },
    {
        "id": "heat",
        "query": "一根200x20x20mm的钢杆，左端温度200度，右端温度25度，求稳态温度分布",
        "category": "热传导 (RAG)",
        "type": "native",
        "expect_patterns": ["HeatTransferStep", "Conductivity", "TemperatureBC"],
        "reject_patterns": [],
    },
    {
        "id": "buckling",
        "query": "一根细长钢柱500x10x10mm，底部固定，顶部受轴向压力1MPa，求前3阶屈曲临界载荷",
        "category": "屈曲分析 (RAG)",
        "type": "native",
        "expect_patterns": ["BuckleStep"],
        "reject_patterns": [],
    },
    {
        "id": "force",
        "query": "一块100x100x5mm铝板，四边固定，中心施加500N向下的集中力",
        "category": "集中力",
        "type": "simplified",
        "expect_patterns": ["AbaqusModel", "concentrated_force_at_point", "generate"],
        "reject_patterns": [],
    },
]


def sep(char="=", width=78):
    print(char * width)


def extract_code(response):
    import re
    m = re.search(r"<code>(.*?)</code>", response, re.DOTALL)
    return m.group(1).strip() if m else ""


def extract_plan(response):
    import re
    m = re.search(r"<plan>(.*?)</plan>", response, re.DOTALL)
    return m.group(1).strip() if m else ""


def main():
    from agent.llm import LLMClient
    from agent.rag import AbaqusRAG
    from agent.prompts import build_system_prompt
    from agent.code_validator import validate_api_calls
    from agent.agent import _validate_code, CodeValidationError

    sep()
    print("Abaqus Agent End-to-End Performance Test")
    sep()

    llm = LLMClient()
    rag = AbaqusRAG()
    print(f"LLM model:   {llm.model}")
    print(f"RAG docs:    {rag.count()}")
    print(f"Test cases:  {len(TEST_CASES)}")
    print()

    results = []

    for i, tc in enumerate(TEST_CASES):
        sep("-")
        print(f"[{i+1}/{len(TEST_CASES)}] {tc['id']} — {tc['category']}")
        print(f"Query: {tc['query'][:70]}...")
        sep("-")

        timings = {}
        errors = []

        # ── Phase 1: RAG Retrieval ──
        t0 = time.time()
        rag_results = rag.retrieve(tc["query"], top_k=5)
        timings["rag"] = time.time() - t0

        rag_titles = [r["title"] for r in rag_results]
        rag_manual = [r for r in rag_results if r["source"] == "manual"]
        print(f"  RAG: {len(rag_results)} results ({len(rag_manual)} manual) in {timings['rag']:.2f}s")
        if rag_results:
            for r in rag_results[:3]:
                print(f"    {r['distance']:.4f} | {r['title'][:50]}")

        # ── Phase 2: Prompt Build ──
        t0 = time.time()
        system_prompt = build_system_prompt(
            user_input=tc["query"], llm_client=llm, rag=rag
        )
        timings["prompt"] = time.time() - t0
        prompt_len = len(system_prompt)
        print(f"  Prompt: {prompt_len} chars, built in {timings['prompt']:.2f}s")

        # ── Phase 3: LLM Generation ──
        print(f"  LLM: generating...", end="", flush=True)
        t0 = time.time()
        try:
            response = llm.generate(
                system=system_prompt,
                messages=[{"role": "user", "content": tc["query"]}],
                max_tokens=4096,
            )
            timings["llm"] = time.time() - t0
            print(f" {timings['llm']:.1f}s")
        except Exception as e:
            timings["llm"] = time.time() - t0
            print(f" FAILED ({e})")
            errors.append(f"LLM error: {e}")
            results.append({**tc, "timings": timings, "errors": errors,
                           "code_lines": 0, "patterns_found": [], "patterns_missing": tc["expect_patterns"],
                           "validation_ok": False, "has_plan": False, "has_code": False})
            time.sleep(5)
            continue

        # ── Phase 4: Parse Response ──
        plan = extract_plan(response)
        code = extract_code(response)
        has_plan = bool(plan)
        has_code = bool(code)
        code_lines = len(code.splitlines()) if code else 0

        print(f"  Parse: plan={'YES' if has_plan else 'NO'}, code={'YES' if has_code else 'NO'} ({code_lines} lines)")

        if plan:
            # Show first 3 lines of plan
            plan_preview = "\n".join(plan.splitlines()[:3])
            print(f"    Plan: {plan_preview}")

        if not has_code:
            # Show response preview
            print(f"    Response: {response[:200]}...")
            errors.append("No <code> block in response")

        # ── Phase 5: Code Validation ──
        safety_ok = False
        api_ok = False
        api_errors = []

        if has_code:
            # Safety validation
            try:
                _validate_code(code)
                safety_ok = True
            except CodeValidationError as e:
                errors.append(f"Safety: {e}")

            # API signature validation
            api_errors = validate_api_calls(code)
            api_ok = len(api_errors) == 0
            if api_errors:
                for ae in api_errors[:3]:
                    errors.append(f"API: {ae}")

        validation_ok = safety_ok and api_ok
        timings["validate"] = 0  # negligible

        print(f"  Validation: safety={'PASS' if safety_ok else 'FAIL'}, api={'PASS' if api_ok else 'FAIL'}")
        if api_errors:
            for ae in api_errors[:2]:
                print(f"    {ae[:70]}")

        # ── Phase 6: Pattern Check ──
        check_text = code or response
        found = [p for p in tc["expect_patterns"] if p in check_text]
        missing = [p for p in tc["expect_patterns"] if p not in check_text]
        rejected = [p for p in tc.get("reject_patterns", []) if p in check_text]

        hit_rate = len(found) / len(tc["expect_patterns"]) if tc["expect_patterns"] else 1.0
        print(f"  Patterns: {len(found)}/{len(tc['expect_patterns'])} ({hit_rate:.0%})")
        if missing:
            print(f"    Missing: {missing}")
        if rejected:
            print(f"    Unexpected: {rejected}")
            errors.append(f"Unexpected patterns: {rejected}")

        # ── Code Preview ──
        if has_code:
            lines = code.splitlines()
            preview = "\n".join(f"    {l}" for l in lines[:10])
            if len(lines) > 10:
                preview += f"\n    ... ({len(lines)} lines total)"
            print(f"  Code:\n{preview}")

        total_time = sum(timings.values())
        print(f"  Total: {total_time:.1f}s (rag={timings['rag']:.2f}s, prompt={timings['prompt']:.2f}s, llm={timings['llm']:.1f}s)")

        results.append({
            **tc,
            "timings": timings,
            "errors": errors,
            "code_lines": code_lines,
            "patterns_found": found,
            "patterns_missing": missing,
            "hit_rate": hit_rate,
            "validation_ok": validation_ok,
            "has_plan": has_plan,
            "has_code": has_code,
        })

        # Rate limit delay
        if i < len(TEST_CASES) - 1:
            time.sleep(3)

    # ─────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────
    print("\n")
    sep()
    print("PERFORMANCE SUMMARY")
    sep()

    print(f"\n{'Test':<15} {'Category':<20} {'Code':>5} {'Valid':>6} {'Patterns':>9} {'LLM(s)':>7} {'Errors':>7}")
    print("-" * 78)

    for r in results:
        code_mark = "YES" if r["has_code"] else "NO"
        valid_mark = "PASS" if r["validation_ok"] else "FAIL"
        pattern_str = f"{len(r['patterns_found'])}/{len(r['patterns_found'])+len(r['patterns_missing'])}"
        llm_time = f"{r['timings'].get('llm', 0):.1f}"
        err_count = str(len(r["errors"])) if r["errors"] else "-"
        print(f"{r['id']:<15} {r['category']:<20} {code_mark:>5} {valid_mark:>6} {pattern_str:>9} {llm_time:>7} {err_count:>7}")

    # Aggregate
    total = len(results)
    has_code_count = sum(1 for r in results if r["has_code"])
    valid_count = sum(1 for r in results if r["validation_ok"])
    avg_hit = sum(r.get("hit_rate", 0) for r in results) / total
    avg_llm = sum(r["timings"].get("llm", 0) for r in results) / total
    avg_rag = sum(r["timings"].get("rag", 0) for r in results) / total

    simplified = [r for r in results if r["type"] == "simplified"]
    native = [r for r in results if r["type"] == "native"]
    simp_hit = sum(r.get("hit_rate", 0) for r in simplified) / len(simplified) if simplified else 0
    nat_hit = sum(r.get("hit_rate", 0) for r in native) / len(native) if native else 0

    print("-" * 78)
    print(f"\nCode generation rate:     {has_code_count}/{total} ({has_code_count/total:.0%})")
    print(f"Validation pass rate:     {valid_count}/{total} ({valid_count/total:.0%})")
    print(f"Overall pattern hit rate: {avg_hit:.0%}")
    print(f"  Simplified API tasks:   {simp_hit:.0%} ({len(simplified)} tests)")
    print(f"  Native API tasks (RAG): {nat_hit:.0%} ({len(native)} tests)")
    print(f"Avg LLM response time:    {avg_llm:.1f}s")
    print(f"Avg RAG retrieval time:   {avg_rag:.3f}s")


if __name__ == "__main__":
    main()
