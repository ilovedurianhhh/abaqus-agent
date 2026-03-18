"""复杂/模糊场景测试 — 验证 Kimi 面对复杂描述时能否生成可用代码。

测试三类情况：
  1. 模糊描述（缺少关键信息，看 Kimi 能否合理补全）
  2. 复杂组合（多载荷、多约束、复杂几何）
  3. 边界情况（非标准表述、口语化描述）

Usage:
    python tests/test_complex_scenarios.py
"""

import sys
import os
import time
import io

# Fix Windows GBK encoding issue
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.llm import LLMClient
from agent.prompts import build_system_prompt
from agent.agent import _parse_response, _validate_code, CodeValidationError
from agent.code_validator import validate_api_calls


# ── 第一类：模糊描述（缺尺寸、缺材料、缺参数） ──
VAGUE_CASES = [
    {
        "name": "缺尺寸",
        "user_input": "帮我分析一根悬臂梁，左端固定，右端受压力",
        "note": "没给尺寸和材料参数，Kimi 需要自行补全合理默认值",
    },
    {
        "name": "缺材料",
        "user_input": "一块 200x100x10mm 的板，四边固定，中心受 500N 集中力",
        "note": "没指定材料，Kimi 应该选择一种常见材料（钢/铝）",
    },
    {
        "name": "口语化",
        "user_input": "有个铁柱子，底下固定住，上面压一下，看看会怎样",
        "note": "非常口语化，没有具体参数",
    },
]

# ── 第二类：复杂组合（多载荷、多步骤、复杂几何） ──
COMPLEX_CASES = [
    {
        "name": "多载荷组合",
        "user_input": "一根 300x20x20mm 的钢梁，左端固定，右端面施加 5MPa 压力，同时在中点施加 200N 向下的集中力",
        "note": "同时有 pressure + concentrated_force，需要两个不同的面/集合",
    },
    {
        "name": "重力+压力组合",
        "user_input": "一根 500x30x30mm 的钢梁，两端简支，考虑自重，同时在顶面均匀施加 2MPa 压力",
        "note": "gravity + pressure 组合，需要 create_density",
    },
    {
        "name": "非线性+位移控制",
        "user_input": "一根 100x10x2mm 的铝薄板，左端固定，右端施加 5mm 强制弯曲位移，考虑大变形和塑性",
        "note": "displacement_bc + nlgeom=True + create_plastic 组合",
    },
    {
        "name": "壳结构+网格细化",
        "user_input": "一个 500x200mm 的钢制薄壁板，壁厚 2mm，一端固定，固定端附近网格加密，另一端施加 1MPa 压力，用壳单元",
        "note": "extrude_shell + seed_edge_by_number + pressure 组合",
    },
    {
        "name": "多零件+多载荷",
        "user_input": "一块 200x200x10mm 的钢底板上放一个 50x50x50mm 的铝块，底板四角固定，铝块顶面施加 5MPa 压力，底板底面受 1MPa 压力",
        "note": "assembly + 多个 pressure，需要两个零件各自建面",
    },
]

# ── 第三类：边界情况 ──
EDGE_CASES = [
    {
        "name": "英文输入",
        "user_input": "Analyze a 200x20x20mm steel cantilever beam, fixed at left end, 10MPa pressure on right face",
        "note": "英文描述，Kimi 应该也能处理",
    },
    {
        "name": "中英混合",
        "user_input": "做一个 beam 的模态分析，尺寸 300x20x15mm，steel 材料，求前 5 阶 frequency",
        "note": "中英混合表述",
    },
    {
        "name": "极简描述",
        "user_input": "圆柱受压",
        "note": "极简，只有 4 个字，Kimi 需要补全所有细节",
    },
]

ALL_CASES = [
    ("模糊描述", VAGUE_CASES),
    ("复杂组合", COMPLEX_CASES),
    ("边界情况", EDGE_CASES),
]


def run_test(llm, case, index, total):
    """对单个场景调用 Kimi 并验证。"""
    name = case["name"]
    user_input = case["user_input"]
    note = case.get("note", "")

    print(f"\n{'─' * 70}")
    print(f"[{index}/{total}] {name}")
    print(f"{'─' * 70}")
    print(f"  输入: {user_input}")
    if note:
        print(f"  难点: {note}")

    # 构建 prompt + 调用 Kimi
    system_prompt = build_system_prompt(user_input=user_input, llm_client=llm)
    messages = [{"role": "user", "content": user_input}]

    try:
        t0 = time.time()
        response = llm.generate(system=system_prompt, messages=messages, max_tokens=4096)
        elapsed = time.time() - t0
    except Exception as e:
        print(f"  [ERROR] Kimi 调用失败: {e}")
        return {"name": name, "passed": False, "reason": f"LLM error: {e}"}

    # 解析
    plan, code = _parse_response(response)
    result = {
        "name": name,
        "elapsed": elapsed,
        "has_plan": bool(plan),
        "has_code": bool(code),
        "safety_ok": False,
        "api_errors": [],
        "passed": False,
        "code": code,
        "plan": plan,
    }

    if not code:
        result["reason"] = "没有生成 <code> 块"
        _print_result(result)
        return result

    # 安全校验
    try:
        _validate_code(code)
        result["safety_ok"] = True
    except CodeValidationError as e:
        result["reason"] = f"安全校验失败: {e}"

    # API 签名校验
    api_errors = validate_api_calls(code)
    result["api_errors"] = api_errors

    # 基本结构检查
    has_import = "from abaqus_api import AbaqusModel" in code
    has_submit = "submit" in code
    has_odb = "max_values" in code or "field_output" in code

    result["has_import"] = has_import
    result["has_submit"] = has_submit
    result["has_odb"] = has_odb

    # 综合判定：有 plan + code，安全通过，API 签名无错
    result["passed"] = (
        result["has_plan"]
        and result["has_code"]
        and has_import
        and has_submit
        and result["safety_ok"]
        and not result["api_errors"]
    )

    _print_result(result)
    return result


def _print_result(r):
    """打印结果。"""
    status = "PASS" if r["passed"] else "FAIL"

    if "elapsed" in r:
        print(f"  耗时: {r['elapsed']:.1f}s")
    print(f"  <plan>:    {'有' if r['has_plan'] else '缺失'}")
    print(f"  <code>:    {'有' if r['has_code'] else '缺失'}")
    print(f"  import:    {'有' if r.get('has_import') else '缺失'}")
    print(f"  submit:    {'有' if r.get('has_submit') else '缺失'}")
    print(f"  ODB:       {'有' if r.get('has_odb') else '—'}")
    print(f"  安全校验:  {'通过' if r.get('safety_ok') else '失败'}")

    if r.get("api_errors"):
        print(f"  API 签名问题 ({len(r['api_errors'])} 项):")
        for err in r["api_errors"]:
            print(f"    - {err}")
    else:
        print(f"  API 签名:  全部正确")

    print(f"  [{status}] {r['name']}")

    if r.get("plan"):
        # 打印 plan 摘要（前 3 行）
        plan_lines = r["plan"].strip().split("\n")[:3]
        print(f"\n  --- plan 摘要 ---")
        for line in plan_lines:
            print(f"  {line}")
        if len(r["plan"].strip().split("\n")) > 3:
            print(f"  ...")

    # 失败时打印完整代码
    if not r["passed"] and r.get("code"):
        print(f"\n  --- 生成的代码 ---")
        for i, line in enumerate(r["code"].split("\n"), 1):
            print(f"  {i:3d} | {line}")
        print(f"  --- 结束 ---")


def main():
    print("=" * 70)
    print("复杂/模糊场景 Kimi 生成测试")
    print("=" * 70)

    llm = LLMClient()

    all_results = []
    case_index = 0
    total = sum(len(cases) for _, cases in ALL_CASES)

    for category, cases in ALL_CASES:
        print(f"\n{'=' * 70}")
        print(f"  {category}（{len(cases)} 个场景）")
        print(f"{'=' * 70}")

        for case in cases:
            case_index += 1
            r = run_test(llm, case, case_index, total)
            all_results.append((category, r))
            if case_index < total:
                time.sleep(1)

    # ── 汇总 ──
    print(f"\n{'=' * 70}")
    print("测试汇总")
    print(f"{'=' * 70}")

    category_stats = {}
    for category, r in all_results:
        if category not in category_stats:
            category_stats[category] = {"pass": 0, "total": 0}
        category_stats[category]["total"] += 1
        if r["passed"]:
            category_stats[category]["pass"] += 1

    total_pass = sum(s["pass"] for s in category_stats.values())
    total_count = sum(s["total"] for s in category_stats.values())
    total_api_errors = sum(len(r.get("api_errors", [])) for _, r in all_results)

    for category, r in all_results:
        status = "PASS" if r["passed"] else "FAIL"
        note = ""
        if r.get("api_errors"):
            note = f"  (API问题: {len(r['api_errors'])}项)"
        print(f"  [{status}] {r['name']}{note}")

    print(f"{'─' * 70}")
    for cat, stats in category_stats.items():
        print(f"  {cat}: {stats['pass']}/{stats['total']}")
    print(f"{'─' * 70}")
    print(f"  总计: {total_pass}/{total_count} 通过")
    print(f"  API 签名问题总数: {total_api_errors}")
    print(f"{'=' * 70}")

    if total_pass == total_count:
        print("全部通过!")
    else:
        fail_count = total_count - total_pass
        print(f"有 {fail_count} 个场景未通过。")

    sys.exit(0 if total_pass == total_count else 1)


if __name__ == "__main__":
    main()
