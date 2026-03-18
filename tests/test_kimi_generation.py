"""Kimi 代码生成质量测试 — 验证 Kimi 能否为所有 12 种分析场景生成正确代码。

对每种场景：
  1. 发送用户描述给 Kimi
  2. 检查返回中是否包含 <plan>、<code>、以及关键 API 调用
  3. 用 API 签名校验器验证生成代码的正确性
  4. 用安全校验器检查是否有危险调用

Usage:
    python tests/test_kimi_generation.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.llm import LLMClient
from agent.prompts import build_system_prompt
from agent.agent import _parse_response, _validate_code, CodeValidationError
from agent.code_validator import validate_api_calls


# 12 种测试场景
TEST_CASES = [
    {
        "name": "静力学-压力",
        "user_input": "分析一个 300x30x30mm 的钢制悬臂梁，左端固定，右端面施加 8MPa 压力",
        "required_apis": ["AbaqusModel", "create_sketch", "extrude_solid", "create_static",
                          "pressure", "create_surface", "fix", "submit"],
    },
    {
        "name": "静力学-集中力",
        "user_input": "一块 150x150x8mm 的铝板，四边固定，中心施加 300N 向下集中力",
        "required_apis": ["AbaqusModel", "concentrated_force", "fix", "submit"],
    },
    {
        "name": "重力加载",
        "user_input": "一根 400x25x25mm 的钢梁，两端简支，分析自重作用下的变形",
        "required_apis": ["AbaqusModel", "create_density", "gravity", "submit"],
    },
    {
        "name": "非线性大变形",
        "user_input": "一块 150x40x2mm 的铝板，一端固定，另一端施加 100N 集中力，考虑大变形和塑性",
        "required_apis": ["AbaqusModel", "nlgeom=True", "create_plastic", "submit"],
    },
    {
        "name": "模态分析",
        "user_input": "分析一个 250x15x10mm 钢制悬臂梁的前 8 阶固有频率",
        "required_apis": ["AbaqusModel", "create_frequency", "create_density", "submit"],
    },
    {
        "name": "动力显式",
        "user_input": "一根 80x8x8mm 的钢杆，左端固定，右端面受 30MPa 冲击压力，持续 0.002s",
        "required_apis": ["AbaqusModel", "create_dynamic_explicit", "create_density", "submit"],
    },
    {
        "name": "壳结构",
        "user_input": "一个 300x150mm 的钢制薄壁板，壁厚 1.5mm，一端固定，另一端施加 0.5MPa 压力，用壳单元分析",
        "required_apis": ["AbaqusModel", "extrude_shell", "create_shell_section", "submit"],
    },
    {
        "name": "位移约束",
        "user_input": "一根 120x12x12mm 的钢杆，左端固定，右端施加 0.5mm 的拉伸位移",
        "required_apis": ["AbaqusModel", "displacement_bc", "submit"],
    },
    {
        "name": "旋转体",
        "user_input": "一个半径 20mm、高 80mm 的钢制实心圆柱，底部固定，顶部施加 15MPa 压力，用旋转体建模",
        "required_apis": ["AbaqusModel", "revolve_solid", "submit"],
    },
    {
        "name": "多零件装配",
        "user_input": "一块 150x150x8mm 的钢底板上放一个 40x40x40mm 的铝块，底板底面固定，铝块顶面施加 3MPa 压力，需要建两个零件并装配",
        "required_apis": ["AbaqusModel", "create_instance", "translate", "submit"],
    },
    {
        "name": "网格细化",
        "user_input": "一个 200x20x20mm 的悬臂梁，左端固定处需要局部网格加密（用 seed_edge_by_number），右端面施加 10MPa 压力",
        "required_apis": ["AbaqusModel", "seed_edge_by_number", "submit"],
    },
    {
        "name": "自定义场输出",
        "user_input": "分析一个悬臂梁弯曲，材料有塑性，需要设置场输出包含 S, U, RF, PEEQ，并用 field_output 读取 PEEQ 数据",
        "required_apis": ["AbaqusModel", "set_field_output", "field_output", "submit"],
    },
]


def run_test(llm, case, index, total):
    """对单个场景调用 Kimi 并验证生成结果。"""
    name = case["name"]
    user_input = case["user_input"]
    required_apis = case["required_apis"]

    print(f"\n{'─' * 70}")
    print(f"[{index}/{total}] {name}")
    print(f"{'─' * 70}")
    print(f"  输入: {user_input}")

    # 构建 prompt（含路由选择的示例）
    system_prompt = build_system_prompt(user_input=user_input)
    messages = [{"role": "user", "content": user_input}]

    # 调用 Kimi
    try:
        t0 = time.time()
        response = llm.generate(system=system_prompt, messages=messages, max_tokens=4096)
        elapsed = time.time() - t0
    except Exception as e:
        print(f"  [ERROR] Kimi 调用失败: {e}")
        return {"name": name, "passed": False, "reason": f"LLM error: {e}"}

    # 解析响应
    plan, code = _parse_response(response)
    has_plan = bool(plan)
    has_code = bool(code)

    result = {
        "name": name,
        "elapsed": elapsed,
        "has_plan": has_plan,
        "has_code": has_code,
        "safety_ok": False,
        "api_errors": [],
        "missing_apis": [],
        "has_import": False,
        "has_odb": False,
        "passed": False,
        "code": code,
    }

    if not has_code:
        result["reason"] = "没有生成 <code> 块"
        _print_result(result)
        return result

    # 检查必需的 API 调用
    for api in required_apis:
        if api not in code:
            result["missing_apis"].append(api)

    result["has_import"] = "from abaqus_api import AbaqusModel" in code
    result["has_odb"] = "max_values" in code or "field_output" in code

    # ── 安全校验 ──
    try:
        _validate_code(code)
        result["safety_ok"] = True
    except CodeValidationError as e:
        result["safety_ok"] = False
        result["reason"] = f"安全校验失败: {e}"

    # ── API 签名校验 ──
    api_errors = validate_api_calls(code)
    result["api_errors"] = api_errors

    # 综合判定
    result["passed"] = (
        has_plan
        and has_code
        and result["has_import"]
        and result["has_odb"]
        and result["safety_ok"]
        and not result["missing_apis"]
        and not result["api_errors"]
    )

    _print_result(result)
    return result


def _print_result(r):
    """格式化打印单个测试结果。"""
    status = "PASS" if r["passed"] else "FAIL"

    if "elapsed" in r:
        print(f"  耗时: {r['elapsed']:.1f}s")
    print(f"  <plan>:  {'有' if r['has_plan'] else '缺失'}")
    print(f"  <code>:  {'有' if r['has_code'] else '缺失'}")
    print(f"  import:  {'有' if r.get('has_import') else '缺失'}")
    print(f"  ODB:     {'有' if r.get('has_odb') else '缺失'}")
    print(f"  安全校验: {'通过' if r.get('safety_ok') else '失败'}")

    if r.get("missing_apis"):
        print(f"  缺少 API: {r['missing_apis']}")

    if r.get("api_errors"):
        print(f"  API 签名问题 ({len(r['api_errors'])} 项):")
        for err in r["api_errors"]:
            print(f"    - {err}")
    else:
        print(f"  API 签名: 全部正确")

    print(f"  [{status}] {r['name']}")

    # 失败时打印完整代码
    if not r["passed"] and r.get("code"):
        print(f"\n  --- 生成的代码 ---")
        for i, line in enumerate(r["code"].split("\n"), 1):
            print(f"  {i:3d} | {line}")
        print(f"  --- 结束 ---")


def main():
    print("=" * 70)
    print("Kimi 代码生成质量测试（含 API 签名校验）")
    print("12 种分析场景 × Kimi 生成 + 安全校验 + API 签名校验")
    print("=" * 70)

    llm = LLMClient()
    total = len(TEST_CASES)

    results = []
    for i, case in enumerate(TEST_CASES, 1):
        r = run_test(llm, case, i, total)
        results.append(r)
        if i < total:
            time.sleep(1)

    # ── 汇总 ──
    pass_count = sum(1 for r in results if r["passed"])
    fail_count = total - pass_count
    api_error_count = sum(len(r.get("api_errors", [])) for r in results)

    print(f"\n{'=' * 70}")
    print("测试汇总")
    print(f"{'=' * 70}")

    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        api_note = ""
        if r.get("api_errors"):
            api_note = f"  (API问题: {len(r['api_errors'])}项)"
        if r.get("missing_apis"):
            api_note += f"  (缺少: {r['missing_apis']})"
        print(f"  [{status}] {r['name']}{api_note}")

    print(f"{'─' * 70}")
    print(f"  代码生成: {pass_count}/{total} 通过")
    print(f"  API签名问题总数: {api_error_count}")
    print(f"{'=' * 70}")

    if fail_count == 0:
        print("全部通过! Kimi 生成的代码全部通过安全校验和 API 签名校验。")
    else:
        print(f"有 {fail_count} 个场景未通过，请检查上方详细输出。")

    sys.exit(0 if fail_count == 0 else 1)


if __name__ == "__main__":
    main()
