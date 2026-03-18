"""路由效果演示脚本 — 验证所有 12 种分析类型的关键词匹配。

Usage:
    python tests/test_routing_demo.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.prompts import select_examples, build_system_prompt, EXAMPLES


def test_all_routes():
    """逐个测试每种分析类型的路由效果。"""
    test_cases = [
        # (用户输入, 期望命中的 key, 期望包含的 API 方法)
        ("悬臂梁右端面施加 10MPa 压力",       "static_pressure", ["pressure"]),
        ("中心施加 500N 集中力",              "static_force",    ["concentrated_force"]),
        ("分析梁在自重作用下的变形",            "gravity",         ["create_density", "gravity"]),
        ("考虑大变形和塑性效应",               "nonlinear",       ["nlgeom=True", "create_plastic"]),
        ("悬臂梁的前5阶固有频率",              "frequency",       ["create_frequency"]),
        ("钢杆受到冲击载荷，显式分析",          "dynamic",         ["create_dynamic_explicit"]),
        ("薄壁圆筒受外压",                    "shell",           ["extrude_shell", "create_shell_section"]),
        ("右端施加强制位移1mm",               "displacement",    ["displacement_bc"]),
        ("实心圆柱体轴对称受压",               "revolve",         ["revolve_solid"]),
        ("两个零件装配在一起分析",              "assembly",        ["translate"]),
        ("固定端附近网格细化加密",              "mesh_control",    ["seed_edge_by_number"]),
        ("需要自定义场输出包括PEEQ",           "field_output",    ["set_field_output", "field_output"]),
    ]

    print("=" * 70)
    print("关键词路由效果测试")
    print("=" * 70)

    passed = 0
    failed = 0

    for user_input, expected_key, expected_apis in test_cases:
        examples = select_examples(user_input)
        combined = "\n".join(examples)

        # 检查所有期望的 API 方法是否出现在返回的示例中
        missing = [api for api in expected_apis if api not in combined]

        if not missing:
            status = "PASS"
            passed += 1
        else:
            status = "FAIL"
            failed += 1

        print(f"\n[{status}] 输入: \"{user_input}\"")
        print(f"       期望类型: {expected_key}")
        print(f"       返回示例数: {len(examples)}")
        # 打印返回示例的第一行（标题）
        for i, ex in enumerate(examples):
            title = ex.strip().split("\n")[0]
            print(f"       示例 {i+1}: {title}")
        if missing:
            print(f"       缺少 API: {missing}")

    print("\n" + "=" * 70)
    print(f"结果: {passed} 通过, {failed} 失败, 共 {passed + failed} 项")
    print("=" * 70)
    return failed == 0


def test_default_fallback():
    """测试无关输入的默认回退。"""
    print("\n" + "=" * 70)
    print("默认回退测试")
    print("=" * 70)

    cases = [
        "你好",
        "帮我做个分析",
        "hello world",
        "",
    ]

    all_ok = True
    for user_input in cases:
        examples = select_examples(user_input)
        combined = "\n".join(examples)
        has_default = "BeamBending" in combined
        status = "PASS" if has_default else "FAIL"
        if not has_default:
            all_ok = False
        display = f"\"{user_input}\"" if user_input else "(空字符串)"
        print(f"[{status}] 输入: {display} → 返回默认 static_pressure 示例")

    return all_ok


def test_prompt_size():
    """测试不同输入下 prompt 大小是否保持稳定。"""
    print("\n" + "=" * 70)
    print("Prompt 大小稳定性测试")
    print("=" * 70)

    inputs = [
        ("默认 (无输入)",    ""),
        ("压力载荷",        "悬臂梁施加压力载荷"),
        ("模态分析",        "分析固有频率振动模态"),
        ("冲击动力",        "钢杆受到冲击载荷显式分析"),
        ("装配体",         "两个零件装配在一起"),
        ("网格细化",        "固定端网格加密细化"),
        ("场输出",         "自定义场输出PEEQ"),
    ]

    sizes = []
    for label, user_input in inputs:
        prompt = build_system_prompt(user_input=user_input)
        size = len(prompt)
        sizes.append(size)
        print(f"  {label:20s} → {size:,} 字符")

    max_s = max(sizes)
    min_s = min(sizes)
    ratio = max_s / min_s if min_s > 0 else 0
    ok = ratio < 1.5
    status = "PASS" if ok else "FAIL"
    print(f"\n[{status}] 最大/最小 = {max_s:,}/{min_s:,} = {ratio:.2f}x (阈值 < 1.5x)")
    return ok


def test_api_coverage():
    """验证所有 API 方法至少在一个示例中出现。"""
    print("\n" + "=" * 70)
    print("API 方法覆盖率测试")
    print("=" * 70)

    # 所有需要覆盖的 API 方法
    all_apis = [
        # PartBuilder
        "create_sketch", "rectangle", "circle", "line",
        "extrude_solid", "extrude_shell", "revolve_solid",
        "create_face_set", "create_surface", "create_set_by_bounding_box",
        # MaterialBuilder
        "create_elastic", "create_density", "create_plastic",
        "create_solid_section", "create_shell_section", "assign_section",
        # AssemblyBuilder
        "create_instance", "translate",
        # StepBuilder
        "create_static", "create_dynamic_explicit", "create_frequency",
        "set_field_output",
        # LoadBuilder
        "fix", "displacement_bc", "pressure", "concentrated_force", "gravity",
        # MeshBuilder
        "seed_part", "seed_edge_by_number", "set_element_type", "generate",
        # OdbReader
        "max_values", "field_output",
    ]

    # 合并所有示例文本
    all_examples_text = "\n".join(e["example"] for e in EXAMPLES.values())

    covered = []
    missing = []
    for api in all_apis:
        if api in all_examples_text:
            covered.append(api)
        else:
            missing.append(api)

    for api in covered:
        print(f"  [OK] {api}")
    for api in missing:
        print(f"  [MISSING] {api}")

    coverage = len(covered) / len(all_apis) * 100
    ok = len(missing) == 0
    status = "PASS" if ok else "FAIL"
    print(f"\n[{status}] 覆盖率: {len(covered)}/{len(all_apis)} ({coverage:.0f}%)")
    if missing:
        print(f"  未覆盖: {missing}")
    return ok


def main():
    results = []
    results.append(("路由匹配",     test_all_routes()))
    results.append(("默认回退",     test_default_fallback()))
    results.append(("Prompt 大小", test_prompt_size()))
    results.append(("API 覆盖率",  test_api_coverage()))

    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    all_ok = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"  [{status}] {name}")

    print("=" * 70)
    if all_ok:
        print("全部通过!")
    else:
        print("存在失败项，请检查上方输出。")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
