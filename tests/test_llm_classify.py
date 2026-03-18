"""LLM 分类器效果测试 — 对比关键词匹配 vs Kimi 分类。

包含：
  1. 12 种标准场景（关键词匹配也能覆盖的）
  2. 10 种刁钻表述（关键词匹配容易失败的）

Usage:
    python tests/test_llm_classify.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.llm import LLMClient
from agent.prompts import select_examples, _select_by_keywords, _classify_by_llm, EXAMPLES


# ── 标准场景：关键词匹配也应该能覆盖 ──
STANDARD_CASES = [
    ("悬臂梁右端面施加 10MPa 压力",     "static_pressure"),
    ("中心施加 500N 集中力",            "static_force"),
    ("分析梁在自重作用下的变形",          "gravity"),
    ("考虑大变形和塑性效应",             "nonlinear"),
    ("悬臂梁的前5阶固有频率",            "frequency"),
    ("钢杆受到冲击载荷，显式分析",        "dynamic"),
    ("薄壁圆筒受外压",                  "shell"),
    ("右端施加强制位移1mm",             "displacement"),
    ("实心圆柱体轴对称受压",             "revolve"),
    ("两个零件装配在一起分析",            "assembly"),
    ("固定端附近网格细化加密",            "mesh_control"),
    ("需要自定义场输出包括PEEQ",         "field_output"),
]

# ── 刁钻表述：关键词匹配容易失败的 ──
TRICKY_CASES = [
    ("分析一个梁的振动特性",             "frequency",    "没有'模态''频率'关键词"),
    ("求解结构的动态响应特征",            "frequency",    "'动态响应'不在关键词表"),
    ("做一个弹塑性分析",                "nonlinear",    "'弹塑性'不是直接关键词"),
    ("柱体轴压稳定性分析",              "static_pressure", "'稳定性分析'通常是静力学"),
    ("两个零件的连接分析",              "assembly",     "'连接'不在assembly关键词中"),
    ("分析热应力问题",                  "static_pressure", "不支持热分析，应回退到静力学"),
    ("一根管子承受内压",                "shell",        "'管子''内压'不是直接关键词"),
    ("分析结构在地震作用下的响应",        "dynamic",      "'地震'不在关键词表"),
    ("带孔板的应力集中分析，孔边加密网格", "mesh_control",  "'应力集中''孔边加密'不是直接关键词"),
    ("圆形法兰盘受轴向压力",             "revolve",      "'法兰盘'包含'法兰'但'圆形'也可能干扰"),
]


def test_standard(llm):
    """测试标准场景。"""
    print("=" * 70)
    print("第一部分：标准场景（12 种）")
    print("=" * 70)

    kw_pass = 0
    llm_pass = 0

    for user_input, expected_key in STANDARD_CASES:
        # 关键词匹配
        kw_keys = _select_by_keywords(user_input)
        kw_hit = expected_key in kw_keys

        # LLM 分类
        llm_keys = _classify_by_llm(user_input, llm)
        llm_hit = expected_key in llm_keys

        kw_status = "OK" if kw_hit else "MISS"
        llm_status = "OK" if llm_hit else "MISS"

        if kw_hit:
            kw_pass += 1
        if llm_hit:
            llm_pass += 1

        print(f"  \"{user_input}\"")
        print(f"    期望: {expected_key}")
        print(f"    关键词: [{kw_status}] {kw_keys}")
        print(f"    LLM:   [{llm_status}] {llm_keys}")
        print()

        time.sleep(0.5)

    total = len(STANDARD_CASES)
    print(f"  标准场景: 关键词 {kw_pass}/{total}, LLM {llm_pass}/{total}")
    return kw_pass, llm_pass, total


def test_tricky(llm):
    """测试刁钻表述。"""
    print("\n" + "=" * 70)
    print("第二部分：刁钻表述（10 种 — 关键词匹配容易失败）")
    print("=" * 70)

    kw_pass = 0
    llm_pass = 0

    for user_input, expected_key, note in TRICKY_CASES:
        # 关键词匹配
        kw_keys = _select_by_keywords(user_input)
        kw_hit = expected_key in kw_keys

        # LLM 分类
        llm_keys = _classify_by_llm(user_input, llm)
        llm_hit = expected_key in llm_keys

        kw_status = "OK" if kw_hit else "MISS"
        llm_status = "OK" if llm_hit else "MISS"

        if kw_hit:
            kw_pass += 1
        if llm_hit:
            llm_pass += 1

        print(f"  \"{user_input}\"  ({note})")
        print(f"    期望: {expected_key}")
        print(f"    关键词: [{kw_status}] {kw_keys}")
        print(f"    LLM:   [{llm_status}] {llm_keys}")
        print()

        time.sleep(0.5)

    total = len(TRICKY_CASES)
    print(f"  刁钻表述: 关键词 {kw_pass}/{total}, LLM {llm_pass}/{total}")
    return kw_pass, llm_pass, total


def main():
    print("关键词匹配 vs LLM 分类 — 对比测试\n")
    llm = LLMClient()

    std_kw, std_llm, std_total = test_standard(llm)
    tricky_kw, tricky_llm, tricky_total = test_tricky(llm)

    all_kw = std_kw + tricky_kw
    all_llm = std_llm + tricky_llm
    all_total = std_total + tricky_total

    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print(f"  {'场景':12s} {'关键词':>10s} {'LLM':>10s}")
    print(f"  {'─' * 35}")
    print(f"  {'标准(12)':12s} {std_kw:>7d}/{std_total:<2d}  {std_llm:>7d}/{std_total:<2d}")
    print(f"  {'刁钻(10)':12s} {tricky_kw:>7d}/{tricky_total:<2d}  {tricky_llm:>7d}/{tricky_total:<2d}")
    print(f"  {'─' * 35}")
    print(f"  {'总计(22)':12s} {all_kw:>7d}/{all_total:<2d}  {all_llm:>7d}/{all_total:<2d}")
    print("=" * 70)


if __name__ == "__main__":
    main()
