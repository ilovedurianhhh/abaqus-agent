"""Tests for the Abaqus Agent layer.

Test 1-3 run without API key or Abaqus.
Test 4 (end-to-end) requires ANTHROPIC_API_KEY and Abaqus CAE.

Usage:
    python tests/test_agent.py              # offline tests only
    python tests/test_agent.py --e2e        # include end-to-end test
"""

import sys
import os
import unittest

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.prompts import build_system_prompt, build_api_reference, select_examples
from agent.agent import _parse_response, _format_result, _validate_code, CodeValidationError
from agent.history import ConversationHistory


class TestPrompts(unittest.TestCase):
    """Test 1: system prompt contains complete API reference."""

    def test_system_prompt_contains_api_methods(self):
        prompt = build_system_prompt()
        # Should contain key builder methods
        self.assertIn("m.part.create_sketch", prompt)
        self.assertIn("m.part.extrude_solid", prompt)
        self.assertIn("m.part.create_face_set", prompt)
        self.assertIn("m.part.create_surface", prompt)
        self.assertIn("m.material.create_elastic", prompt)
        self.assertIn("m.material.assign_section", prompt)
        self.assertIn("m.assembly.create_instance", prompt)
        self.assertIn("m.step.create_static", prompt)
        self.assertIn("m.load.fix", prompt)
        self.assertIn("m.load.pressure", prompt)
        self.assertIn("m.load.concentrated_force", prompt)
        self.assertIn("m.mesh.seed_part", prompt)
        self.assertIn("m.mesh.set_element_type", prompt)
        self.assertIn("m.mesh.generate", prompt)
        print("[PASS] System prompt contains all API methods")

    def test_system_prompt_contains_few_shot(self):
        prompt = build_system_prompt()
        # Default (no user_input) should include static_pressure example
        self.assertIn("AbaqusModel", prompt)
        self.assertIn("pressure", prompt)
        print("[PASS] System prompt contains few-shot examples")

    def test_api_reference_generation(self):
        ref = build_api_reference()
        # Should have section headers
        self.assertIn("PartBuilder", ref)
        self.assertIn("MaterialBuilder", ref)
        self.assertIn("LoadBuilder", ref)
        self.assertIn("MeshBuilder", ref)
        # Should not be empty
        self.assertGreater(len(ref), 200)
        print(f"[PASS] API reference generated ({len(ref)} chars)")


class TestExampleRouting(unittest.TestCase):
    """Test keyword-based example routing."""

    def test_modal_analysis_route(self):
        """Input '模态分析' should return frequency example."""
        examples = select_examples("做一个模态分析")
        combined = "\n".join(examples)
        self.assertIn("create_frequency", combined)
        print("[PASS] '模态分析' routes to frequency example")

    def test_frequency_keyword_route(self):
        """Input '固有频率' should return frequency example."""
        examples = select_examples("分析悬臂梁的前5阶固有频率")
        combined = "\n".join(examples)
        self.assertIn("create_frequency", combined)
        print("[PASS] '固有频率' routes to frequency example")

    def test_impact_route(self):
        """Input '冲击载荷' should return dynamic explicit example."""
        examples = select_examples("一根钢柱受到冲击载荷")
        combined = "\n".join(examples)
        self.assertIn("create_dynamic_explicit", combined)
        print("[PASS] '冲击载荷' routes to dynamic example")

    def test_gravity_route(self):
        """Input '自重' should return gravity example."""
        examples = select_examples("分析梁在自重作用下的变形")
        combined = "\n".join(examples)
        self.assertIn("create_density", combined)
        self.assertIn("gravity", combined)
        print("[PASS] '自重' routes to gravity example")

    def test_nonlinear_route(self):
        """Input '大变形' should return nonlinear example."""
        examples = select_examples("考虑大变形和塑性效应")
        combined = "\n".join(examples)
        self.assertIn("nlgeom=True", combined)
        self.assertIn("create_plastic", combined)
        print("[PASS] '大变形' routes to nonlinear example")

    def test_shell_route(self):
        """Input '薄壁' should return shell example."""
        examples = select_examples("一个薄壁圆筒受压")
        combined = "\n".join(examples)
        self.assertIn("extrude_shell", combined)
        self.assertIn("create_shell_section", combined)
        print("[PASS] '薄壁' routes to shell example")

    def test_displacement_route(self):
        """Input '强制位移' should return displacement example."""
        examples = select_examples("右端施加强制位移1mm")
        combined = "\n".join(examples)
        self.assertIn("displacement_bc", combined)
        print("[PASS] '强制位移' routes to displacement example")

    def test_no_keyword_defaults_to_static(self):
        """Unrelated input should default to static_pressure example."""
        examples = select_examples("你好，帮我做个分析")
        combined = "\n".join(examples)
        self.assertIn("pressure", combined)
        self.assertIn("BeamBending", combined)
        print("[PASS] No keyword match defaults to static_pressure")

    def test_empty_input_defaults(self):
        """Empty input should default to static_pressure example."""
        examples = select_examples("")
        self.assertEqual(len(examples), 1)
        self.assertIn("BeamBending", examples[0])
        print("[PASS] Empty input defaults to static_pressure")

    def test_top_k_limits_results(self):
        """At most top_k examples are returned."""
        examples = select_examples("模态分析", top_k=1)
        self.assertEqual(len(examples), 1)
        print("[PASS] top_k limits number of returned examples")

    def test_multiple_keywords_match(self):
        """Input with keywords from two categories returns both."""
        examples = select_examples("冲击动力显式分析")
        # Should match dynamic (冲击, 动力, 显式 — 3 keywords)
        combined = "\n".join(examples)
        self.assertIn("create_dynamic_explicit", combined)
        print("[PASS] Multiple keyword matches work correctly")

    def test_revolve_route(self):
        """Input '圆柱' should return revolve example."""
        examples = select_examples("一个实心圆柱体受压")
        combined = "\n".join(examples)
        self.assertIn("revolve_solid", combined)
        print("[PASS] '圆柱' routes to revolve example")

    def test_assembly_route(self):
        """Input '装配' should return assembly example."""
        examples = select_examples("两个零件装配在一起")
        combined = "\n".join(examples)
        self.assertIn("translate", combined)
        self.assertIn("create_instance", combined)
        print("[PASS] '装配' routes to assembly example")

    def test_mesh_control_route(self):
        """Input '网格细化' should return mesh_control example."""
        examples = select_examples("固定端附近网格细化加密")
        combined = "\n".join(examples)
        self.assertIn("seed_edge_by_number", combined)
        print("[PASS] '网格细化' routes to mesh_control example")

    def test_field_output_route(self):
        """Input '场输出' should return field_output example."""
        examples = select_examples("需要自定义场输出包括PEEQ")
        combined = "\n".join(examples)
        self.assertIn("set_field_output", combined)
        self.assertIn("field_output", combined)
        print("[PASS] '场输出' routes to field_output example")


class TestBuildSystemPromptWithInput(unittest.TestCase):
    """Test that build_system_prompt correctly injects routed examples."""

    def test_prompt_with_modal_input(self):
        prompt = build_system_prompt(user_input="分析固有频率和振动模态")
        self.assertIn("create_frequency", prompt)
        self.assertIn("IMPORTANT RULES", prompt)  # Role prompt still present
        print("[PASS] build_system_prompt with modal input works")

    def test_prompt_with_no_input(self):
        prompt = build_system_prompt()
        self.assertIn("BeamBending", prompt)  # Default static_pressure
        print("[PASS] build_system_prompt with no input works")

    def test_prompt_size_is_bounded(self):
        """Prompt should include at most 2 examples regardless of total."""
        prompt1 = build_system_prompt(user_input="压力载荷")
        prompt2 = build_system_prompt(user_input="模态分析")
        # Both should be roughly same size (within 50% of each other)
        ratio = len(prompt1) / len(prompt2) if len(prompt2) > 0 else 0
        self.assertGreater(ratio, 0.5)
        self.assertLess(ratio, 2.0)
        print(f"[PASS] Prompt sizes bounded: {len(prompt1)} vs {len(prompt2)} chars")


class TestCodeParsing(unittest.TestCase):
    """Test 2: parse <plan> and <code> blocks from LLM output."""

    def test_parse_full_response(self):
        response = """
Some intro text.
<plan>
- Geometry: 200x20x20 mm beam
- Material: Steel
</plan>
<code>
from abaqus_api import AbaqusModel
m = AbaqusModel("Test")
m.part.create_sketch("s", sheet_size=400)
</code>
"""
        plan, code = _parse_response(response)
        self.assertIn("200x20x20", plan)
        self.assertIn("AbaqusModel", code)
        self.assertIn("create_sketch", code)
        print("[PASS] Full response parsed correctly")

    def test_parse_code_only(self):
        response = """<code>
x = 1 + 1
</code>"""
        plan, code = _parse_response(response)
        self.assertEqual(plan, "")
        self.assertIn("x = 1", code)
        print("[PASS] Code-only response parsed correctly")

    def test_parse_no_code(self):
        response = "I need more information about your problem."
        plan, code = _parse_response(response)
        self.assertEqual(plan, "")
        self.assertEqual(code, "")
        print("[PASS] No-code response handled correctly")


class TestHistory(unittest.TestCase):
    """Test 3: conversation history management."""

    def test_add_and_retrieve(self):
        h = ConversationHistory(max_turns=5)
        h.add_user("hello")
        h.add_assistant("hi")
        msgs = h.to_messages()
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["role"], "user")
        self.assertEqual(msgs[1]["role"], "assistant")
        print("[PASS] History add/retrieve works")

    def test_rolling_window(self):
        h = ConversationHistory(max_turns=3)
        for i in range(10):
            h.add_user(f"msg-{i}")
            h.add_assistant(f"reply-{i}")
        msgs = h.to_messages()
        # max_turns=3 → 6 messages kept
        self.assertEqual(len(msgs), 6)
        # Oldest kept should be msg-7
        self.assertIn("msg-7", msgs[0]["content"])
        print("[PASS] Rolling window trims correctly")

    def test_clear(self):
        h = ConversationHistory()
        h.add_user("test")
        h.clear()
        self.assertEqual(len(h.to_messages()), 0)
        print("[PASS] History clear works")


class TestFormatResult(unittest.TestCase):
    """Test result formatting."""

    def test_format_success(self):
        plan = "- Steel beam\n- Fixed left"
        odb = {"max_mises": 42.15, "max_displacement": 1.2345, "max_rf_magnitude": 1000.0}
        text = _format_result(plan, odb, "/output/test")
        self.assertIn("42.15", text)
        self.assertIn("1.2345", text)
        self.assertIn("1000.00", text)
        self.assertIn("/output/test", text)
        print("[PASS] Success result formatted correctly")

    def test_format_error(self):
        plan = ""
        odb = {"error": True, "message": "ODB not found"}
        text = _format_result(plan, odb, "")
        self.assertIn("ODB not found", text)
        print("[PASS] Error result formatted correctly")


class TestCodeValidation(unittest.TestCase):
    """Test AST-based code safety validation."""

    def test_valid_abaqus_code(self):
        code = 'from abaqus_api import AbaqusModel\nm = AbaqusModel("Test")'
        _validate_code(code)  # Should not raise
        print("[PASS] Valid abaqus code passes validation")

    def test_reject_os_import(self):
        code = 'import os\nos.system("rm -rf /")'
        with self.assertRaises(CodeValidationError):
            _validate_code(code)
        print("[PASS] os import rejected")

    def test_reject_subprocess(self):
        code = 'import subprocess\nsubprocess.run(["ls"])'
        with self.assertRaises(CodeValidationError):
            _validate_code(code)
        print("[PASS] subprocess import rejected")

    def test_reject_eval(self):
        code = 'eval("1+1")'
        with self.assertRaises(CodeValidationError):
            _validate_code(code)
        print("[PASS] eval() call rejected")

    def test_reject_exec(self):
        code = 'exec("print(1)")'
        with self.assertRaises(CodeValidationError):
            _validate_code(code)
        print("[PASS] exec() call rejected")

    def test_reject_from_os_import(self):
        code = 'from os import system\nsystem("whoami")'
        with self.assertRaises(CodeValidationError):
            _validate_code(code)
        print("[PASS] from os import rejected")

    def test_syntax_error(self):
        code = 'def foo(\n'
        with self.assertRaises(CodeValidationError):
            _validate_code(code)
        print("[PASS] Syntax error rejected")

    def test_allow_math(self):
        code = 'import math\nx = math.sqrt(4)'
        _validate_code(code)  # math is not forbidden
        print("[PASS] math import allowed")


class TestEndToEnd(unittest.TestCase):
    """Test 4: end-to-end (requires ANTHROPIC_API_KEY + Abaqus)."""

    @unittest.skipUnless(
        os.environ.get("KIMI_API_KEY") and "--e2e" in sys.argv,
        "Skipped: set KIMI_API_KEY and pass --e2e to run"
    )
    def test_cantilever_beam(self):
        from agent import AbaqusAgent

        agent = AbaqusAgent()
        response = agent.chat(
            "分析一个 200x20x20mm 钢制悬臂梁，左端固定，右端面压力 10MPa"
        )
        print(f"\nAgent response:\n{response}")

        self.assertIn("分析完成", response)
        self.assertIn("Mises", response)
        self.assertIn("Displacement", response)
        print("[PASS] End-to-end cantilever beam analysis succeeded")


if __name__ == "__main__":
    # Remove --e2e from argv so unittest doesn't choke on it
    if "--e2e" in sys.argv:
        sys.argv.remove("--e2e")
        # Re-add as environment flag for the skip decorator
        os.environ["_RUN_E2E"] = "1"

    unittest.main(verbosity=2)
