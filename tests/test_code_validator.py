"""Tests for API signature validator — verify it catches common LLM mistakes."""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.code_validator import validate_api_calls, _build_api_signatures, _extract_api_calls


class TestSignatureIntrospection(unittest.TestCase):
    """Verify we can introspect all API signatures."""

    def test_signatures_loaded(self):
        sigs = _build_api_signatures()
        # Should have methods from all builders
        self.assertIn(("part", "create_sketch"), sigs)
        self.assertIn(("part", "extrude_solid"), sigs)
        self.assertIn(("material", "create_elastic"), sigs)
        self.assertIn(("load", "fix"), sigs)
        self.assertIn(("load", "pressure"), sigs)
        self.assertIn(("step", "create_static"), sigs)
        self.assertIn(("mesh", "generate"), sigs)
        print(f"[PASS] Loaded {len(sigs)} API signatures")

    def test_signature_details(self):
        sigs = _build_api_signatures()
        # create_elastic should require name, E, nu
        sig = sigs[("material", "create_elastic")]
        self.assertIn("name", sig["required"])
        self.assertIn("E", sig["required"])
        self.assertIn("nu", sig["required"])
        print("[PASS] Signature details correct")


class TestCallExtraction(unittest.TestCase):
    """Verify we extract m.xxx.yyy() calls correctly."""

    def test_extract_builder_calls(self):
        code = '''
m.part.create_sketch("s", sheet_size=400.0)
m.part.rectangle(p1=(0,0), p2=(100,100))
m.material.create_elastic("Steel", E=210000.0, nu=0.3)
m.submit("Job", wait=True)
'''
        calls = _extract_api_calls(code)
        builders = [(b, meth) for b, meth, _, _, _ in calls]
        self.assertIn(("part", "create_sketch"), builders)
        self.assertIn(("part", "rectangle"), builders)
        self.assertIn(("material", "create_elastic"), builders)
        self.assertIn(("model", "submit"), builders)
        print(f"[PASS] Extracted {len(calls)} API calls")

    def test_ignore_non_m_calls(self):
        code = '''
x = some_func()
result = other.method()
m.part.create_sketch("s")
'''
        calls = _extract_api_calls(code)
        self.assertEqual(len(calls), 1)
        print("[PASS] Non-m calls ignored")


class TestValidCorrectCode(unittest.TestCase):
    """Verify correct code passes validation."""

    def test_valid_static_analysis(self):
        code = '''
from abaqus_api import AbaqusModel
m = AbaqusModel("Test")
m.part.create_sketch("s", sheet_size=400.0)
m.part.rectangle(p1=(0,0), p2=(200,20))
m.part.extrude_solid("Beam", depth=20.0)
m.part.create_face_set("Beam", "Fix", xmin=-0.1, ymin=-0.1, zmin=-0.1, xmax=0.1, ymax=20.1, zmax=20.1)
m.part.create_surface("Beam", "Load", xmin=199.9, ymin=-0.1, zmin=-0.1, xmax=200.1, ymax=20.1, zmax=20.1)
m.material.create_elastic("Steel", E=210000.0, nu=0.3)
m.material.create_solid_section("Sec", material="Steel")
m.material.assign_section("Beam", "Sec")
m.assembly.create_instance("Beam")
m.step.create_static("Load")
m.load.fix("Fix", instance="Beam-1", set_name="Fix")
m.load.pressure("P", step="Load", instance="Beam-1", surface="Load", magnitude=10.0)
m.mesh.seed_part("Beam", size=5.0)
m.mesh.set_element_type("Beam", "C3D8R")
m.mesh.generate("Beam")
result = m.submit("Job", wait=True)
odb_summary = m.odb.max_values("Job.odb", work_dir=result["output_dir"])
'''
        errors = validate_api_calls(code)
        self.assertEqual(errors, [], f"Unexpected errors: {errors}")
        print("[PASS] Valid static analysis code passes")

    def test_valid_frequency_analysis(self):
        code = '''
from abaqus_api import AbaqusModel
m = AbaqusModel("Modal")
m.part.create_sketch("s", sheet_size=400.0)
m.part.rectangle(p1=(0,0), p2=(300,20))
m.part.extrude_solid("Beam", depth=10.0)
m.part.create_face_set("Beam", "Fix", xmin=-0.1, ymin=-0.1, zmin=-0.1, xmax=0.1, ymax=20.1, zmax=10.1)
m.material.create_elastic("Steel", E=210000.0, nu=0.3)
m.material.create_density("Steel", density=7.85e-9)
m.material.create_solid_section("Sec", material="Steel")
m.material.assign_section("Beam", "Sec")
m.assembly.create_instance("Beam")
m.step.create_frequency("Modal", num_eigen=5)
m.load.fix("Fix", instance="Beam-1", set_name="Fix")
m.mesh.seed_part("Beam", size=5.0)
m.mesh.set_element_type("Beam", "C3D8R")
m.mesh.generate("Beam")
result = m.submit("Job", wait=True)
'''
        errors = validate_api_calls(code)
        self.assertEqual(errors, [], f"Unexpected errors: {errors}")
        print("[PASS] Valid frequency analysis code passes")


class TestCatchErrors(unittest.TestCase):
    """Verify common LLM mistakes are caught."""

    def test_catch_nonexistent_method(self):
        """LLM invents a method that doesn't exist."""
        code = '''
m.part.create_cylinder("Cyl", radius=10, height=50)
'''
        errors = validate_api_calls(code)
        self.assertTrue(any("create_cylinder" in e and "不存在" in e for e in errors),
                        f"Should catch nonexistent method. Got: {errors}")
        print("[PASS] Catches nonexistent method: create_cylinder")

    def test_catch_pressure_with_set_name(self):
        """LLM passes set_name to pressure() instead of surface."""
        code = '''
m.load.pressure("P", step="Load", instance="Beam-1", set_name="Face1", magnitude=10.0)
'''
        errors = validate_api_calls(code)
        self.assertTrue(any("pressure" in e and "surface" in e for e in errors),
                        f"Should catch set_name in pressure. Got: {errors}")
        print("[PASS] Catches pressure() with set_name instead of surface")

    def test_catch_fix_with_surface(self):
        """LLM passes surface to fix() instead of set_name."""
        code = '''
m.load.fix("Fix", instance="Beam-1", surface="Surf1")
'''
        errors = validate_api_calls(code)
        self.assertTrue(any("fix" in e for e in errors),
                        f"Should catch surface in fix. Got: {errors}")
        print("[PASS] Catches fix() with surface instead of set_name")

    def test_catch_unknown_parameter(self):
        """LLM passes a parameter that doesn't exist."""
        code = '''
m.material.create_elastic("Steel", E=210000.0, nu=0.3, density=7.85e-9)
'''
        errors = validate_api_calls(code)
        self.assertTrue(any("density" in e for e in errors),
                        f"Should catch unknown param. Got: {errors}")
        print("[PASS] Catches unknown parameter: density in create_elastic")

    def test_catch_missing_required_param(self):
        """LLM forgets a required parameter."""
        code = '''
m.material.create_elastic("Steel", E=210000.0)
'''
        errors = validate_api_calls(code)
        self.assertTrue(any("nu" in e for e in errors),
                        f"Should catch missing nu. Got: {errors}")
        print("[PASS] Catches missing required parameter: nu")

    def test_catch_wrong_step_method(self):
        """LLM invents a step type."""
        code = '''
m.step.create_heat_transfer("HeatStep", time_period=1.0)
'''
        errors = validate_api_calls(code)
        self.assertTrue(any("create_heat_transfer" in e and "不存在" in e for e in errors),
                        f"Should catch nonexistent step. Got: {errors}")
        print("[PASS] Catches nonexistent step type: create_heat_transfer")

    def test_catch_wrong_mesh_element(self):
        """LLM invents a mesh method."""
        code = '''
m.mesh.refine_region("Beam", region="Corner", size=1.0)
'''
        errors = validate_api_calls(code)
        self.assertTrue(any("refine_region" in e for e in errors),
                        f"Should catch nonexistent mesh method. Got: {errors}")
        print("[PASS] Catches nonexistent mesh method: refine_region")


class TestEdgeCases(unittest.TestCase):
    """Edge cases and robustness."""

    def test_empty_code(self):
        errors = validate_api_calls("")
        self.assertEqual(errors, [])
        print("[PASS] Empty code returns no errors")

    def test_syntax_error_code(self):
        errors = validate_api_calls("def foo(\n")
        self.assertEqual(errors, [])  # Can't parse, returns empty
        print("[PASS] Syntax error code returns no errors (handled by safety validator)")

    def test_no_api_calls(self):
        errors = validate_api_calls("x = 1 + 1\nprint(x)")
        self.assertEqual(errors, [])
        print("[PASS] Code without API calls passes")


if __name__ == "__main__":
    unittest.main(verbosity=2)
