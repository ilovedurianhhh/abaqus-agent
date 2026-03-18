"""End-to-end test: cantilever beam bending analysis.

A 200x20x20 mm steel beam, fixed at x=0, pressure on face at x=200.
Validates script generation, Abaqus execution, and ODB result reading.

Usage:
    python tests/test_cantilever.py              # full run (requires Abaqus)
    python tests/test_cantilever.py --preview    # only print generated script
"""

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abaqus_api import AbaqusModel


def build_model():
    """Build the cantilever beam model and return the AbaqusModel instance."""
    m = AbaqusModel("BeamBending")

    # --- Geometry ---
    m.part.create_sketch("beam_sketch", sheet_size=400.0)
    m.part.rectangle(p1=(0, 0), p2=(200.0, 20.0))
    m.part.extrude_solid("Beam", depth=20.0)

    # --- Sets & Surfaces ---
    # Fixed end: face at x=0
    m.part.create_face_set("Beam", "FixedEnd",
        xmin=-0.1, ymin=-0.1, zmin=-0.1, xmax=0.1, ymax=20.1, zmax=20.1)
    # Load surface: face at x=200
    m.part.create_surface("Beam", "LoadSurf",
        xmin=199.9, ymin=-0.1, zmin=-0.1, xmax=200.1, ymax=20.1, zmax=20.1)

    # --- Material ---
    m.material.create_elastic("Steel", E=210000.0, nu=0.3)
    m.material.create_solid_section("BeamSec", material="Steel")
    m.material.assign_section("Beam", "BeamSec")

    # --- Assembly ---
    m.assembly.create_instance("Beam")

    # --- Step ---
    m.step.create_static("Loading")

    # --- Boundary Conditions & Loads ---
    m.load.fix("Fixed", instance="Beam-1", set_name="FixedEnd")
    m.load.pressure("Tip", step="Loading", instance="Beam-1",
                     surface="LoadSurf", magnitude=10.0)

    # --- Mesh ---
    m.mesh.seed_part("Beam", size=5.0)
    m.mesh.set_element_type("Beam", "C3D8R")
    m.mesh.generate("Beam")

    return m


def test_preview():
    """Test 1: Verify the generated script is syntactically reasonable."""
    m = build_model()
    script = m.preview()

    print("=" * 60)
    print("GENERATED ABAQUS SCRIPT")
    print("=" * 60)
    print(script)
    print("=" * 60)

    # Basic checks
    assert "from abaqus import *" in script, "Missing abaqus import"
    assert "ConstrainedSketch" in script, "Missing sketch creation"
    assert "BaseSolidExtrude" in script, "Missing extrude"
    assert "HomogeneousSolidSection" in script, "Missing section"
    assert "SectionAssignment" in script, "Missing section assignment"
    assert "Instance" in script, "Missing instance"
    assert "StaticStep" in script, "Missing step"
    assert "EncastreBC" in script, "Missing BC"
    assert "Pressure" in script, "Missing pressure load"
    assert "seedPart" in script, "Missing mesh seed"
    assert "generateMesh" in script, "Missing mesh generation"

    print("\n[PASS] Script generation validated.")
    return script


def test_full_run():
    """Test 2-4: Execute in Abaqus, submit job, read ODB results."""
    m = build_model()

    print("\nSubmitting job to Abaqus...")
    result = m.submit("BeamJob", wait=True, timeout=300)
    print(f"Submit result: {result}")

    if "error" in result and result["error"]:
        print(f"\n[FAIL] Job submission failed: {result.get('message', 'unknown')}")
        if "stdout" in result:
            print(f"STDOUT:\n{result['stdout']}")
        if "stderr" in result:
            print(f"STDERR:\n{result['stderr']}")
        return False

    output_dir = result.get("output_dir", "")
    assert result.get("odb_exists"), "ODB file not created"
    print(f"[PASS] Job completed. Output dir: {output_dir}")

    # --- ODB Post-processing ---
    print("\nReading ODB results...")
    summary = m.odb.max_values("BeamJob.odb", work_dir=output_dir)
    print(f"ODB result: {summary}")

    if "error" in summary and summary["error"]:
        print(f"[FAIL] ODB reading failed: {summary.get('message')}")
        return False

    max_mises = summary["max_mises"]
    max_u = summary["max_displacement"]
    max_rf = summary["max_rf_magnitude"]

    print(f"\n  Max Mises Stress : {max_mises:.2f} MPa")
    print(f"  Max Displacement : {max_u:.4f} mm")
    print(f"  Max Reaction Force: {max_rf:.2f} N")

    # Sanity checks (beam bending - values should be positive and reasonable)
    assert max_mises > 0, "Mises stress should be positive"
    assert max_u > 0, "Displacement should be positive"
    assert max_rf > 0, "Reaction force should be positive"

    print(f"\n[PASS] All results validated.")
    print(f"Output files saved to: {output_dir}")
    return True


if __name__ == "__main__":
    if "--preview" in sys.argv:
        test_preview()
    else:
        test_preview()
        print("\n" + "-" * 60)
        test_full_run()
