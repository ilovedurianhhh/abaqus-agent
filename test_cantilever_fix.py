"""Test script to verify the cantilever beam analysis works correctly."""

from abaqus_api import AbaqusModel

# Create model
m = AbaqusModel("CantileverBeam")

# Geometry: 200x20 sketch, extrude 10mm in Z
m.part.create_sketch("beam_sketch", sheet_size=400.0)
m.part.rectangle(p1=(0, 0), p2=(200.0, 20.0))
m.part.extrude_solid("Beam", depth=10.0)

# Face sets using bounding box (±0.1 tolerance)
# Fixed end at x=0
m.part.create_face_set("Beam", "FixedEnd",
    xmin=-0.1, ymin=-0.1, zmin=-0.1, xmax=0.1, ymax=20.1, zmax=10.1)

# Material: Aluminum
# E=70GPa=70000MPa, nu=0.33, density=2700kg/m³=2.7e-9 tonne/mm³
m.material.create_elastic("Aluminum", E=70000.0, nu=0.33)
m.material.create_density("Aluminum", density=2.7e-9)
m.material.create_solid_section("BeamSec", material="Aluminum")
m.material.assign_section("Beam", "BeamSec")

# Assembly
m.assembly.create_instance("Beam")

# Step: Static with geometric nonlinearity
m.step.create_static("AnalysisStep", nlgeom=True)

# Boundary condition: Fix one end
m.load.fix("Fixed", instance="Beam-1", set_name="FixedEnd")

# Mesh FIRST — must be done before concentrated_force_at_point
m.mesh.seed_part("Beam", size=2.0)
m.mesh.set_element_type("Beam", "C3D8R")
m.mesh.generate("Beam")

# Load: Concentrated force at the top edge midpoint of the free end
# Free end is at x=200, top edge is at y=20, midpoint in Z is at z=5
# Force is -500N in Y direction (downward)
m.load.concentrated_force_at_point("TipLoad", step="AnalysisStep",
    instance="Beam-1", point=(200.0, 20.0, 5.0), tolerance=3.0, cf2=-500.0)

# Submit and read results
result = m.submit("CantileverBeamJob", wait=True)
odb_summary = m.odb.max_values("CantileverBeamJob.odb", work_dir=result["output_dir"])

print("Analysis completed successfully!")
print(f"Max von Mises Stress: {odb_summary.get('max_mises', 0):.2f} MPa")
print(f"Max Displacement: {odb_summary.get('max_displacement', 0):.4f} mm")
print(f"Max Reaction Force: {odb_summary.get('max_rf_magnitude', 0):.2f} N")
