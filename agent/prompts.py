"""System prompt construction with API introspection and few-shot examples."""

import inspect
import importlib
import functools


# Builder classes to introspect (module_name, class_name, accessor)
_BUILDERS = [
    ("abaqus_api.part", "PartBuilder", "m.part"),
    ("abaqus_api.material", "MaterialBuilder", "m.material"),
    ("abaqus_api.assembly", "AssemblyBuilder", "m.assembly"),
    ("abaqus_api.step", "StepBuilder", "m.step"),
    ("abaqus_api.load", "LoadBuilder", "m.load"),
    ("abaqus_api.mesh", "MeshBuilder", "m.mesh"),
]

# OdbReader is accessed differently
_ODB_MODULE = ("abaqus_api.odb", "OdbReader", "m.odb")

# AbaqusModel top-level methods
_MODEL_MODULE = ("abaqus_api.model", "AbaqusModel", "m")


def _introspect_class(module_name, class_name, accessor):
    """Extract public method signatures from a builder class."""
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    lines = []
    for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        if name.startswith("_"):
            continue
        sig = inspect.signature(method)
        # Remove 'self' parameter
        params = [p for p in sig.parameters.values() if p.name != "self"]
        param_strs = []
        for p in params:
            if p.default is inspect.Parameter.empty:
                param_strs.append(p.name)
            else:
                param_strs.append(f"{p.name}={p.default!r}")
        doc = (method.__doc__ or "").strip().split("\n")[0]
        lines.append(f"  {accessor}.{name}({', '.join(param_strs)})")
        if doc:
            lines.append(f"      # {doc}")
    return lines


@functools.lru_cache(maxsize=1)
def build_api_reference():
    """Generate API reference from builder class introspection (cached)."""
    sections = []

    # AbaqusModel top-level
    sections.append("## AbaqusModel (from abaqus_api import AbaqusModel)")
    sections.append("  m = AbaqusModel(model_name)")
    sections.append("  m.preview()          # Return accumulated script as string")
    sections.append("  m.reset()            # Clear code buffer")
    sections.append("  m.submit(job_name, wait=True, description='', num_cpus=1, memory=90, timeout=600)")
    sections.append("      # Submit job and return dict with output_dir, odb_exists, etc.")
    sections.append("  m.odb.max_values(odb_path, work_dir=None)")
    sections.append("      # Read max Mises stress, displacement, reaction force from ODB")
    sections.append("")

    # Builder sub-objects
    for module_name, class_name, accessor in _BUILDERS:
        sections.append(f"## {class_name} ({accessor})")
        sections.extend(_introspect_class(module_name, class_name, accessor))
        sections.append("")

    return "\n".join(sections)


ROLE_PROMPT = """\
You are an Abaqus structural analysis assistant.
You receive natural language descriptions of structural analysis problems and generate complete Python scripts using the `abaqus_api` library.

IMPORTANT RULES:
1. Always import with: from abaqus_api import AbaqusModel
2. Create model: m = AbaqusModel("ModelName")
3. Use bounding box selection for faces/sets — add a small tolerance (±0.1) around target coordinates
4. Instance name is "<PartName>-1" by default (Abaqus convention)
5. ALWAYS create an analysis step (m.step.create_static/create_frequency/create_dynamic_explicit)
   - Static analysis: m.step.create_static("StepName", nlgeom=True/False)
   - Modal/frequency: m.step.create_frequency("ModalStep", num_eigen=N)
   - Dynamic explicit: m.step.create_dynamic_explicit("ImpactStep", time_period=T)
6. Always call m.submit(job_name, wait=True) to run the analysis
7. After submit, call m.odb.max_values("JobName.odb", work_dir=result["output_dir"]) to read results
8. Store the ODB summary in a variable called `odb_summary` so the framework can capture it

COORDINATE CONVENTION — you MUST follow this consistently:
- Sketch is drawn in the X-Y plane, extruded along Z
- X axis = length direction (长度方向, longest dimension)
- Y axis = height direction (高度方向, vertical). "向上" = +Y, "向下" = -Y
- Z axis = width direction (宽度方向, extrusion depth)
- For a part described as "长L × 宽W × 高H":
  * rectangle(p1=(0,0), p2=(L, H)), extrude_solid(depth=W)
  * X ranges 0~L, Y ranges 0~H, Z ranges 0~W
- "垂直向下的力" = cf2 with negative value (e.g. cf2=-500.0)
- "水平力沿长度方向" = cf1, "沿宽度方向" = cf3

CRITICAL — set type selection:
- m.load.fix() uses FACE SET → create with m.part.create_face_set()
- m.load.pressure() uses SURFACE → create with m.part.create_surface()
- For CONCENTRATED FORCES:
  * ALWAYS use m.load.concentrated_force_at_point(name, step, instance, point=(x,y,z), cf1/cf2/cf3=value)
  * It takes a coordinate tuple (x, y, z) and automatically finds the nearest mesh node
  * Call it AFTER m.mesh.generate()
  * NEVER use m.load.concentrated_force(set_name=...) — it requires manual node set creation and will fail
  * Example: m.load.concentrated_force_at_point("TipLoad", "Step1", "Beam-1", point=(200.0, 20.0, 5.0), cf2=-500.0)
- NEVER pass a face set name to pressure(); NEVER pass a surface name to fix()
- This is the #1 most common error. Double-check every load/BC call.

HYBRID API USAGE:
- PREFER using the simplified API (m.part, m.load, m.mesh, etc.) whenever possible
- If the simplified API does not have the needed function, use native Abaqus Python commands
  from the "Abaqus Native API Reference" section below
- When using native commands, wrap them with m._buf.emit("...") to add to the code buffer
- NEVER mix: either use simplified API for a task, or use native commands, not both for the same operation

OUTPUT FORMAT — you MUST respond with exactly this structure:
<plan>
Brief analysis plan (geometry, material, BCs, loads, mesh, analysis step type)
</plan>
<code>
from abaqus_api import AbaqusModel
m = AbaqusModel("...")
...
m.step.create_static("StepName", nlgeom=...)  # REQUIRED: Always create a step
...
result = m.submit("JobName", wait=True)
odb_summary = m.odb.max_values("JobName.odb", work_dir=result["output_dir"])
</code>
"""


# ---------------------------------------------------------------------------
# Few-shot examples dictionary — keyed by analysis type
# ---------------------------------------------------------------------------

EXAMPLES = {
    "static_pressure": {
        "keywords": ["压力", "均布", "pressure", "面力"],
        "example": """\
## Example: Cantilever beam with pressure load

User: 分析一个 200x20x20mm 的钢制悬臂梁，左端固定，右端面施加 10MPa 压力

<plan>
- 几何：200×20×20 mm 长方体
- 材料：钢 (E=210000 MPa, ν=0.3)
- 边界条件：左端面 (x=0) 全固定
- 载荷：右端面 (x=200) 压力 10 MPa
- 网格：C3D8R, 种子尺寸 5mm
</plan>
<code>
from abaqus_api import AbaqusModel

m = AbaqusModel("BeamBending")

# Geometry: 200x20 sketch, extrude 20mm in Z
m.part.create_sketch("beam_sketch", sheet_size=400.0)
m.part.rectangle(p1=(0, 0), p2=(200.0, 20.0))
m.part.extrude_solid("Beam", depth=20.0)

# Face sets using bounding box (±0.1 tolerance)
m.part.create_face_set("Beam", "FixedEnd",
    xmin=-0.1, ymin=-0.1, zmin=-0.1, xmax=0.1, ymax=20.1, zmax=20.1)
m.part.create_surface("Beam", "LoadSurf",
    xmin=199.9, ymin=-0.1, zmin=-0.1, xmax=200.1, ymax=20.1, zmax=20.1)

# Material
m.material.create_elastic("Steel", E=210000.0, nu=0.3)
m.material.create_solid_section("BeamSec", material="Steel")
m.material.assign_section("Beam", "BeamSec")

# Assembly
m.assembly.create_instance("Beam")

# Step
m.step.create_static("Loading")

# BCs and loads
m.load.fix("Fixed", instance="Beam-1", set_name="FixedEnd")
m.load.pressure("Tip", step="Loading", instance="Beam-1",
                 surface="LoadSurf", magnitude=10.0)

# Mesh
m.mesh.seed_part("Beam", size=5.0)
m.mesh.set_element_type("Beam", "C3D8R")
m.mesh.generate("Beam")

# Submit and read results
result = m.submit("BeamJob", wait=True)
odb_summary = m.odb.max_values("BeamJob.odb", work_dir=result["output_dir"])
</code>""",
    },

    "static_force": {
        "keywords": ["集中力", "点力", "concentrated", "force"],
        "example": """\
## Example: Plate with concentrated force

User: 一块 100x100x5mm 的铝板，四边固定，中心施加 500N 向下的集中力

<plan>
- 几何：100×100×5 mm 板
- 材料：铝合金 (E=70000 MPa, ν=0.33)
- 边界条件：四边 (x=0, x=100, y=0, y=100 的面) 全固定
- 载荷：中心点 (50, 50, 2.5) 集中力 CF2=-500 N
- 网格：C3D8R, 种子尺寸 5mm
</plan>
<code>
from abaqus_api import AbaqusModel

m = AbaqusModel("PlateCenter")

# Geometry: 100x100 sketch, extrude 5mm
m.part.create_sketch("plate_sketch", sheet_size=200.0)
m.part.rectangle(p1=(0, 0), p2=(100.0, 100.0))
m.part.extrude_solid("Plate", depth=5.0)

# Four edge face sets (±0.1 tolerance)
m.part.create_face_set("Plate", "EdgeX0",
    xmin=-0.1, ymin=-0.1, zmin=-0.1, xmax=0.1, ymax=100.1, zmax=5.1)
m.part.create_face_set("Plate", "EdgeX100",
    xmin=99.9, ymin=-0.1, zmin=-0.1, xmax=100.1, ymax=100.1, zmax=5.1)
m.part.create_face_set("Plate", "EdgeY0",
    xmin=-0.1, ymin=-0.1, zmin=-0.1, xmax=100.1, ymax=0.1, zmax=5.1)
m.part.create_face_set("Plate", "EdgeY100",
    xmin=-0.1, ymin=99.9, zmin=-0.1, xmax=100.1, ymax=100.1, zmax=5.1)

# Material
m.material.create_elastic("Aluminum", E=70000.0, nu=0.33)
m.material.create_solid_section("PlateSec", material="Aluminum")
m.material.assign_section("Plate", "PlateSec")

# Assembly
m.assembly.create_instance("Plate")

# Step
m.step.create_static("Loading")

# BCs — fix all four edges
m.load.fix("FixX0", instance="Plate-1", set_name="EdgeX0")
m.load.fix("FixX100", instance="Plate-1", set_name="EdgeX100")
m.load.fix("FixY0", instance="Plate-1", set_name="EdgeY0")
m.load.fix("FixY100", instance="Plate-1", set_name="EdgeY100")

# Mesh FIRST — must be done before concentrated_force_at_point
m.mesh.seed_part("Plate", size=5.0)
m.mesh.set_element_type("Plate", "C3D8R")
m.mesh.generate("Plate")

# Load — concentrated force at center point (50, 50, 2.5), AFTER meshing
# IMPORTANT: Use concentrated_force_at_point with point=(x,y,z), NOT concentrated_force with set_name
m.load.concentrated_force_at_point("CenterForce", step="Loading",
    instance="Plate-1", point=(50.0, 50.0, 2.5), tolerance=5.0, cf2=-500.0)

# Submit and read results
result = m.submit("PlateJob", wait=True)
odb_summary = m.odb.max_values("PlateJob.odb", work_dir=result["output_dir"])
</code>""",
    },

    "cantilever": {
        "keywords": ["悬臂", "悬臂梁", "cantilever"],
        "example": """\
## Example: Cantilever beam with tip load

User: 设计一个长200mm、宽20mm、高10mm的铝合金悬臂梁，一端固定，另一端顶边中点施加500N向下集中力

<plan>
- 几何：200×20×10 mm 长方体 (X=长200, Y=高10, Z=宽20)
- 材料：铝合金 (E=70000 MPa, ν=0.33)
- 边界条件：左端面 (x=0) 全固定
- 载荷：右端顶边中点 (200, 10, 10) 集中力 CF2=-500 N (向下=-Y)
- 网格：C3D8R, 种子尺寸 2mm
</plan>
<code>
from abaqus_api import AbaqusModel

m = AbaqusModel("CantileverBeam")

# Geometry: sketch in X-Y plane (长×高), extrude along Z (宽)
# X=长=200, Y=高=10, Z=宽=20
m.part.create_sketch("beam_sketch", sheet_size=400.0)
m.part.rectangle(p1=(0, 0), p2=(200.0, 10.0))
m.part.extrude_solid("Beam", depth=20.0)

# Face set for fixed end (x=0 face, ±0.1 tolerance)
m.part.create_face_set("Beam", "FixedEnd",
    xmin=-0.1, ymin=-0.1, zmin=-0.1, xmax=0.1, ymax=10.1, zmax=20.1)

# Material
m.material.create_elastic("Aluminum", E=70000.0, nu=0.33)
m.material.create_solid_section("BeamSec", material="Aluminum")
m.material.assign_section("Beam", "BeamSec")

# Assembly
m.assembly.create_instance("Beam")

# Step
m.step.create_static("Loading")

# BC — fix left end
m.load.fix("Fixed", instance="Beam-1", set_name="FixedEnd")

# Mesh FIRST — must be done before concentrated_force_at_point
m.mesh.seed_part("Beam", size=2.0)
m.mesh.set_element_type("Beam", "C3D8R")
m.mesh.generate("Beam")

# Load — tip top-edge midpoint: x=200(tip), y=10(top), z=10(mid-width)
# "向下" = -Y direction = cf2=-500
m.load.concentrated_force_at_point("TipLoad", step="Loading",
    instance="Beam-1", point=(200.0, 10.0, 10.0), tolerance=3.0, cf2=-500.0)

# Submit and read results
result = m.submit("CantileverJob", wait=True)
odb_summary = m.odb.max_values("CantileverJob.odb", work_dir=result["output_dir"])
</code>""",
    },

    "gravity": {
        "keywords": ["重力", "自重", "gravity", "重量"],
        "example": """\
## Example: Steel beam under gravity (self-weight)

User: 一根 500x30x30mm 的钢梁，两端简支，分析自重作用下的变形

<plan>
- 几何：500×30×30 mm 长方体
- 材料：钢 (E=210000 MPa, ν=0.3, ρ=7.85e-9 t/mm³)
- 边界条件：左端面 (x=0) 固定，右端面 (x=500) 仅约束 Y 方向
- 载荷：重力加速度 -Y 方向 (9810 mm/s²)
- 网格：C3D8R, 种子尺寸 8mm
</plan>
<code>
from abaqus_api import AbaqusModel

m = AbaqusModel("GravityBeam")

# Geometry
m.part.create_sketch("beam_sketch", sheet_size=600.0)
m.part.rectangle(p1=(0, 0), p2=(500.0, 30.0))
m.part.extrude_solid("Beam", depth=30.0)

# Face sets
m.part.create_face_set("Beam", "LeftEnd",
    xmin=-0.1, ymin=-0.1, zmin=-0.1, xmax=0.1, ymax=30.1, zmax=30.1)
m.part.create_face_set("Beam", "RightEnd",
    xmin=499.9, ymin=-0.1, zmin=-0.1, xmax=500.1, ymax=30.1, zmax=30.1)

# Material with density
m.material.create_elastic("Steel", E=210000.0, nu=0.3)
m.material.create_density("Steel", density=7.85e-9)
m.material.create_solid_section("BeamSec", material="Steel")
m.material.assign_section("Beam", "BeamSec")

# Assembly
m.assembly.create_instance("Beam")

# Step
m.step.create_static("GravityStep")

# BCs
m.load.fix("FixLeft", instance="Beam-1", set_name="LeftEnd")
m.load.displacement_bc("RollerRight", instance="Beam-1",
    set_name="RightEnd", step="Initial", u2=0.0)

# Gravity load
m.load.gravity("Gravity", step="GravityStep", comp2=-9810.0)

# Mesh
m.mesh.seed_part("Beam", size=8.0)
m.mesh.set_element_type("Beam", "C3D8R")
m.mesh.generate("Beam")

# Submit and read results
result = m.submit("GravityJob", wait=True)
odb_summary = m.odb.max_values("GravityJob.odb", work_dir=result["output_dir"])
</code>""",
    },

    "nonlinear": {
        "keywords": ["大变形", "塑性", "屈服", "非线性", "nonlinear", "plastic"],
        "example": """\
## Example: Aluminum plate large-deformation bending

User: 一块 200x50x3mm 的铝板，一端固定，另一端施加 200N 集中力，考虑大变形和塑性

<plan>
- 几何：200×50×3 mm 薄板
- 材料：铝合金 (E=70000 MPa, ν=0.33, 屈服强度 275 MPa, 塑性应变表)
- 边界条件：左端面 (x=0) 全固定
- 载荷：右端中心集中力 CF2=-200 N
- 分析：nlgeom=True 大变形
- 网格：C3D8R, 种子尺寸 5mm
</plan>
<code>
from abaqus_api import AbaqusModel

m = AbaqusModel("NonlinearPlate")

# Geometry
m.part.create_sketch("plate_sketch", sheet_size=400.0)
m.part.rectangle(p1=(0, 0), p2=(200.0, 50.0))
m.part.extrude_solid("Plate", depth=3.0)

# Face sets
m.part.create_face_set("Plate", "FixedEnd",
    xmin=-0.1, ymin=-0.1, zmin=-0.1, xmax=0.1, ymax=50.1, zmax=3.1)

# Material with plasticity
m.material.create_elastic("Aluminum", E=70000.0, nu=0.33)
m.material.create_plastic("Aluminum", table=[(275.0, 0.0), (310.0, 0.05), (350.0, 0.1)])
m.material.create_solid_section("PlateSec", material="Aluminum")
m.material.assign_section("Plate", "PlateSec")

# Assembly
m.assembly.create_instance("Plate")

# Step — large deformation enabled
m.step.create_static("Loading", nlgeom=True)

# BCs
m.load.fix("Fixed", instance="Plate-1", set_name="FixedEnd")

# Mesh FIRST
m.mesh.seed_part("Plate", size=5.0)
m.mesh.set_element_type("Plate", "C3D8R")
m.mesh.generate("Plate")

# Load — concentrated force at right-end center (200, 25, 1.5), AFTER meshing
m.load.concentrated_force_at_point("Tip", step="Loading",
    instance="Plate-1", point=(200.0, 25.0, 1.5), tolerance=5.0, cf2=-200.0)

# Submit and read results
result = m.submit("NonlinearJob", wait=True)
odb_summary = m.odb.max_values("NonlinearJob.odb", work_dir=result["output_dir"])
</code>""",
    },

    "frequency": {
        "keywords": ["模态", "频率", "振动", "固有", "frequency", "modal", "eigen"],
        "example": """\
## Example: Cantilever beam modal analysis

User: 分析一个 300x20x10mm 钢制悬臂梁的前 5 阶固有频率

<plan>
- 几何：300×20×10 mm 长方体
- 材料：钢 (E=210000 MPa, ν=0.3, ρ=7.85e-9 t/mm³) — 模态分析必须有密度
- 边界条件：左端面 (x=0) 全固定
- 载荷：无（模态分析不施加外力）
- 分析步：Frequency, 提取 5 阶模态
- 网格：C3D8R, 种子尺寸 5mm
</plan>
<code>
from abaqus_api import AbaqusModel

m = AbaqusModel("ModalBeam")

# Geometry
m.part.create_sketch("beam_sketch", sheet_size=400.0)
m.part.rectangle(p1=(0, 0), p2=(300.0, 20.0))
m.part.extrude_solid("Beam", depth=10.0)

# Face set for fixed end
m.part.create_face_set("Beam", "FixedEnd",
    xmin=-0.1, ymin=-0.1, zmin=-0.1, xmax=0.1, ymax=20.1, zmax=10.1)

# Material — density is required for frequency analysis
m.material.create_elastic("Steel", E=210000.0, nu=0.3)
m.material.create_density("Steel", density=7.85e-9)
m.material.create_solid_section("BeamSec", material="Steel")
m.material.assign_section("Beam", "BeamSec")

# Assembly
m.assembly.create_instance("Beam")

# Frequency step — extract 5 eigenvalues
m.step.create_frequency("ModalStep", num_eigen=5)

# BC — fixed left end (no loads needed for modal analysis)
m.load.fix("Fixed", instance="Beam-1", set_name="FixedEnd")

# Mesh
m.mesh.seed_part("Beam", size=5.0)
m.mesh.set_element_type("Beam", "C3D8R")
m.mesh.generate("Beam")

# Submit and read results
result = m.submit("ModalJob", wait=True)
odb_summary = m.odb.max_values("ModalJob.odb", work_dir=result["output_dir"])
</code>""",
    },

    "dynamic": {
        "keywords": ["冲击", "动力", "显式", "dynamic", "explicit", "impact"],
        "example": """\
## Example: Bar under impact loading (dynamic explicit)

User: 一根 100x10x10mm 的钢杆，左端固定，右端面受 50MPa 冲击压力，持续 0.001s

<plan>
- 几何：100×10×10 mm 杆件
- 材料：钢 (E=210000 MPa, ν=0.3, ρ=7.85e-9 t/mm³) — 显式分析必须有密度
- 边界条件：左端面 (x=0) 全固定
- 载荷：右端面 (x=100) 压力 50 MPa
- 分析步：Dynamic/Explicit, 时长 0.001s
- 网格：C3D8R, 种子尺寸 3mm
</plan>
<code>
from abaqus_api import AbaqusModel

m = AbaqusModel("ImpactBar")

# Geometry
m.part.create_sketch("bar_sketch", sheet_size=200.0)
m.part.rectangle(p1=(0, 0), p2=(100.0, 10.0))
m.part.extrude_solid("Bar", depth=10.0)

# Face sets
m.part.create_face_set("Bar", "FixedEnd",
    xmin=-0.1, ymin=-0.1, zmin=-0.1, xmax=0.1, ymax=10.1, zmax=10.1)
m.part.create_surface("Bar", "ImpactSurf",
    xmin=99.9, ymin=-0.1, zmin=-0.1, xmax=100.1, ymax=10.1, zmax=10.1)

# Material — density required for explicit dynamics
m.material.create_elastic("Steel", E=210000.0, nu=0.3)
m.material.create_density("Steel", density=7.85e-9)
m.material.create_solid_section("BarSec", material="Steel")
m.material.assign_section("Bar", "BarSec")

# Assembly
m.assembly.create_instance("Bar")

# Dynamic explicit step
m.step.create_dynamic_explicit("Impact", time_period=0.001)

# BCs and loads
m.load.fix("Fixed", instance="Bar-1", set_name="FixedEnd")
m.load.pressure("ImpactLoad", step="Impact", instance="Bar-1",
                 surface="ImpactSurf", magnitude=50.0)

# Mesh
m.mesh.seed_part("Bar", size=3.0)
m.mesh.set_element_type("Bar", "C3D8R")
m.mesh.generate("Bar")

# Submit and read results
result = m.submit("ImpactJob", wait=True)
odb_summary = m.odb.max_values("ImpactJob.odb", work_dir=result["output_dir"])
</code>""",
    },

    "shell": {
        "keywords": ["壳", "薄板", "薄壁", "shell", "圆筒", "筒体"],
        "example": """\
## Example: Thin-walled cylinder under external pressure (shell)

User: 一个半径 50mm、高 200mm、壁厚 2mm 的薄壁圆筒，底部固定，外表面受 1MPa 压力

<plan>
- 几何：矩形截面 (高200 × 弧长π×50) 拉伸为壳体，或用矩形草图拉伸壳
- 简化：用 200×100 矩形壳代表展开的筒壁
- 材料：钢 (E=210000 MPa, ν=0.3)
- 截面：壳截面, 壁厚 2mm
- 边界条件：底边 (y=0) 全固定
- 载荷：整个壳面压力 1 MPa
- 网格：S4R 壳单元, 种子尺寸 5mm
</plan>
<code>
from abaqus_api import AbaqusModel

m = AbaqusModel("ShellCylinder")

# Geometry — shell extrusion
m.part.create_sketch("shell_sketch", sheet_size=400.0)
m.part.rectangle(p1=(0, 0), p2=(200.0, 100.0))
m.part.extrude_shell("Panel", depth=2.0)

# Face sets
m.part.create_face_set("Panel", "BottomEdge",
    xmin=-0.1, ymin=-0.1, zmin=-0.1, xmax=200.1, ymax=0.1, zmax=2.1)
m.part.create_surface("Panel", "OuterSurf",
    xmin=-0.1, ymin=-0.1, zmin=-0.1, xmax=200.1, ymax=100.1, zmax=2.1)

# Material and shell section
m.material.create_elastic("Steel", E=210000.0, nu=0.3)
m.material.create_shell_section("ShellSec", material="Steel", thickness=2.0)
m.material.assign_section("Panel", "ShellSec")

# Assembly
m.assembly.create_instance("Panel")

# Step
m.step.create_static("Loading")

# BCs and loads
m.load.fix("FixBottom", instance="Panel-1", set_name="BottomEdge")
m.load.pressure("ExtPressure", step="Loading", instance="Panel-1",
                 surface="OuterSurf", magnitude=1.0)

# Mesh
m.mesh.seed_part("Panel", size=5.0)
m.mesh.set_element_type("Panel", "S4R")
m.mesh.generate("Panel")

# Submit and read results
result = m.submit("ShellJob", wait=True)
odb_summary = m.odb.max_values("ShellJob.odb", work_dir=result["output_dir"])
</code>""",
    },

    "displacement": {
        "keywords": ["位移约束", "强制位移", "对称", "位移控制", "displacement"],
        "example": """\
## Example: Tensile bar with displacement control

User: 一根 100x10x10mm 的钢杆，左端固定，右端施加 1mm 拉伸位移

<plan>
- 几何：100×10×10 mm 杆件
- 材料：钢 (E=210000 MPa, ν=0.3)
- 边界条件：左端面 (x=0) 全固定；右端面 (x=100) 施加 u1=1.0 mm
- 载荷：无外力，通过位移控制
- 网格：C3D8R, 种子尺寸 3mm
</plan>
<code>
from abaqus_api import AbaqusModel

m = AbaqusModel("TensileBar")

# Geometry
m.part.create_sketch("bar_sketch", sheet_size=200.0)
m.part.rectangle(p1=(0, 0), p2=(100.0, 10.0))
m.part.extrude_solid("Bar", depth=10.0)

# Face sets
m.part.create_face_set("Bar", "FixedEnd",
    xmin=-0.1, ymin=-0.1, zmin=-0.1, xmax=0.1, ymax=10.1, zmax=10.1)
m.part.create_face_set("Bar", "PullEnd",
    xmin=99.9, ymin=-0.1, zmin=-0.1, xmax=100.1, ymax=10.1, zmax=10.1)

# Material
m.material.create_elastic("Steel", E=210000.0, nu=0.3)
m.material.create_solid_section("BarSec", material="Steel")
m.material.assign_section("Bar", "BarSec")

# Assembly
m.assembly.create_instance("Bar")

# Step
m.step.create_static("Pulling")

# BCs — fixed left, displacement-controlled right
m.load.fix("Fixed", instance="Bar-1", set_name="FixedEnd")
m.load.displacement_bc("Pull", instance="Bar-1",
    set_name="PullEnd", step="Pulling", u1=1.0)

# Mesh
m.mesh.seed_part("Bar", size=3.0)
m.mesh.set_element_type("Bar", "C3D8R")
m.mesh.generate("Bar")

# Submit and read results
result = m.submit("TensileJob", wait=True)
odb_summary = m.odb.max_values("TensileJob.odb", work_dir=result["output_dir"])
</code>""",
    },

    "revolve": {
        "keywords": ["旋转", "轴对称", "圆柱", "圆盘", "revolve", "轴", "法兰", "圆形"],
        "example": """\
## Example: Axisymmetric solid cylinder under axial compression

User: 一个半径 25mm、高 100mm 的钢制实心圆柱，底部固定，顶部施加 20MPa 压力

<plan>
- 几何：圆形草图 (半径 25mm) 旋转 360° 生成实心圆柱
- 材料：钢 (E=210000 MPa, ν=0.3)
- 边界条件：底面 (y=0) 全固定
- 载荷：顶面 (y=100) 压力 20 MPa
- 网格：C3D8R, 种子尺寸 5mm
</plan>
<code>
from abaqus_api import AbaqusModel

m = AbaqusModel("CylinderCompress")

# Geometry — circle sketch + revolve to create solid cylinder
m.part.create_sketch("cyl_sketch", sheet_size=200.0)
m.part.rectangle(p1=(0, 0), p2=(25.0, 100.0))
m.part.revolve_solid("Cylinder", angle=360.0)

# Face sets (after revolve: x-z plane is radial, y is axial height)
m.part.create_face_set("Cylinder", "BottomFace",
    xmin=-25.1, ymin=-0.1, zmin=-25.1, xmax=25.1, ymax=0.1, zmax=25.1)
m.part.create_surface("Cylinder", "TopSurf",
    xmin=-25.1, ymin=99.9, zmin=-25.1, xmax=25.1, ymax=100.1, zmax=25.1)

# Material
m.material.create_elastic("Steel", E=210000.0, nu=0.3)
m.material.create_solid_section("CylSec", material="Steel")
m.material.assign_section("Cylinder", "CylSec")

# Assembly
m.assembly.create_instance("Cylinder")

# Step
m.step.create_static("Compression")

# BCs and loads
m.load.fix("FixBottom", instance="Cylinder-1", set_name="BottomFace")
m.load.pressure("TopLoad", step="Compression", instance="Cylinder-1",
                 surface="TopSurf", magnitude=20.0)

# Mesh
m.mesh.seed_part("Cylinder", size=5.0)
m.mesh.set_element_type("Cylinder", "C3D8R")
m.mesh.generate("Cylinder")

# Submit and read results
result = m.submit("CylinderJob", wait=True)
odb_summary = m.odb.max_values("CylinderJob.odb", work_dir=result["output_dir"])
</code>""",
    },

    "assembly": {
        "keywords": ["装配", "多零件", "组装", "接触", "assembly", "translate", "rotate"],
        "example": """\
## Example: Two-part assembly — block on a base plate

User: 一块 200x200x10mm 的钢底板上放一个 50x50x50mm 的铝块，底板底面固定，铝块顶面施加 5MPa 压力

<plan>
- 几何：底板 200×200×10 mm，铝块 50×50×50 mm（两个独立零件）
- 材料：底板钢 (E=210000, ν=0.3)，铝块 (E=70000, ν=0.33)
- 装配：铝块平移到底板上方中心 (75, 10, 75)
- 边界条件：底板底面 (y=0) 全固定
- 载荷：铝块顶面 (y=60) 压力 5 MPa
- 网格：C3D8R, 种子尺寸 8mm
</plan>
<code>
from abaqus_api import AbaqusModel

m = AbaqusModel("BlockOnPlate")

# Part 1: Base plate
m.part.create_sketch("plate_sketch", sheet_size=400.0)
m.part.rectangle(p1=(0, 0), p2=(200.0, 10.0))
m.part.extrude_solid("BasePlate", depth=200.0)

m.part.create_face_set("BasePlate", "BottomFace",
    xmin=-0.1, ymin=-0.1, zmin=-0.1, xmax=200.1, ymax=0.1, zmax=200.1)

m.material.create_elastic("Steel", E=210000.0, nu=0.3)
m.material.create_solid_section("PlateSec", material="Steel")
m.material.assign_section("BasePlate", "PlateSec")

# Part 2: Aluminum block
m.part.create_sketch("block_sketch", sheet_size=200.0)
m.part.rectangle(p1=(0, 0), p2=(50.0, 50.0))
m.part.extrude_solid("Block", depth=50.0)

m.part.create_surface("Block", "TopSurf",
    xmin=-0.1, ymin=49.9, zmin=-0.1, xmax=50.1, ymax=50.1, zmax=50.1)

m.material.create_elastic("Aluminum", E=70000.0, nu=0.33)
m.material.create_solid_section("BlockSec", material="Aluminum")
m.material.assign_section("Block", "BlockSec")

# Assembly — create instances and position the block
m.assembly.create_instance("BasePlate")
m.assembly.create_instance("Block")
m.assembly.translate("Block-1", vector=(75.0, 10.0, 75.0))

# Step
m.step.create_static("Loading")

# BCs and loads
m.load.fix("FixBase", instance="BasePlate-1", set_name="BottomFace")
m.load.pressure("TopPressure", step="Loading", instance="Block-1",
                 surface="TopSurf", magnitude=5.0)

# Mesh
m.mesh.seed_part("BasePlate", size=8.0)
m.mesh.set_element_type("BasePlate", "C3D8R")
m.mesh.generate("BasePlate")

m.mesh.seed_part("Block", size=8.0)
m.mesh.set_element_type("Block", "C3D8R")
m.mesh.generate("Block")

# Submit and read results
result = m.submit("AssemblyJob", wait=True)
odb_summary = m.odb.max_values("AssemblyJob.odb", work_dir=result["output_dir"])
</code>""",
    },

    "mesh_control": {
        "keywords": ["网格细化", "局部细化", "种子", "加密", "seed", "refine", "二次单元", "C3D20"],
        "example": """\
## Example: Beam with local mesh refinement near fixed end

User: 一个 200x20x20mm 的梁，左端固定处网格加密到 2mm，其余用 8mm，右端面施加 5MPa 压力

<plan>
- 几何：200×20×20 mm 长方体，用 line() 画 L 形草图
- 材料：钢 (E=210000 MPa, ν=0.3)
- 网格：全局种子 8mm，左端附近边用 seed_edge_by_number 加密
- 边界条件：左端面 (x=0) 全固定
- 载荷：右端面 (x=200) 压力 5 MPa
</plan>
<code>
from abaqus_api import AbaqusModel

m = AbaqusModel("RefinedBeam")

# Geometry — use line() to draw an L-shaped profile
m.part.create_sketch("beam_sketch", sheet_size=400.0)
m.part.line(p1=(0, 0), p2=(200.0, 0))
m.part.line(p1=(200.0, 0), p2=(200.0, 20.0))
m.part.line(p1=(200.0, 20.0), p2=(0, 20.0))
m.part.line(p1=(0, 20.0), p2=(0, 0))
m.part.extrude_solid("Beam", depth=20.0)

# Face sets
m.part.create_face_set("Beam", "FixedEnd",
    xmin=-0.1, ymin=-0.1, zmin=-0.1, xmax=0.1, ymax=20.1, zmax=20.1)
m.part.create_surface("Beam", "LoadSurf",
    xmin=199.9, ymin=-0.1, zmin=-0.1, xmax=200.1, ymax=20.1, zmax=20.1)

# Material
m.material.create_elastic("Steel", E=210000.0, nu=0.3)
m.material.create_solid_section("BeamSec", material="Steel")
m.material.assign_section("Beam", "BeamSec")

# Assembly
m.assembly.create_instance("Beam")

# Step
m.step.create_static("Loading")

# BCs and loads
m.load.fix("Fixed", instance="Beam-1", set_name="FixedEnd")
m.load.pressure("Tip", step="Loading", instance="Beam-1",
                 surface="LoadSurf", magnitude=5.0)

# Mesh — global seed 8mm, local refinement near fixed end
m.mesh.seed_part("Beam", size=8.0)
m.mesh.seed_edge_by_number("Beam", None, 10,
    xmin=-0.1, ymin=-0.1, zmin=-0.1, xmax=0.1, ymax=20.1, zmax=20.1)
m.mesh.set_element_type("Beam", "C3D8R")
m.mesh.generate("Beam")

# Submit and read results
result = m.submit("RefinedJob", wait=True)
odb_summary = m.odb.max_values("RefinedJob.odb", work_dir=result["output_dir"])
</code>""",
    },

    "field_output": {
        "keywords": ["场输出", "应力场", "自定义输出", "field output", "PEEQ", "应变", "温度场"],
        "example": """\
## Example: Beam analysis with custom field output

User: 分析一个悬臂梁，需要输出应力、位移、塑性应变 (PEEQ) 和反力

<plan>
- 几何：200×20×20 mm 长方体
- 材料：铝合金 (E=70000 MPa, ν=0.33, 塑性)
- 边界条件：左端面 (x=0) 全固定
- 载荷：右端中心集中力 CF2=-1000 N
- 场输出：配置 S, U, RF, PEEQ 输出
- 后处理：使用 odb.field_output() 读取特定场变量
</plan>
<code>
from abaqus_api import AbaqusModel

m = AbaqusModel("FieldOutputBeam")

# Geometry
m.part.create_sketch("beam_sketch", sheet_size=400.0)
m.part.rectangle(p1=(0, 0), p2=(200.0, 20.0))
m.part.extrude_solid("Beam", depth=20.0)

# Face sets
m.part.create_face_set("Beam", "FixedEnd",
    xmin=-0.1, ymin=-0.1, zmin=-0.1, xmax=0.1, ymax=20.1, zmax=20.1)

# Material with plasticity
m.material.create_elastic("Aluminum", E=70000.0, nu=0.33)
m.material.create_plastic("Aluminum", table=[(275.0, 0.0), (310.0, 0.05)])
m.material.create_solid_section("BeamSec", material="Aluminum")
m.material.assign_section("Beam", "BeamSec")

# Assembly
m.assembly.create_instance("Beam")

# Step with custom field output
m.step.create_static("Loading", nlgeom=True)
m.step.set_field_output("Loading", variables=("S", "U", "RF", "PEEQ"))

# BCs
m.load.fix("Fixed", instance="Beam-1", set_name="FixedEnd")

# Mesh FIRST
m.mesh.seed_part("Beam", size=5.0)
m.mesh.set_element_type("Beam", "C3D8R")
m.mesh.generate("Beam")

# Load — concentrated force at beam tip center (200, 10, 10), AFTER meshing
m.load.concentrated_force_at_point("Tip", step="Loading",
    instance="Beam-1", point=(200.0, 10.0, 10.0), tolerance=5.0, cf2=-1000.0)

# Submit
result = m.submit("FieldJob", wait=True)

# Post-processing — read standard max values and specific field output
odb_summary = m.odb.max_values("FieldJob.odb", work_dir=result["output_dir"])
peeq_data = m.odb.field_output("FieldJob.odb", field_name="PEEQ",
    work_dir=result["output_dir"])
</code>""",
    },

    "hybrid_heat": {
        "keywords": ["热传导", "温度", "导热", "稳态", "瞬态", "heat", "thermal", "温度场", "传热"],
        "example": """\
## Example: Steady-state heat transfer using HYBRID approach (simplified API + native commands)

User: 一根200x20x20mm的钢杆，左端温度200度，右端25度，求稳态温度分布

<plan>
- 几何：200×20×20 mm 长方体（用简化API）
- 材料：钢，导热系数 50 W/(m·K)（用原生API定义 Conductivity）
- 分析步：HeatTransferStep, STEADY_STATE（原生API）
- 边界条件：左端面 200°C, 右端面 25°C（原生API: TemperatureBC）
- 网格：DC3D8 热传导单元（原生API设置单元类型）
</plan>
<code>
from abaqus_api import AbaqusModel

m = AbaqusModel("HeatBar")

# Geometry — use simplified API
m.part.create_sketch("bar_sketch", sheet_size=400.0)
m.part.rectangle(p1=(0, 0), p2=(200.0, 20.0))
m.part.extrude_solid("Bar", depth=20.0)

# Face sets for temperature BCs
m.part.create_face_set("Bar", "LeftFace",
    xmin=-0.1, ymin=-0.1, zmin=-0.1, xmax=0.1, ymax=20.1, zmax=20.1)
m.part.create_face_set("Bar", "RightFace",
    xmin=199.9, ymin=-0.1, zmin=-0.1, xmax=200.1, ymax=20.1, zmax=20.1)

# Material — simplified API for basic definition, native for conductivity
m.material.create_elastic("Steel", E=210000.0, nu=0.3)
m.material.create_solid_section("BarSec", material="Steel")
m.material.assign_section("Bar", "BarSec")

# Native: add thermal conductivity
m._buf.emit("mdb.models['HeatBar'].materials['Steel'].Conductivity(table=((50.0, ), ))")

# Assembly
m.assembly.create_instance("Bar")

# Native: create heat transfer step (not available in simplified API)
m._buf.emit("mdb.models['HeatBar'].HeatTransferStep(name='HeatStep', previous='Initial', response=STEADY_STATE, maxNumInc=100, initialInc=1.0, minInc=1e-5, maxInc=1.0)")

# Native: temperature boundary conditions
m._buf.emit("region_left = mdb.models['HeatBar'].rootAssembly.instances['Bar-1'].sets['LeftFace']")
m._buf.emit("mdb.models['HeatBar'].TemperatureBC(name='HotEnd', createStepName='HeatStep', region=region_left, magnitude=200.0)")
m._buf.emit("region_right = mdb.models['HeatBar'].rootAssembly.instances['Bar-1'].sets['RightFace']")
m._buf.emit("mdb.models['HeatBar'].TemperatureBC(name='ColdEnd', createStepName='HeatStep', region=region_right, magnitude=25.0)")

# Mesh — use simplified API but override element type for thermal
m.mesh.seed_part("Bar", size=5.0)
m._buf.emit("elemType = mesh.ElemType(elemCode=DC3D8, elemLibrary=STANDARD)")
m._buf.emit("cells = mdb.models['HeatBar'].parts['Bar'].cells[:]")
m._buf.emit("mdb.models['HeatBar'].parts['Bar'].setElementType(regions=(cells, ), elemTypes=(elemType, ))")
m.mesh.generate("Bar")

# Submit and read results
result = m.submit("HeatJob", wait=True)
odb_summary = m.odb.max_values("HeatJob.odb", work_dir=result["output_dir"])
</code>""",
    },

    "hybrid_buckling": {
        "keywords": ["屈曲", "临界载荷", "失稳", "稳定性", "buckling"],
        "example": """\
## Example: Linear buckling analysis using HYBRID approach

User: 一根细长钢柱500x10x10mm，底部固定，顶部受1MPa压力，求前3阶屈曲临界载荷

<plan>
- 几何：500×10×10 mm 细长柱（简化API）
- 材料：钢 (E=210000, ν=0.3)
- 分析步1：StaticStep 施加参考载荷（原生API）
- 分析步2：BuckleStep 求特征值（原生API）
- 边界条件：底面固定，顶面压力
</plan>
<code>
from abaqus_api import AbaqusModel

m = AbaqusModel("BucklingColumn")

# Geometry — simplified API
m.part.create_sketch("col_sketch", sheet_size=600.0)
m.part.rectangle(p1=(0, 0), p2=(500.0, 10.0))
m.part.extrude_solid("Column", depth=10.0)

# Sets
m.part.create_face_set("Column", "BottomFace",
    xmin=-0.1, ymin=-0.1, zmin=-0.1, xmax=500.1, ymax=0.1, zmax=10.1)
m.part.create_surface("Column", "TopSurf",
    xmin=-0.1, ymin=9.9, zmin=-0.1, xmax=500.1, ymax=10.1, zmax=10.1)

# Material
m.material.create_elastic("Steel", E=210000.0, nu=0.3)
m.material.create_solid_section("ColSec", material="Steel")
m.material.assign_section("Column", "ColSec")

# Assembly
m.assembly.create_instance("Column")

# Native: Static preload step + Buckling step
m._buf.emit("mdb.models['BucklingColumn'].StaticStep(name='Preload', previous='Initial')")
m._buf.emit("mdb.models['BucklingColumn'].BuckleStep(name='Buckle', previous='Preload', numEigen=3, maxIterations=30)")

# BCs and loads — simplified API for fixed BC, native for pressure in preload step
m.load.fix("FixBottom", instance="Column-1", set_name="BottomFace")
m._buf.emit("region_top = mdb.models['BucklingColumn'].rootAssembly.instances['Column-1'].surfaces['TopSurf']")
m._buf.emit("mdb.models['BucklingColumn'].Pressure(name='AxialLoad', createStepName='Preload', region=region_top, magnitude=1.0)")

# Mesh
m.mesh.seed_part("Column", size=5.0)
m.mesh.set_element_type("Column", "C3D8R")
m.mesh.generate("Column")

# Submit
result = m.submit("BuckleJob", wait=True)
odb_summary = m.odb.max_values("BuckleJob.odb", work_dir=result["output_dir"])
</code>""",
    },
}


import logging

logger = logging.getLogger(__name__)

# Classification prompt
_CLASSIFY_SYSTEM = """\
你是一个有限元分析类型分类器。根据用户的分析需求，从以下类型中选择最相关的1-2个。
只返回类型key，用逗号分隔，不要解释。

## 可选类型

- static_pressure: 面压力/均布载荷/分布力。适用于：梁、板、柱等承受面压力的静力学分析
- static_force: 集中力/点力/节点力。适用于：结构某点施加集中力
- gravity: 重力/自重/重量分析。适用于：需要考虑材料密度和重力加速度
- nonlinear: 大变形/塑性/屈服/弹塑性/材料非线性。适用于：超过弹性范围的分析
- frequency: 模态/固有频率/振动特性/动态响应特征/自由振动/共振。适用于：求解结构的固有频率和振型
- dynamic: 冲击/瞬态动力/显式分析/爆炸/碰撞/地震响应。适用于：时间很短的动态过程
- shell: 壳体/薄板/薄壁/管/筒/容器/内压/外压。适用于：壁厚远小于其他尺寸的结构（管子、容器、薄板）
- displacement: 位移约束/强制位移/位移控制/对称边界。适用于：通过施加位移而非力来加载
- revolve: 旋转体/轴对称/圆柱/圆盘/法兰/轮/轴/环形/圆形截面零件。适用于：绕轴旋转生成的实体几何
- assembly: 多零件/装配/组装/连接/叠放/接触/两个零件。适用于：包含多个独立零件的分析
- mesh_control: 网格细化/局部加密/应力集中区域网格/种子控制/二次单元。适用于：需要局部精细网格
- field_output: 自定义场输出/PEEQ/塑性应变输出/特定场变量读取。适用于：需要输出非默认的场变量
- hybrid_heat: 热传导/温度场/稳态传热/瞬态传热/导热/对流换热。适用于：热分析（需要HeatTransferStep）
- hybrid_buckling: 屈曲/失稳/临界载荷/稳定性。适用于：线性特征值屈曲分析（需要BuckleStep）

## 分类示例

用户: "一根管子承受内压" → shell
用户: "圆形法兰盘受轴向压力" → revolve
用户: "分析结构在地震作用下的响应" → dynamic
用户: "两个零件的连接分析" → assembly
用户: "求解梁的动态响应特征" → frequency
用户: "柱体轴压稳定性分析" → static_pressure"""


def _classify_by_llm(user_input, llm_client):
    """Use Kimi to classify the analysis type. Returns list of keys.

    Args:
        user_input: The user's natural language input.
        llm_client: An LLMClient instance.

    Returns:
        List of matched EXAMPLES keys, or empty list on failure.
    """
    prompt = _CLASSIFY_SYSTEM
    messages = [{"role": "user", "content": user_input}]

    try:
        response = llm_client.generate(
            system=prompt,
            messages=messages,
            max_tokens=50,
        )
        # Parse comma-separated keys from response
        raw_keys = [k.strip().lower() for k in response.replace("\n", ",").split(",")]
        valid_keys = [k for k in raw_keys if k in EXAMPLES]
        if valid_keys:
            logger.info("LLM classification: input='%s' → %s", user_input[:50], valid_keys)
            return valid_keys
        logger.warning("LLM classification returned no valid keys: '%s'", response)
    except Exception as e:
        logger.warning("LLM classification failed, falling back to keywords: %s", e)

    return []


def _select_by_keywords(user_input, top_k=2):
    """Fallback: select examples by keyword matching.

    Args:
        user_input: The user's natural language input.
        top_k: Number of examples to return.

    Returns:
        List of EXAMPLES keys, ordered by relevance.
    """
    scores = []
    for key, entry in EXAMPLES.items():
        count = sum(1 for kw in entry["keywords"] if kw in user_input)
        scores.append((count, key))

    scores.sort(key=lambda x: x[0], reverse=True)

    if scores[0][0] == 0:
        return ["static_pressure"]

    return [key for count, key in scores[:top_k] if count > 0]


def select_examples(user_input, top_k=2, llm_client=None):
    """Select the most relevant few-shot examples.

    Strategy: keyword matching first (fast, free). Only falls back to
    LLM classification when keywords score 0 and llm_client is available.

    Args:
        user_input: The user's natural language input.
        top_k: Number of examples to return.
        llm_client: Optional LLMClient for LLM-based classification.

    Returns:
        List of example text strings, ordered by relevance.
    """
    if not user_input:
        return [EXAMPLES["static_pressure"]["example"]]

    # Try keyword matching first (fast, no API call)
    keys = _select_by_keywords(user_input, top_k)

    # Only use LLM classification if keywords found nothing useful
    if keys == ["static_pressure"] and llm_client is not None:
        llm_keys = _classify_by_llm(user_input, llm_client)
        if llm_keys:
            keys = llm_keys
            logger.info("LLM classification override: %s", keys)

    return [EXAMPLES[k]["example"] for k in keys[:top_k]]


def build_system_prompt(user_input="", llm_client=None, rag=None):
    """Build the complete system prompt with role, API reference, and examples.

    Args:
        user_input: The user's message, used to select relevant few-shot examples.
        llm_client: Optional LLMClient for LLM-based example classification.
        rag: Optional AbaqusRAG instance for native API documentation retrieval.
    """
    api_ref = build_api_reference()
    examples = select_examples(user_input, llm_client=llm_client)
    examples_text = "\n\n".join(examples)

    # RAG retrieval for native Abaqus API documentation
    rag_context = ""
    if rag and user_input:
        try:
            rag_results = rag.retrieve(user_input, top_k=5)
            if rag_results:
                rag_context = "\n\n# Abaqus Native API Reference (from documentation)\n"
                rag_context += (
                    "When the simplified API above does not cover the needed functionality, "
                    "you may use these native Abaqus Python commands directly.\n"
                    "Wrap each native command with m._buf.emit(\"...\") to add it to the code buffer.\n\n"
                )
                for doc in rag_results:
                    rag_context += f"## {doc['title']}\n{doc['content']}\n\n"
        except Exception as e:
            logger.warning("RAG retrieval failed: %s", e)

    return f"""{ROLE_PROMPT}

# Simplified API Reference (preferred)
{api_ref}

# Few-shot Examples
{examples_text}
{rag_context}"""
