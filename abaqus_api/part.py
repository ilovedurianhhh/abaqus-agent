"""Geometry modeling: sketches, extrusions, sets, and surfaces."""


class PartBuilder:
    def __init__(self, buf, model_name):
        self._buf = buf
        self._model = model_name

    def create_sketch(self, name, sheet_size=200.0):
        """Create a constrained sketch."""
        self._buf.emit(
            f"s = mdb.models['{self._model}'].ConstrainedSketch("
            f"name='{name}', sheetSize={sheet_size})"
        )

    def rectangle(self, p1, p2):
        """Draw a rectangle on the current sketch."""
        self._buf.emit(
            f"s.rectangle(point1=({p1[0]}, {p1[1]}), point2=({p2[0]}, {p2[1]}))"
        )

    def circle(self, center, radius):
        """Draw a circle on the current sketch."""
        self._buf.emit(
            f"s.CircleByCenterPerimeter(center=({center[0]}, {center[1]}), "
            f"point1=({center[0] + radius}, {center[1]}))"
        )

    def line(self, p1, p2):
        """Draw a line on the current sketch."""
        self._buf.emit(
            f"s.Line(point1=({p1[0]}, {p1[1]}), point2=({p2[0]}, {p2[1]}))"
        )

    def extrude_solid(self, part_name, depth):
        """Create a 3D solid part by extruding the current sketch."""
        self._buf.emit(
            f"p = mdb.models['{self._model}'].Part("
            f"name='{part_name}', dimensionality=THREE_D, type=DEFORMABLE_BODY)"
        )
        self._buf.emit(f"p.BaseSolidExtrude(sketch=s, depth={depth})")
        self._buf.emit(f"del s")

    def extrude_shell(self, part_name, depth):
        """Create a 3D shell part by extruding the current sketch."""
        self._buf.emit(
            f"p = mdb.models['{self._model}'].Part("
            f"name='{part_name}', dimensionality=THREE_D, type=DEFORMABLE_BODY)"
        )
        self._buf.emit(f"p.BaseShellExtrude(sketch=s, depth={depth})")
        self._buf.emit(f"del s")

    def revolve_solid(self, part_name, angle=360.0):
        """Create a 3D solid part by revolving the current sketch."""
        self._buf.emit(
            f"p = mdb.models['{self._model}'].Part("
            f"name='{part_name}', dimensionality=THREE_D, type=DEFORMABLE_BODY)"
        )
        self._buf.emit(f"p.BaseSolidRevolve(sketch=s, angle={angle})")
        self._buf.emit(f"del s")

    def create_set_by_bounding_box(self, part_name, set_name,
                                    xmin, ymin, zmin, xmax, ymax, zmax):
        """Create a cell/node set using a bounding box selection."""
        self._buf.emit(
            f"p = mdb.models['{self._model}'].parts['{part_name}']"
        )
        self._buf.emit(
            f"cells = p.cells.getByBoundingBox("
            f"{xmin}, {ymin}, {zmin}, {xmax}, {ymax}, {zmax})"
        )
        self._buf.emit(
            f"p.Set(cells=cells, name='{set_name}')"
        )

    def create_face_set(self, part_name, set_name,
                        xmin, ymin, zmin, xmax, ymax, zmax):
        """Create a face set using a bounding box selection."""
        self._buf.emit(
            f"p = mdb.models['{self._model}'].parts['{part_name}']"
        )
        self._buf.emit(
            f"faces = p.faces.getByBoundingBox("
            f"{xmin}, {ymin}, {zmin}, {xmax}, {ymax}, {zmax})"
        )
        self._buf.emit(
            f"p.Set(faces=faces, name='{set_name}')"
        )

    def create_vertex_set(self, part_name, set_name,
                          xmin, ymin, zmin, xmax, ymax, zmax):
        """Create a vertex (node) set using a bounding box selection.

        Use this to select geometry points for concentrated forces, etc.
        The bounding box should be small enough to capture only the
        desired vertex/vertices.
        """
        self._buf.emit(
            f"p = mdb.models['{self._model}'].parts['{part_name}']"
        )
        self._buf.emit(
            f"verts = p.vertices.getByBoundingBox("
            f"{xmin}, {ymin}, {zmin}, {xmax}, {ymax}, {zmax})"
        )
        self._buf.emit(
            f"p.Set(vertices=verts, name='{set_name}')"
        )

    def create_node_set_by_bounding_box(self, part_name, set_name,
                                         xmin, ymin, zmin, xmax, ymax, zmax):
        """Create a node set using a bounding box selection (mesh-level).

        Unlike create_vertex_set (geometry-level), this selects mesh nodes
        directly. The part must be meshed before calling this method.
        """
        self._buf.emit(
            f"p = mdb.models['{self._model}'].parts['{part_name}']"
        )
        self._buf.emit(
            f"nodes = p.nodes.getByBoundingBox("
            f"{xmin}, {ymin}, {zmin}, {xmax}, {ymax}, {zmax})"
        )
        self._buf.emit(
            f"p.Set(nodes=nodes, name='{set_name}')"
        )

    def create_surface(self, part_name, surface_name,
                       xmin, ymin, zmin, xmax, ymax, zmax):
        """Create a surface using a bounding box face selection."""
        self._buf.emit(
            f"p = mdb.models['{self._model}'].parts['{part_name}']"
        )
        self._buf.emit(
            f"faces = p.faces.getByBoundingBox("
            f"{xmin}, {ymin}, {zmin}, {xmax}, {ymax}, {zmax})"
        )
        self._buf.emit(
            f"p.Surface(side1Faces=faces, name='{surface_name}')"
        )
