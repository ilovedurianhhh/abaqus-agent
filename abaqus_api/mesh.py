"""Mesh seeding, element type assignment, and mesh generation."""


class MeshBuilder:
    def __init__(self, buf, model_name):
        self._buf = buf
        self._model = model_name

    def seed_part(self, part_name, size):
        """Seed a part with a global element size."""
        self._buf.emit(
            f"p = mdb.models['{self._model}'].parts['{part_name}']"
        )
        self._buf.emit(f"p.seedPart(size={size}, deviationFactor=0.1, minSizeFactor=0.1)")

    def seed_edge_by_number(self, part_name, edge_bbox, number,
                            xmin=None, ymin=None, zmin=None,
                            xmax=None, ymax=None, zmax=None):
        """Seed edges selected by bounding box with a fixed number of elements."""
        self._buf.emit(
            f"p = mdb.models['{self._model}'].parts['{part_name}']"
        )
        self._buf.emit(
            f"edges = p.edges.getByBoundingBox({xmin}, {ymin}, {zmin}, {xmax}, {ymax}, {zmax})"
        )
        self._buf.emit(f"p.seedEdgeByNumber(edges=edges, number={number})")

    def set_element_type(self, part_name, elem_code, region_set=None):
        """Set element type for a part.

        Args:
            part_name: Part name.
            elem_code: Element type code, e.g. 'C3D8R', 'C3D20R', 'C3D4'.
            region_set: Optional set name; if None, applies to all cells.
        """
        self._buf.emit(
            f"p = mdb.models['{self._model}'].parts['{part_name}']"
        )
        self._buf.emit(f"import mesh")
        self._buf.emit(
            f"elemType = mesh.ElemType(elemCode={elem_code}, elemLibrary=STANDARD)"
        )
        if region_set:
            self._buf.emit(
                f"region = p.sets['{region_set}']"
            )
        else:
            self._buf.emit(
                f"region = (p.cells[:], )"
            )
        self._buf.emit(f"p.setElementType(regions=region, elemTypes=(elemType, ))")

    def generate(self, part_name):
        """Generate the mesh for a part."""
        self._buf.emit(
            f"p = mdb.models['{self._model}'].parts['{part_name}']"
        )
        self._buf.emit(f"p.generateMesh()")
