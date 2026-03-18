"""Material definition, section creation, and section assignment."""


class MaterialBuilder:
    def __init__(self, buf, model_name):
        self._buf = buf
        self._model = model_name

    def create_elastic(self, name, E, nu):
        """Create an isotropic elastic material."""
        self._buf.emit(
            f"mat = mdb.models['{self._model}'].Material(name='{name}')"
        )
        self._buf.emit(
            f"mat.Elastic(table=(({E}, {nu}), ))"
        )

    def create_density(self, name, density):
        """Add density to an existing material."""
        self._buf.emit(
            f"mdb.models['{self._model}'].materials['{name}']"
            f".Density(table=(({density}, ), ))"
        )

    def create_plastic(self, name, table):
        """Add plasticity to an existing material.

        Args:
            name: Material name (must already exist).
            table: List of (yield_stress, plastic_strain) tuples.
        """
        table_str = ", ".join(f"({s}, {e})" for s, e in table)
        self._buf.emit(
            f"mdb.models['{self._model}'].materials['{name}']"
            f".Plastic(table=({table_str}, ))"
        )

    def create_solid_section(self, name, material, part=None):
        """Create a homogeneous solid section."""
        self._buf.emit(
            f"mdb.models['{self._model}'].HomogeneousSolidSection("
            f"name='{name}', material='{material}', thickness=None)"
        )

    def create_shell_section(self, name, material, thickness):
        """Create a homogeneous shell section."""
        self._buf.emit(
            f"mdb.models['{self._model}'].HomogeneousShellSection("
            f"name='{name}', material='{material}', thickness={thickness})"
        )

    def assign_section(self, part_name, section_name, set_name=None):
        """Assign a section to a part (or a specific set within the part).

        If set_name is None, assigns to the entire part using a set of all cells.
        """
        self._buf.emit(
            f"p = mdb.models['{self._model}'].parts['{part_name}']"
        )
        if set_name:
            self._buf.emit(
                f"region = p.sets['{set_name}']"
            )
        else:
            self._buf.emit(
                f"region = p.Set(cells=p.cells[:], name='_WholePartSet')"
            )
        self._buf.emit(
            f"p.SectionAssignment(region=region, sectionName='{section_name}')"
        )
