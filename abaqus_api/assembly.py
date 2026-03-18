"""Assembly and instance management."""


class AssemblyBuilder:
    def __init__(self, buf, model_name):
        self._buf = buf
        self._model = model_name

    def create_instance(self, part_name, instance_name=None, dependent=True):
        """Create a part instance in the root assembly.

        Args:
            part_name: Name of the part to instance.
            instance_name: Instance name. Defaults to '{part_name}-1'.
            dependent: If True, instance is dependent on the part mesh.
        """
        dep = "ON" if dependent else "OFF"
        inst_name = instance_name or f"{part_name}-1"
        self._buf.emit(
            f"a = mdb.models['{self._model}'].rootAssembly"
        )
        self._buf.emit(
            f"a.DatumCsysByDefault(CARTESIAN)"
        )
        self._buf.emit(
            f"a.Instance(name='{inst_name}', "
            f"part=mdb.models['{self._model}'].parts['{part_name}'], "
            f"dependent={dep})"
        )

    def translate(self, instance_name, vector):
        """Translate an instance by a vector (x, y, z)."""
        self._buf.emit(
            f"a = mdb.models['{self._model}'].rootAssembly"
        )
        self._buf.emit(
            f"a.translate(instanceList=('{instance_name}', ), "
            f"vector=({vector[0]}, {vector[1]}, {vector[2]}))"
        )

    def rotate(self, instance_name, axis_point, axis_direction, angle):
        """Rotate an instance around an axis."""
        self._buf.emit(
            f"a = mdb.models['{self._model}'].rootAssembly"
        )
        self._buf.emit(
            f"a.rotate(instanceList=('{instance_name}', ), "
            f"axisPoint=({axis_point[0]}, {axis_point[1]}, {axis_point[2]}), "
            f"axisDirection=({axis_direction[0]}, {axis_direction[1]}, {axis_direction[2]}), "
            f"angle={angle})"
        )
