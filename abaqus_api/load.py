"""Loads and boundary conditions."""


class LoadBuilder:
    def __init__(self, buf, model_name):
        self._buf = buf
        self._model = model_name

    def _instance_set_region(self, instance, set_name):
        """Generate code to get a region from an instance set."""
        return (
            f"a = mdb.models['{self._model}'].rootAssembly\n"
            f"region = a.instances['{instance}'].sets['{set_name}']"
        )

    def _instance_surface_region(self, instance, surface):
        """Generate code to get a region from an instance surface."""
        return (
            f"a = mdb.models['{self._model}'].rootAssembly\n"
            f"region = a.instances['{instance}'].surfaces['{surface}']"
        )

    def fix(self, name, instance, set_name, step="Initial"):
        """Apply fully fixed (encastre) boundary condition."""
        self._buf.emit(self._instance_set_region(instance, set_name))
        self._buf.emit(
            f"mdb.models['{self._model}'].EncastreBC("
            f"name='{name}', createStepName='{step}', region=region)"
        )

    def displacement_bc(self, name, instance, set_name, step="Initial",
                        u1=None, u2=None, u3=None, ur1=None, ur2=None, ur3=None):
        """Apply displacement/rotation boundary condition."""
        self._buf.emit(self._instance_set_region(instance, set_name))
        kwargs = {}
        for key, val in [("u1", u1), ("u2", u2), ("u3", u3),
                         ("ur1", ur1), ("ur2", ur2), ("ur3", ur3)]:
            if val is not None:
                kwargs[key] = val
        kw_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        self._buf.emit(
            f"mdb.models['{self._model}'].DisplacementBC("
            f"name='{name}', createStepName='{step}', region=region, {kw_str})"
        )

    def pressure(self, name, step, instance, surface, magnitude):
        """Apply pressure load on a surface."""
        self._buf.emit(self._instance_surface_region(instance, surface))
        self._buf.emit(
            f"mdb.models['{self._model}'].Pressure("
            f"name='{name}', createStepName='{step}', region=region, "
            f"magnitude={magnitude})"
        )

    def concentrated_force(self, name, step, instance, set_name,
                           cf1=None, cf2=None, cf3=None):
        """Apply concentrated force on a pre-existing set."""
        self._buf.emit(self._instance_set_region(instance, set_name))
        kwargs = {}
        for key, val in [("cf1", cf1), ("cf2", cf2), ("cf3", cf3)]:
            if val is not None:
                kwargs[key] = val
        kw_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        self._buf.emit(
            f"mdb.models['{self._model}'].ConcentratedForce("
            f"name='{name}', createStepName='{step}', region=region, {kw_str})"
        )

    def concentrated_force_at_point(self, name, step, instance,
                                     point, tolerance=2.0,
                                     cf1=None, cf2=None, cf3=None):
        """Apply concentrated force at a coordinate point (mesh-level).

        Finds the closest mesh node to the given point and applies the
        force. Call AFTER m.mesh.generate(). Auto-expands search if needed.

        Args:
            name: Load name.
            step: Step name.
            instance: Instance name (e.g. 'Beam-1').
            point: (x, y, z) tuple of the target coordinate.
            tolerance: Initial half-width of bounding box for node search (default: 2.0).
                       Automatically expands up to 32x if no nodes found.
            cf1, cf2, cf3: Force components (e.g., cf2=-500.0 for -Y direction).

        Example:
            m.load.concentrated_force_at_point("TipLoad", "Step1", "Beam-1",
                                               point=(200.0, 20.0, 5.0), cf2=-500.0)
        """
        x, y, z = point
        tol = tolerance
        set_name = f"_CF_{name}"
        self._buf.emit(
            f"import math\n"
            f"a = mdb.models['{self._model}'].rootAssembly\n"
            f"a.regenerate()\n"
            f"_tol = {tol}\n"
            f"_nodes = None\n"
            f"for _attempt in range(5):\n"
            f"    _nodes = a.instances['{instance}'].nodes.getByBoundingBox("
            f"{x} - _tol, {y} - _tol, {z} - _tol, "
            f"{x} + _tol, {y} + _tol, {z} + _tol)\n"
            f"    if len(_nodes) > 0:\n"
            f"        break\n"
            f"    _tol = _tol * 2\n"
            f"if len(_nodes) == 0:\n"
            f"    raise RuntimeError("
            f"'No mesh nodes found near ({x}, {y}, {z}) even with tol=' "
            f"+ str(_tol) + '. Check that mesh.generate() was called first.')\n"
            f"_best = min(_nodes, key=lambda _n: math.sqrt("
            f"(_n.coordinates[0] - {x})**2 + "
            f"(_n.coordinates[1] - {y})**2 + "
            f"(_n.coordinates[2] - {z})**2))\n"
            f"_all_nodes = a.instances['{instance}'].nodes\n"
            f"_single = _all_nodes[_best.label - 1 : _best.label]\n"
            f"region = a.Set(nodes=_single, name='{set_name}')"
        )
        kwargs = {}
        for key, val in [("cf1", cf1), ("cf2", cf2), ("cf3", cf3)]:
            if val is not None:
                kwargs[key] = val
        kw_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        self._buf.emit(
            f"mdb.models['{self._model}'].ConcentratedForce("
            f"name='{name}', createStepName='{step}', region=region, {kw_str})"
        )

    def gravity(self, name, step, comp2=-9810.0):
        """Apply gravity load (default: -Y direction, mm/s^2)."""
        self._buf.emit(
            f"mdb.models['{self._model}'].Gravity("
            f"name='{name}', createStepName='{step}', comp2={comp2})"
        )
