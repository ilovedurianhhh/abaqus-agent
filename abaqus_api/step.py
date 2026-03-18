"""Analysis step definitions."""


class StepBuilder:
    def __init__(self, buf, model_name):
        self._buf = buf
        self._model = model_name

    def create_static(self, name, previous="Initial", nlgeom=False,
                      initial_inc=1.0, max_inc=1.0, min_inc=1e-5, max_num_inc=100):
        """Create a Static/General analysis step."""
        nl = "ON" if nlgeom else "OFF"
        self._buf.emit(
            f"mdb.models['{self._model}'].StaticStep("
            f"name='{name}', previous='{previous}', nlgeom={nl}, "
            f"initialInc={initial_inc}, maxInc={max_inc}, "
            f"minInc={min_inc}, maxNumInc={max_num_inc})"
        )

    def create_dynamic_explicit(self, name, time_period, previous="Initial"):
        """Create a Dynamic/Explicit analysis step."""
        self._buf.emit(
            f"mdb.models['{self._model}'].ExplicitDynamicsStep("
            f"name='{name}', previous='{previous}', timePeriod={time_period})"
        )

    def create_frequency(self, name, num_eigen=10, previous="Initial"):
        """Create a Frequency analysis step."""
        self._buf.emit(
            f"mdb.models['{self._model}'].FrequencyStep("
            f"name='{name}', previous='{previous}', numEigen={num_eigen})"
        )

    def set_field_output(self, step_name, variables=None):
        """Configure field output for a step."""
        if variables is None:
            variables = ("S", "U", "RF")
        var_str = ", ".join(f"'{v}'" for v in variables)
        self._buf.emit(
            f"mdb.models['{self._model}'].fieldOutputRequests['F-Output-1'].setValues("
            f"variables=({var_str}, ))"
        )
