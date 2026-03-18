"""ODB post-processing: read stress, displacement, and reaction forces.

This module generates a *separate* script that is executed in its own
AbaqusBridge.execute() call, because the ODB must be opened after the
job has completed and the .odb file is closed by the solver.
"""

import os
import sys

# Allow importing the bridge from the parent directory
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)


class OdbReader:
    def __init__(self, bridge):
        self._bridge = bridge

    def max_values(self, odb_path, work_dir=None):
        """Read max Mises stress and max displacement magnitude from an ODB.

        Args:
            odb_path: Path to the .odb file (filename or absolute path).
            work_dir: Directory where the .odb resides (Abaqus CWD).

        Returns:
            dict with keys: max_mises, max_displacement, max_rf_magnitude
            or dict with 'error' key on failure.
        """
        code = f"""
import odbAccess
import math

odb = odbAccess.openOdb(path=r'{odb_path}', readOnly=True)

# Get last step, last frame
step = odb.steps[odb.steps.keys()[-1]]
frame = step.frames[-1]

# Max Mises stress
stress = frame.fieldOutputs['S']
max_mises = 0.0
for val in stress.values:
    if hasattr(val, 'mises') and val.mises is not None:
        if val.mises > max_mises:
            max_mises = val.mises

# Max displacement magnitude
disp = frame.fieldOutputs['U']
max_u = 0.0
for val in disp.values:
    mag = math.sqrt(val.data[0]**2 + val.data[1]**2 + val.data[2]**2)
    if mag > max_u:
        max_u = mag

# Max reaction force magnitude
max_rf = 0.0
if 'RF' in frame.fieldOutputs:
    rf = frame.fieldOutputs['RF']
    for val in rf.values:
        mag = math.sqrt(val.data[0]**2 + val.data[1]**2 + val.data[2]**2)
        if mag > max_rf:
            max_rf = mag

odb.close()

__result__ = {{
    'max_mises': max_mises,
    'max_displacement': max_u,
    'max_rf_magnitude': max_rf,
}}
"""
        result = self._bridge.execute(code, work_dir=work_dir)
        if result.success and result.result_data:
            return result.result_data
        return {
            "error": True,
            "message": result.stderr or result.stdout,
            "result_data": result.result_data,
        }

    def field_output(self, odb_path, field_name, step_name=None,
                     frame_index=-1, work_dir=None):
        """Read a specific field output and return summary statistics.

        Args:
            odb_path: Path to the .odb file.
            field_name: Field output key, e.g. 'S', 'U', 'RF'.
            step_name: Step name; defaults to last step.
            frame_index: Frame index; defaults to -1 (last frame).
            work_dir: Directory where the .odb resides (Abaqus CWD).

        Returns:
            dict with min, max, and component info.
        """
        step_sel = (
            f"odb.steps['{step_name}']" if step_name
            else "odb.steps[odb.steps.keys()[-1]]"
        )
        code = f"""
import odbAccess
import math

odb = odbAccess.openOdb(path=r'{odb_path}', readOnly=True)
step = {step_sel}
frame = step.frames[{frame_index}]

fo = frame.fieldOutputs['{field_name}']
comp_labels = list(fo.componentLabels) if fo.componentLabels else []

vals = []
for v in fo.values:
    vals.append(list(v.data) if hasattr(v.data, '__len__') else [v.data])

# Compute magnitude for vector/tensor fields
mags = []
for row in vals:
    mags.append(math.sqrt(sum(x**2 for x in row)))

__result__ = {{
    'field': '{field_name}',
    'components': comp_labels,
    'num_values': len(vals),
    'max_magnitude': max(mags) if mags else 0.0,
    'min_magnitude': min(mags) if mags else 0.0,
}}

odb.close()
"""
        result = self._bridge.execute(code, work_dir=work_dir)
        if result.success and result.result_data:
            return result.result_data
        return {
            "error": True,
            "message": result.stderr or result.stdout,
            "result_data": result.result_data,
        }
