"""AbaqusModel - main class orchestrating the script accumulation workflow.

Usage:
    m = AbaqusModel("MyModel")
    m.part.create_sketch(...)
    m.part.extrude_solid(...)
    m.material.create_elastic(...)
    ...
    result = m.submit("JobName", wait=True)
    summary = m.odb.max_values("JobName.odb")
"""

import os
import sys
from datetime import datetime

# Allow importing the bridge from the parent directory
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from abaqus_bridge import AbaqusBridge

from .codegen import CodeBuffer
from .part import PartBuilder
from .material import MaterialBuilder
from .assembly import AssemblyBuilder
from .step import StepBuilder
from .load import LoadBuilder
from .mesh import MeshBuilder
from .job import JobBuilder
from .odb import OdbReader

# Default output root: <project>/output/
OUTPUT_ROOT = os.path.join(_parent, "output")


class AbaqusModel:
    """High-level API for building and running an Abaqus analysis.

    Accumulates Abaqus Python 2.7 code via builder sub-objects,
    then executes the complete script in a single AbaqusBridge call.
    All solver output files are written to a timestamped directory
    under output/.
    """

    def __init__(self, model_name="Model-1", bridge=None):
        self.model_name = model_name
        self._buf = CodeBuffer(model_name)
        self._bridge = bridge or AbaqusBridge()
        self.output_dir = None  # set by submit()

        # Builder sub-objects share the same code buffer
        self.part = PartBuilder(self._buf, model_name)
        self.material = MaterialBuilder(self._buf, model_name)
        self.assembly = AssemblyBuilder(self._buf, model_name)
        self.step = StepBuilder(self._buf, model_name)
        self.load = LoadBuilder(self._buf, model_name)
        self.mesh = MeshBuilder(self._buf, model_name)
        self.job = JobBuilder(self._buf, model_name)
        self.odb = OdbReader(self._bridge)

    def preview(self):
        """Return the accumulated Abaqus Python script as a string."""
        return self._buf.preview()

    def reset(self):
        """Clear the code buffer to start fresh."""
        self._buf.reset()

    def run(self, timeout=120):
        """Execute the accumulated script in Abaqus CAE (no job submission).

        Returns:
            AbaqusResult from the bridge.
        """
        code = self._buf.preview()
        return self._bridge.execute(code, timeout=timeout)

    def _make_output_dir(self, job_name):
        """Create a timestamped output directory and return its absolute path."""
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dirname = f"{job_name}_{stamp}"
        out = os.path.join(OUTPUT_ROOT, dirname)
        os.makedirs(out, exist_ok=True)
        return out

    def submit(self, job_name, wait=True, description="", num_cpus=1,
               memory=90, timeout=600):
        """Create a job, submit it, and optionally wait for completion.

        All solver output files (.odb, .dat, .msg, ...) are written to
        output/<job_name>_<timestamp>/.

        Args:
            job_name: Name for the Abaqus job.
            wait: If True, wait for job completion before returning.
            description: Optional job description.
            num_cpus: Number of CPUs for the solver.
            memory: Memory percentage.
            timeout: Max seconds to wait for the entire execution.

        Returns:
            dict with job status info and output_dir, or error dict on failure.
        """
        self.output_dir = self._make_output_dir(job_name)

        self.job.create(job_name, description=description,
                        num_cpus=num_cpus, memory=memory)
        if wait:
            self.job.submit_and_wait(job_name)
        else:
            self._buf.emit(
                f"mdb.jobs['{job_name}'].submit(consistencyChecking=OFF)"
            )
        self.job.emit_result_capture(job_name)

        result = self._bridge.execute(
            self._buf.preview(), timeout=timeout, work_dir=self.output_dir
        )

        if result.success and result.result_data:
            data = result.result_data
            data["output_dir"] = self.output_dir
            return data
        return {
            "error": True,
            "message": result.stderr or result.stdout,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "result_data": result.result_data,
            "output_dir": self.output_dir,
        }
