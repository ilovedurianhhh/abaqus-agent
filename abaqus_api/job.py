"""Job creation, submission, and monitoring."""


class JobBuilder:
    def __init__(self, buf, model_name):
        self._buf = buf
        self._model = model_name

    def create(self, name, description="", num_cpus=1, memory=90):
        """Create an analysis job."""
        self._buf.emit(
            f"j = mdb.Job(name='{name}', model='{self._model}', "
            f"description='{description}', numCpus={num_cpus}, "
            f"memory={memory}, memoryUnits=PERCENTAGE)"
        )

    def submit_and_wait(self, name):
        """Submit the job and wait for completion."""
        self._buf.emit(f"mdb.jobs['{name}'].submit(consistencyChecking=OFF)")
        self._buf.emit(f"mdb.jobs['{name}'].waitForCompletion()")

    def write_input(self, name):
        """Write the input file without submitting."""
        self._buf.emit(f"mdb.jobs['{name}'].writeInput(consistencyChecking=OFF)")

    def emit_result_capture(self, name):
        """Emit code to capture job completion status as __result__.

        Uses mdb.jobs[name].status as primary check, with .sta file
        fallback for Abaqus versions where status returns None.
        """
        self._buf.emit(
            f"import os\n"
            f"_status = str(mdb.jobs['{name}'].status)\n"
            f"if _status == 'None' and os.path.exists('{name}.sta'):\n"
            f"    with open('{name}.sta', 'r') as _f:\n"
            f"        _sta_content = _f.read()\n"
            f"    if 'COMPLETED SUCCESSFULLY' in _sta_content:\n"
            f"        _status = 'COMPLETED'\n"
            f"    elif 'HAS NOT BEEN COMPLETED' in _sta_content:\n"
            f"        _status = 'ABORTED'\n"
            f"__result__ = {{\n"
            f"    'job_name': '{name}',\n"
            f"    'status': _status,\n"
            f"    'odb_exists': os.path.exists('{name}.odb'),\n"
            f"}}"
        )
