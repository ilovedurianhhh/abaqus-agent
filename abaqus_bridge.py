"""
Abaqus Bridge - Communication layer between Agent and Abaqus CAE.

Usage:
    from abaqus_bridge import AbaqusBridge

    bridge = AbaqusBridge()
    result = bridge.execute("from abaqus import mdb; print(mdb.models.keys())")
    print(result.stdout)
    print(result.returncode)
"""
import subprocess
import tempfile
import os
import json
import time
import uuid

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(WORK_DIR, "scripts")
RESULTS_DIR = os.path.join(WORK_DIR, "results")


class AbaqusResult:
    def __init__(self, returncode, stdout, stderr, result_data=None, duration=0):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.result_data = result_data
        self.duration = duration
        self.success = returncode == 0

    def __repr__(self):
        status = "OK" if self.success else "FAILED"
        return f"AbaqusResult({status}, {self.duration:.1f}s)"


class AbaqusBridge:
    def __init__(self, work_dir=None):
        self.work_dir = work_dir or WORK_DIR
        os.makedirs(SCRIPTS_DIR, exist_ok=True)
        os.makedirs(RESULTS_DIR, exist_ok=True)

    def execute(self, code, timeout=120, return_data=True, work_dir=None):
        """
        Execute Python code inside Abaqus CAE (noGUI mode).

        Args:
            code: Python code string to execute in Abaqus environment.
            timeout: Max seconds to wait for execution.
            return_data: If True, wraps code to capture result as JSON.
            work_dir: Override the working directory for this execution.

        Returns:
            AbaqusResult with stdout, stderr, returncode, and optional result_data.
        """
        cwd = work_dir or self.work_dir
        run_id = uuid.uuid4().hex[:8]
        script_path = os.path.join(SCRIPTS_DIR, f"run_{run_id}.py")
        result_path = os.path.join(RESULTS_DIR, f"result_{run_id}.json")

        # Wrap user code with result capture
        wrapped = self._wrap_code(code, result_path, return_data)

        with open(script_path, "w") as f:
            f.write(wrapped)

        start = time.time()
        try:
            proc = subprocess.run(
                ["powershell.exe", "-Command",
                 f"cd '{cwd}'; abaqus cae noGUI='{script_path}'"],
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
            duration = time.time() - start

            result_data = None
            if return_data and os.path.exists(result_path):
                with open(result_path, "r", encoding="utf-8") as f:
                    raw = f.read()
                try:
                    result_data = json.loads(raw)
                except (json.JSONDecodeError, ValueError):
                    result_data = {"error": True, "message": f"Invalid JSON in result file: {raw[:500]}"}

            return AbaqusResult(
                returncode=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                result_data=result_data,
                duration=duration,
            )
        except subprocess.TimeoutExpired:
            return AbaqusResult(
                returncode=-1,
                stdout="",
                stderr=f"Timeout after {timeout}s",
                duration=timeout,
            )
        finally:
            # Clean up temp files
            for path in [script_path]:
                try:
                    os.remove(path)
                except OSError:
                    pass

    def execute_file(self, script_path, timeout=120):
        """Execute an existing .py script file in Abaqus CAE."""
        start = time.time()
        proc = subprocess.run(
            ["powershell.exe", "-Command",
             f"cd '{self.work_dir}'; abaqus cae noGUI='{script_path}'"],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        duration = time.time() - start
        return AbaqusResult(
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            duration=duration,
        )

    def ping(self):
        """Quick check that Abaqus CAE is reachable."""
        result = self.execute(
            "__result__ = {'status': 'ok', 'msg': 'Abaqus bridge connected'}",
            timeout=30,
        )
        return result.success

    def _wrap_code(self, code, result_path, return_data):
        """Wrap user code to capture __result__ variable as JSON output."""
        result_path_escaped = result_path.replace("\\", "\\\\")
        header = (
            "import json, os, sys, traceback\n"
            "__result__ = None\n"
            "try:\n"
        )
        indented_code = "\n".join("    " + line for line in code.splitlines())
        if return_data:
            footer = (
                "\nexcept Exception as __e__:\n"
                "    def __safe_str__(s):\n"
                "        s = str(s)\n"
                "        try:\n"
                "            return s.decode('utf-8')\n"
                "        except (UnicodeDecodeError, AttributeError):\n"
                "            try:\n"
                "                return s.decode('gbk')\n"
                "            except (UnicodeDecodeError, AttributeError):\n"
                "                return s.decode('latin-1')\n"
                "    __result__ = {'error': True, 'message': __safe_str__(__e__), "
                "'traceback': __safe_str__(traceback.format_exc())}\n"
                "finally:\n"
                f"    __out_path__ = r'{result_path_escaped}'\n"
                "    if __result__ is not None:\n"
                "        with open(__out_path__, 'w') as __f__:\n"
                "            json.dump(__result__, __f__, indent=2, ensure_ascii=True)\n"
            )
        else:
            footer = (
                "\nexcept Exception as __e__:\n"
                "    print('ERROR: ' + str(__e__))\n"
                "    traceback.print_exc()\n"
            )
        return header + indented_code + footer


if __name__ == "__main__":
    bridge = AbaqusBridge()
    print("Pinging Abaqus CAE...")
    if bridge.ping():
        print("Connection successful!")
    else:
        print("Connection failed!")

    # Demo: get model info
    print("\nGetting model info...")
    result = bridge.execute("""
from abaqus import mdb
__result__ = {
    'models': list(mdb.models.keys()),
    'abaqus': 'connected'
}
""")
    print(f"Result: {result}")
    if result.result_data:
        print(f"Data: {json.dumps(result.result_data, indent=2)}")
