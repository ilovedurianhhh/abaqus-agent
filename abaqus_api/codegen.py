"""
Code accumulator for Abaqus Python 2.7 script generation.

All builder modules append code lines to a shared CodeBuffer.
The buffer is flushed as a single script via AbaqusBridge.execute().
"""


class CodeBuffer:
    """Accumulates Abaqus Python 2.7 code lines."""

    def __init__(self, model_name="Model-1"):
        self._lines = []
        self._header_emitted = False
        self._model_name = model_name

    def emit(self, code):
        """Append one or more lines of code to the buffer.

        Args:
            code: A string (may contain newlines) to append.
        """
        if not self._header_emitted:
            self._lines.append("from abaqus import *")
            self._lines.append("from abaqusConstants import *")
            self._lines.append("from caeModules import *")
            self._lines.append("import regionToolset")
            self._lines.append("")
            if self._model_name != "Model-1":
                self._lines.append(
                    f"mdb.models.changeKey(fromName='Model-1', "
                    f"toName='{self._model_name}')"
                )
                self._lines.append("")
            self._header_emitted = True
        self._lines.append(code)

    def emit_lines(self, lines):
        """Append multiple code lines."""
        for line in lines:
            self.emit(line)

    def preview(self):
        """Return the accumulated script as a string."""
        return "\n".join(self._lines)

    def line_count(self):
        """Return the number of accumulated lines."""
        return len(self._lines)

    def reset(self):
        """Clear all accumulated code."""
        self._lines.clear()
        self._header_emitted = False
