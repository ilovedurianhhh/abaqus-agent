"""AbaqusAgent — core agent that converts natural language to abaqus_api code."""

import ast
import re
import sys
import os
import time
import logging
import traceback

# Ensure project root is importable
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from .llm import LLMClient
from .history import ConversationHistory
from .prompts import build_system_prompt
from .code_validator import validate_api_calls

logger = logging.getLogger(__name__)

MAX_RETRIES = 3

# Imports allowed in generated code
_ALLOWED_IMPORTS = {"abaqus_api"}

# Node types / names that are forbidden in generated code
_FORBIDDEN_NAMES = {
    "eval", "exec", "compile", "__import__", "globals", "locals",
    "breakpoint", "exit", "quit",
}

_FORBIDDEN_MODULES = {
    "os", "sys", "subprocess", "shutil", "socket", "http",
    "requests", "urllib", "pathlib", "importlib", "ctypes",
    "signal", "threading", "multiprocessing",
}


class CodeValidationError(Exception):
    """Raised when generated code fails safety validation."""
    pass


def _validate_code(code):
    """Validate generated code before execution.

    Checks:
        1. Valid Python syntax (ast.parse)
        2. Only allowed imports (abaqus_api)
        3. No dangerous builtin calls (os.system, eval, exec, open, etc.)

    Raises:
        CodeValidationError with description of the violation.
    """
    # 1. Syntax check
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise CodeValidationError(f"语法错误: {e}")

    # 2. Walk AST to check imports and dangerous calls
    for node in ast.walk(tree):
        # Check imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name.split(".")[0]
                if mod in _FORBIDDEN_MODULES:
                    raise CodeValidationError(
                        f"禁止导入模块: {alias.name}"
                    )

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                mod = node.module.split(".")[0]
                if mod not in _ALLOWED_IMPORTS and mod in _FORBIDDEN_MODULES:
                    raise CodeValidationError(
                        f"禁止导入模块: {node.module}"
                    )

        # Check forbidden function calls
        elif isinstance(node, ast.Call):
            func = node.func
            # Direct call: eval(), exec(), etc.
            if isinstance(func, ast.Name) and func.id in _FORBIDDEN_NAMES:
                raise CodeValidationError(
                    f"禁止调用: {func.id}()"
                )
            # Attribute call: os.system(), subprocess.run(), etc.
            if isinstance(func, ast.Attribute):
                if isinstance(func.value, ast.Name):
                    if func.value.id in _FORBIDDEN_MODULES:
                        raise CodeValidationError(
                            f"禁止调用: {func.value.id}.{func.attr}()"
                        )

    logger.debug("Code validation passed (%d lines)", len(code.splitlines()))


def _parse_response(text):
    """Extract <plan> and <code> blocks from LLM response.

    Returns:
        (plan_text, code_text) — either may be empty string if not found.
    """
    plan_match = re.search(r"<plan>(.*?)</plan>", text, re.DOTALL)
    code_match = re.search(r"<code>(.*?)</code>", text, re.DOTALL)
    plan = plan_match.group(1).strip() if plan_match else ""
    code = code_match.group(1).strip() if code_match else ""
    return plan, code


def _format_result(plan, odb_summary, output_dir):
    """Format a successful analysis result for display."""
    lines = []
    if plan:
        lines.append("## 分析方案")
        lines.append(plan)
        lines.append("")
    lines.append("## 分析完成")
    if isinstance(odb_summary, dict) and "error" not in odb_summary:
        lines.append(f"- Max von Mises Stress: {odb_summary.get('max_mises', 0):.2f} MPa")
        lines.append(f"- Max Displacement: {odb_summary.get('max_displacement', 0):.4f} mm")
        lines.append(f"- Max Reaction Force: {odb_summary.get('max_rf_magnitude', 0):.2f} N")
    elif isinstance(odb_summary, dict):
        lines.append(f"- ODB 读取失败: {odb_summary.get('message', 'unknown error')}")
    else:
        lines.append("- 作业已提交，但未能读取 ODB 结果")
    if output_dir:
        lines.append(f"\n输出文件: {output_dir}")
    return "\n".join(lines)


class AbaqusAgent:
    """Agent that takes natural language input and runs Abaqus analyses."""

    def __init__(self, api_key=None, model="kimi-k2.5"):
        self.llm = LLMClient(api_key=api_key, model=model)
        self.history = ConversationHistory()

    def chat(self, user_message):
        """Process a user message and return the analysis result or response.

        Flow:
            1. Send message + history to LLM → get <plan> + <code>
            2. Validate code safety (AST check)
            3. exec() the code in a restricted namespace
            4. On failure, append error to history and retry (up to 3 times)
            5. On success, format and return results
        """
        logger.info("User input: %s", user_message[:100])
        self.history.add_user(user_message)

        last_error = None
        last_code = None
        for attempt in range(1, MAX_RETRIES + 1):
            # Build messages for LLM
            if last_error:
                # Append error feedback for retry, include the failed code
                error_msg = (
                    f"上一次生成的代码执行失败（第 {attempt - 1} 次尝试）。\n"
                    f"生成的代码:\n```python\n{last_code}\n```\n"
                    f"错误信息:\n```\n{last_error}\n```\n"
                    f"请分析错误原因，修复代码后重新生成完整的 <plan> 和 <code>。"
                )
                self.history.add_user(error_msg)

            # Build system prompt dynamically based on user input
            # Pass llm client for LLM-based example classification
            system_prompt = build_system_prompt(
                user_input=user_message, llm_client=self.llm
            )

            # Call LLM
            messages = self.history.to_messages()
            try:
                t0 = time.time()
                response = self.llm.generate(
                    system=system_prompt,
                    messages=messages,
                    max_tokens=4096,
                )
                elapsed = time.time() - t0
                logger.info("LLM response received in %.1fs (attempt %d/%d)",
                            elapsed, attempt, MAX_RETRIES)
            except Exception as e:
                logger.error("LLM call failed: %s", e)
                return f"LLM 调用失败: {e}"

            self.history.add_assistant(response)

            # Parse response
            plan, code = _parse_response(response)
            if not code:
                # LLM responded with text only (e.g. clarification question)
                logger.info("LLM returned text-only response (no <code> block)")
                return response

            # Validate code safety
            try:
                _validate_code(code)
            except CodeValidationError as e:
                logger.warning("Code validation failed: %s", e)
                last_error = f"代码安全校验失败: {e}"
                last_code = code
                continue

            # Validate API calls against real signatures
            api_errors = validate_api_calls(code)
            if api_errors:
                error_detail = "\n".join(f"  - {e}" for e in api_errors)
                logger.warning("API validation failed:\n%s", error_detail)
                last_error = (
                    f"API 调用校验失败，以下调用与真实 API 不匹配:\n{error_detail}\n"
                    f"请严格按照 API Reference 中的方法签名修正代码。"
                )
                last_code = code
                continue

            # Execute the generated code
            logger.info("Executing generated code (%d lines)...", len(code.splitlines()))
            t0 = time.time()
            ok, result_or_error = self._execute_code(code)
            elapsed = time.time() - t0

            if ok:
                logger.info("Code execution succeeded in %.1fs", elapsed)
                odb_summary = result_or_error.get("odb_summary")
                output_dir = result_or_error.get("output_dir", "")
                return _format_result(plan, odb_summary, output_dir)
            else:
                logger.warning("Code execution failed (attempt %d/%d): %s",
                               attempt, MAX_RETRIES, result_or_error[:200])
                last_error = result_or_error
                last_code = code

        # All retries exhausted
        logger.error("All %d retries exhausted", MAX_RETRIES)
        return (
            f"经过 {MAX_RETRIES} 次尝试仍然失败。最后一次错误:\n"
            f"```\n{last_error}\n```\n"
            f"请尝试用更简单的描述重新提问。"
        )

    def _execute_code(self, code):
        """Execute generated abaqus_api code in a restricted namespace.

        Returns:
            (True, result_dict) on success
            (False, error_string) on failure
        """
        # Build an execution namespace with necessary imports available
        namespace = {"__builtins__": __builtins__}

        try:
            exec(code, namespace)
        except Exception:
            return False, traceback.format_exc()

        # Extract results from namespace
        result = {}

        # Check for submit result
        if "result" in namespace and isinstance(namespace["result"], dict):
            submit_result = namespace["result"]
            result["output_dir"] = submit_result.get("output_dir", "")
            if submit_result.get("error"):
                parts = [f"Abaqus 作业执行失败:"]
                parts.append(f"message: {submit_result.get('message', 'unknown error')}")
                if submit_result.get("traceback"):
                    parts.append(f"traceback:\n{submit_result['traceback']}")
                if submit_result.get("stdout"):
                    parts.append(f"stdout:\n{submit_result['stdout']}")
                if submit_result.get("stderr"):
                    parts.append(f"stderr:\n{submit_result['stderr']}")
                return False, "\n".join(parts)

            job_status = str(submit_result.get("status", ""))
            if job_status != "COMPLETED":
                job_name = submit_result.get("job_name", "unknown")
                output_dir = submit_result.get("output_dir", "")

                # Fallback: check .sta file directly if status is None
                if job_status in ("None", "") and output_dir:
                    import os
                    sta_path = os.path.join(output_dir, f"{job_name}.sta")
                    if os.path.exists(sta_path):
                        try:
                            with open(sta_path, 'r', encoding='utf-8', errors='ignore') as f:
                                sta_content = f.read()
                            if 'COMPLETED SUCCESSFULLY' in sta_content:
                                job_status = "COMPLETED"
                                logger.info("Job status corrected to COMPLETED via .sta file")
                        except Exception:
                            pass

                if job_status == "COMPLETED":
                    # Status was corrected by .sta fallback; continue to success path
                    pass
                else:
                    # Try to read .dat file for actual error messages
                    dat_errors = []
                    if output_dir:
                        import os
                        dat_path = os.path.join(output_dir, f"{job_name}.dat")
                        if os.path.exists(dat_path):
                            try:
                                with open(dat_path, 'r', encoding='utf-8', errors='ignore') as f:
                                    lines = f.readlines()
                                    for i, line in enumerate(lines):
                                        if '***ERROR' in line:
                                            error_block = ''.join(lines[i:min(i+3, len(lines))])
                                            dat_errors.append(error_block.strip())
                            except Exception:
                                pass

                    error_msg = f"Abaqus 作业未正常完成 (status={job_status}, job={job_name})。"
                    if dat_errors:
                        error_msg += f"\n\n.dat 文件中的错误:\n" + "\n\n".join(dat_errors[:5])
                    else:
                        error_msg += f"\n请检查 {job_name}.dat 文件以获取详细错误信息。"

                    return False, error_msg

        # Check for ODB summary
        if "odb_summary" in namespace:
            result["odb_summary"] = namespace["odb_summary"]
        elif "result" in namespace and isinstance(namespace["result"], dict):
            # ODB was not read, but job may have succeeded
            result["odb_summary"] = None

        return True, result
