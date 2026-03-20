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
from .prompts import build_system_prompt, build_agent_system_prompt
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

    Supports multiple code formats:
        1. <code>...</code>  (preferred)
        2. ```python...```   (common LLM fallback)
        3. ```...```         (generic code block)

    Returns:
        (plan_text, code_text) — either may be empty string if not found.
    """
    plan_match = re.search(r"<plan>(.*?)</plan>", text, re.DOTALL)
    plan = plan_match.group(1).strip() if plan_match else ""

    # Try <code> tags first (preferred format)
    code_match = re.search(r"<code>(.*?)</code>", text, re.DOTALL)
    if code_match:
        return plan, code_match.group(1).strip()

    # Fallback: ```python ... ```
    code_match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if code_match:
        return plan, code_match.group(1).strip()

    # Fallback: generic ``` ... ```
    code_match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if code_match:
        code = code_match.group(1).strip()
        # Only accept if it looks like Python code
        if "import" in code or "AbaqusModel" in code or "m._buf" in code or "mdb.models" in code:
            return plan, code

    return plan, ""


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


class TaskRouter:
    """Embedding-based router that decides pipeline vs agent mode.

    Pre-computes embeddings for "complex task" reference descriptions (things
    that need native Abaqus API). If a user query is semantically close to
    any reference → agent mode. Otherwise → pipeline.

    Single-threshold design: avoids the ambiguity of comparing two overlapping
    reference sets.
    """

    # Reference descriptions for tasks that NEED native Abaqus API
    _COMPLEX_REFS = [
        "接触分析 摩擦 两个零件之间的面面接触 surface to surface contact friction",
        "热传导分析 温度场 稳态传热 heat transfer thermal conductivity temperature",
        "屈曲分析 临界载荷 失稳 buckling critical load stability",
        "螺栓预紧力 bolt pretension load tightening",
        "弹簧单元 阻尼器 spring dashpot element",
        "刚体约束 rigid body constraint",
        "耦合约束 分布耦合 coupling distributing constraint",
        "多点约束 MPC multi-point constraint",
        "热力耦合 温度位移耦合 thermal mechanical coupled",
        "内聚力单元 分层 脱粘 cohesive delamination",
        "子模型 submodel global local",
        "连接器 铰接 connector hinge",
        "热膨胀 热应力 thermal expansion stress",
        "绑定约束 两个零件贴合 tie constraint bond",
        "幅值 载荷随时间变化 amplitude time varying",
        "通用接触 碰撞 general contact collision",
        "温度边界条件 对流换热 convection film condition",
        "压杆稳定性 细长柱被压弯 column buckling stability slender",
        "摩擦系数 法向接触 tangential friction penalty",
        "齿轮啮合 传动 gear mesh engagement contact",
    ]

    # If embedding distance < this → definitely AGENT
    _THRESHOLD_STRONG = 0.71
    # If embedding distance < this AND keyword matches → also AGENT
    _THRESHOLD_WEAK = 0.90

    # Fallback keywords for when embedding is uncertain
    _NATIVE_KEYWORDS = {
        "接触", "摩擦", "热传导", "温度场", "导热", "传热", "屈曲", "失稳",
        "临界载荷", "螺栓", "预紧", "弹簧单元", "阻尼器", "刚体约束",
        "耦合约束", "多点约束", "MPC", "内聚力", "分层", "脱粘", "子模型",
        "连接器", "热膨胀", "热力耦合", "热应力", "绑定约束", "贴合",
        "碰撞", "啮合", "压弯", "压杆", "贴在一起", "温度",
        "cohesive", "contact", "buckling", "heat transfer", "tie",
    }

    def __init__(self, embedding_fn):
        """Pre-compute reference embeddings.

        Args:
            embedding_fn: A callable that takes a list of strings and returns
                a list of embedding vectors (list of floats).
        """
        self._embed = embedding_fn
        self._complex_vecs = self._embed(self._COMPLEX_REFS)
        logger.info("TaskRouter initialized: %d complex refs", len(self._complex_vecs))

    def is_complex(self, user_input):
        """Return True if the task should go to agent mode.

        Two-tier decision:
        1. Embedding distance < strong threshold → AGENT (confident)
        2. Embedding distance < weak threshold AND keyword match → AGENT
        3. Otherwise → PIPELINE
        """
        query_vec = self._embed([user_input])[0]
        min_dist = min(self._l2(query_vec, v) for v in self._complex_vecs)

        # Tier 1: strong embedding match
        if min_dist < self._THRESHOLD_STRONG:
            logger.info("TaskRouter: dist=%.3f < %.2f → AGENT (embedding)",
                         min_dist, self._THRESHOLD_STRONG)
            return True

        # Tier 2: weaker embedding + keyword confirmation
        has_keyword = any(kw in user_input.lower() for kw in self._NATIVE_KEYWORDS)
        if min_dist < self._THRESHOLD_WEAK and has_keyword:
            logger.info("TaskRouter: dist=%.3f < %.2f + keyword → AGENT (hybrid)",
                         min_dist, self._THRESHOLD_WEAK)
            return True

        logger.info("TaskRouter: dist=%.3f, keyword=%s → PIPELINE",
                     min_dist, has_keyword)
        return False

    @staticmethod
    def _l2(a, b):
        """L2 distance between two vectors."""
        return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


class AbaqusAgent:
    """Agent that takes natural language input and runs Abaqus analyses."""

    def __init__(self, api_key=None, model="kimi-k2.5"):
        self.llm = LLMClient(api_key=api_key, model=model)
        self.history = ConversationHistory()

        # Initialize RAG if index exists (graceful degradation)
        self.rag = None
        try:
            from .rag import AbaqusRAG
            rag = AbaqusRAG()
            if rag.is_indexed():
                self.rag = rag
                logger.info("RAG enabled with %d documents", rag.count())
            else:
                logger.info("RAG index is empty — running without RAG")
        except ImportError:
            logger.info("chromadb not installed — running without RAG")
        except Exception as e:
            logger.warning("RAG initialization failed: %s", e)

        # Initialize embedding-based task router (reuses RAG's embedding model)
        self.router = None
        if self.rag is not None:
            try:
                self.router = TaskRouter(embedding_fn=self.rag._ef)
            except Exception as e:
                logger.warning("TaskRouter initialization failed: %s", e)

        # Initialize agent harness for complex tasks
        self.harness = None
        try:
            from .harness import AgentHarness
            self.harness = AgentHarness(llm=self.llm, rag=self.rag)
            logger.info("Agent harness initialized")
        except Exception as e:
            logger.warning("Harness initialization failed: %s", e)

    def _is_complex_task(self, user_input):
        """Detect whether the task requires native Abaqus API (agent mode).

        Uses embedding similarity when available, falls back to keyword matching.
        """
        # Embedding-based routing (semantic understanding)
        if self.router:
            return self.router.is_complex(user_input)

        # Fallback: keyword matching
        _native_kw = {
            "接触", "摩擦", "热传导", "温度场", "导热", "稳态传热", "瞬态传热",
            "屈曲", "失稳", "临界载荷", "螺栓", "预紧", "弹簧单元", "阻尼器",
            "刚体约束", "耦合约束", "多点约束", "内聚力", "分层", "脱粘",
            "子模型", "连接器", "热膨胀", "热力耦合", "热应力",
            "cohesive", "contact", "buckling", "heat transfer",
        }
        lower = user_input.lower()
        return any(kw in lower for kw in _native_kw)

    def chat(self, user_message):
        """Route user message to pipeline or agent mode.

        Simple tasks (covered by simplified API) → pipeline (_pipeline_chat)
        Complex tasks (need native Abaqus API) → agent loop (_agent_chat)
        """
        logger.info("User input: %s", user_message[:100])

        if self.harness and self._is_complex_task(user_message):
            logger.info("Routing to AGENT mode (complex task detected)")
            return self._agent_chat(user_message)
        else:
            logger.info("Routing to PIPELINE mode")
            return self._pipeline_chat(user_message)

    def _agent_chat(self, user_message):
        """Handle complex tasks via the agent harness (tool-use loop)."""
        self.history.add_user(user_message)

        system_prompt = build_agent_system_prompt()

        response = self.harness.run(user_message, system_prompt)
        self.history.add_assistant(response)

        # Try to extract and validate code from the response
        plan, code = _parse_response(response)
        if code:
            # Run through the same validation + execution pipeline
            try:
                _validate_code(code)
            except CodeValidationError as e:
                return response  # Return the raw response, user can see the code

            api_errors = validate_api_calls(code)
            if not api_errors:
                ok, result_or_error = self._execute_code(code)
                if ok:
                    odb_summary = result_or_error.get("odb_summary")
                    output_dir = result_or_error.get("output_dir", "")
                    return _format_result(plan, odb_summary, output_dir)

        # Return the raw LLM response (may contain <plan>+<code> for user to review)
        return response

    def _pipeline_chat(self, user_message):
        """Handle simple tasks via the original pipeline (single-shot generation).

        Flow:
            1. Send message + history to LLM → get <plan> + <code>
            2. Validate code safety (AST check)
            3. exec() the code in a restricted namespace
            4. On failure, append error to history and retry (up to 3 times)
            5. On success, format and return results
        """
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
                user_input=user_message, llm_client=self.llm, rag=self.rag
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
