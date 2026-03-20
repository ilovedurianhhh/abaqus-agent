"""Tool definitions and execution for the Agent Harness.

Defines 4 tools that the LLM can invoke via function calling:
- search_abaqus_docs: RAG retrieval over Abaqus documentation
- get_simplified_api: Introspect simplified API method signatures
- validate_code: Safety + API signature validation
- submit_analysis: Execute code in Abaqus CAE
"""

import json
import logging
import traceback

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Tool definitions (OpenAI function-calling format)
# ──────────────────────────────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_abaqus_docs",
            "description": (
                "搜索 Abaqus 原生 Python API 文档。"
                "当简化 API 不支持某个功能（如接触、热传导、屈曲等）时使用。"
                "返回最相关的 API 用法和代码示例。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词，如 '接触分析 摩擦'、'heat transfer step'、'屈曲 BuckleStep'",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_simplified_api",
            "description": (
                "获取简化 API（abaqus_api 库）的方法签名列表。"
                "返回可用的 m.part.xxx()、m.load.xxx() 等方法及其参数。"
                "优先使用简化 API，只有简化 API 不支持时才用原生命令。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "module": {
                        "type": "string",
                        "description": "模块名: part, material, assembly, step, load, mesh, 或 all（获取全部）",
                        "enum": ["part", "material", "assembly", "step", "load", "mesh", "all"],
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_code",
            "description": (
                "校验 Abaqus Python 代码的安全性和 API 调用正确性。"
                "在生成完整代码后、最终输出前使用。"
                "返回错误列表，空列表表示通过。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "要校验的完整 Python 代码",
                    },
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_analysis",
            "description": (
                "提交完整的分析脚本到 Abaqus CAE 执行。"
                "代码必须是完整的可运行脚本（从 import 到 submit）。"
                "返回执行结果（成功时含 ODB 摘要）或错误信息。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "完整的 Python 分析脚本",
                    },
                },
                "required": ["code"],
            },
        },
    },
]


# ──────────────────────────────────────────────────────────────
# Tool execution
# ──────────────────────────────────────────────────────────────

def execute_tool(name, arguments_json, rag=None):
    """Execute a tool call and return the result as a string.

    Args:
        name: Tool function name.
        arguments_json: JSON string of tool arguments.
        rag: Optional AbaqusRAG instance for search_abaqus_docs.

    Returns:
        String result to feed back to the LLM.
    """
    try:
        args = json.loads(arguments_json) if isinstance(arguments_json, str) else arguments_json
    except json.JSONDecodeError:
        return f"Error: invalid JSON arguments: {arguments_json[:200]}"

    logger.info("Executing tool: %s(%s)", name, str(args)[:100])

    if name == "search_abaqus_docs":
        return _tool_search_docs(args, rag)
    elif name == "get_simplified_api":
        return _tool_get_api(args)
    elif name == "validate_code":
        return _tool_validate(args)
    elif name == "submit_analysis":
        return _tool_submit(args)
    else:
        return f"Error: unknown tool '{name}'"


def _tool_search_docs(args, rag):
    """Search Abaqus documentation via RAG."""
    query = args.get("query", "")
    if not query:
        return "Error: query is required"
    if rag is None:
        return "RAG is not available. Please use native Abaqus commands based on your knowledge."

    results = rag.retrieve(query, top_k=5)
    if not results:
        return f"No relevant documentation found for: {query}"

    parts = []
    for r in results:
        parts.append(f"## {r['title']} (distance={r['distance']:.3f})")
        parts.append(r["content"])
        parts.append("")
    return "\n".join(parts)


def _tool_get_api(args):
    """Get simplified API method signatures."""
    from .prompts import build_api_reference, _introspect_class, _BUILDERS

    module = args.get("module", "all")

    if module == "all":
        return build_api_reference()

    # Filter to specific module
    module_map = {b[2].split(".")[1]: b for b in _BUILDERS}  # "part" → ("abaqus_api.part", "PartBuilder", "m.part")
    if module not in module_map:
        return f"Unknown module: {module}. Available: {', '.join(module_map.keys())}, all"

    mod_name, cls_name, accessor = module_map[module]
    lines = [f"## {cls_name} ({accessor})"]
    lines.extend(_introspect_class(mod_name, cls_name, accessor))
    return "\n".join(lines)


def _tool_validate(args):
    """Validate code safety and API calls."""
    code = args.get("code", "")
    if not code:
        return "Error: code is required"

    from .agent import _validate_code, CodeValidationError
    from .code_validator import validate_api_calls

    errors = []

    # Safety validation
    try:
        _validate_code(code)
    except CodeValidationError as e:
        errors.append(f"Safety error: {e}")

    # API signature validation
    api_errors = validate_api_calls(code)
    errors.extend(api_errors)

    if errors:
        return "Validation FAILED:\n" + "\n".join(f"  - {e}" for e in errors)
    return "Validation PASSED: code is safe and API calls are correct."


def _tool_submit(args):
    """Execute analysis code in restricted namespace."""
    code = args.get("code", "")
    if not code:
        return "Error: code is required"

    # Safety check first
    from .agent import _validate_code, CodeValidationError
    try:
        _validate_code(code)
    except CodeValidationError as e:
        return f"Refused to execute: {e}"

    # Execute in restricted namespace
    namespace = {"__builtins__": __builtins__}
    try:
        exec(code, namespace)
    except Exception:
        return f"Execution FAILED:\n{traceback.format_exc()}"

    # Extract results
    parts = ["Execution succeeded."]

    if "result" in namespace and isinstance(namespace["result"], dict):
        r = namespace["result"]
        if r.get("error"):
            return f"Abaqus job failed: {r.get('message', 'unknown error')}"
        parts.append(f"Output directory: {r.get('output_dir', 'N/A')}")

    if "odb_summary" in namespace:
        odb = namespace["odb_summary"]
        if isinstance(odb, dict) and "error" not in odb:
            parts.append(f"Max von Mises Stress: {odb.get('max_mises', 0):.2f} MPa")
            parts.append(f"Max Displacement: {odb.get('max_displacement', 0):.4f} mm")
            parts.append(f"Max Reaction Force: {odb.get('max_rf_magnitude', 0):.2f} N")
        elif isinstance(odb, dict):
            parts.append(f"ODB read error: {odb.get('message', 'unknown')}")

    return "\n".join(parts)
