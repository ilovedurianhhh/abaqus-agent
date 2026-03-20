"""Post-generation code validator — check generated code against real API signatures.

Parses generated code with AST, extracts all m.xxx.yyy() calls, and validates:
1. Method exists in the real API
2. Required parameters are present
3. No unknown parameters passed
4. Known constraint rules (e.g., pressure uses surface, fix uses set)

Returns a list of warnings/errors that get fed back to the LLM for self-correction.
"""

import ast
import inspect
import importlib
import functools
import logging
import re

logger = logging.getLogger(__name__)

# Builder mapping: attribute name → (module, class)
_BUILDER_MAP = {
    "part":     ("abaqus_api.part", "PartBuilder"),
    "material": ("abaqus_api.material", "MaterialBuilder"),
    "assembly": ("abaqus_api.assembly", "AssemblyBuilder"),
    "step":     ("abaqus_api.step", "StepBuilder"),
    "load":     ("abaqus_api.load", "LoadBuilder"),
    "mesh":     ("abaqus_api.mesh", "MeshBuilder"),
    "odb":      ("abaqus_api.odb", "OdbReader"),
}

# Semantic constraint rules — common mistakes to catch
_CONSTRAINT_RULES = [
    {
        "method": ("load", "pressure"),
        "forbidden_params": ["set_name"],
        "required_params": ["surface"],
        "message": "pressure() 必须用 surface 参数，不能用 set_name。"
                   "面需要用 m.part.create_surface() 创建。",
    },
    {
        "method": ("load", "fix"),
        "forbidden_params": ["surface"],
        "required_params": ["set_name"],
        "message": "fix() 必须用 set_name 参数，不能用 surface。"
                   "集合需要用 m.part.create_face_set() 创建。",
    },
    {
        "method": ("load", "concentrated_force"),
        "forbidden_params": ["surface"],
        "required_params": ["set_name"],
        "message": "concentrated_force() 必须用 set_name 参数，不能用 surface。"
                   "建议改用 m.load.concentrated_force_at_point() 直接指定坐标，"
                   "它会自动在网格中找到最近的节点。",
    },
]


@functools.lru_cache(maxsize=1)
def _build_api_signatures():
    """Introspect all builder classes and return a dict of method signatures.

    Returns:
        {
            ("part", "create_sketch"): {
                "required": ["name"],
                "optional": {"sheet_size": 200.0},
                "all_params": ["name", "sheet_size"],
            },
            ...
        }
    """
    signatures = {}

    for attr_name, (module_name, class_name) in _BUILDER_MAP.items():
        try:
            mod = importlib.import_module(module_name)
        except ImportError:
            continue
        cls = getattr(mod, class_name, None)
        if cls is None:
            continue

        for method_name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            if method_name.startswith("_"):
                continue

            sig = inspect.signature(method)
            required = []
            optional = {}
            for p in sig.parameters.values():
                if p.name == "self":
                    continue
                if p.default is inspect.Parameter.empty:
                    required.append(p.name)
                else:
                    optional[p.name] = p.default

            signatures[(attr_name, method_name)] = {
                "required": required,
                "optional": optional,
                "all_params": required + list(optional.keys()),
            }

    # Add AbaqusModel top-level methods
    try:
        from abaqus_api.model import AbaqusModel
        for method_name in ("submit", "preview", "reset"):
            method = getattr(AbaqusModel, method_name, None)
            if method is None:
                continue
            sig = inspect.signature(method)
            required = []
            optional = {}
            for p in sig.parameters.values():
                if p.name == "self":
                    continue
                if p.default is inspect.Parameter.empty:
                    required.append(p.name)
                else:
                    optional[p.name] = p.default
            signatures[("model", method_name)] = {
                "required": required,
                "optional": optional,
                "all_params": required + list(optional.keys()),
            }
    except ImportError:
        pass

    return signatures


def _ast_str_value(node):
    """Extract string value from an AST node, or return None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _extract_api_calls(code):
    """Parse code and extract all m.xxx.yyy(...) style API calls.

    Returns:
        List of (builder_name, method_name, {param_name: ...}, line_number, args_list)
        where args_list is the raw list of ast argument nodes.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        func = node.func
        builder = None
        method = None
        lineno = getattr(node, "lineno", 0)

        # Pattern: m.part.create_sketch(...)
        if (isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Attribute)
                and isinstance(func.value.value, ast.Name)
                and func.value.value.id == "m"):
            builder = func.value.attr
            method = func.attr

        # Pattern: m.submit(...)
        elif (isinstance(func, ast.Attribute)
              and isinstance(func.value, ast.Name)
              and func.value.id == "m"):
            builder = "model"
            method = func.attr

        if builder and method:
            # Extract keyword argument names
            kwargs = {}
            for kw in node.keywords:
                if kw.arg:
                    kwargs[kw.arg] = True
            # Count positional args
            n_positional = len(node.args)
            calls.append((builder, method, kwargs, n_positional, lineno, node.args, node.keywords))

    return calls


def _uses_native_abaqus_code(code):
    """Detect whether the code uses native Abaqus commands (mdb.models[...]).

    Returns True if native Abaqus scripting commands are found.
    """
    # Common patterns for native Abaqus Python scripting
    native_patterns = [
        r'mdb\.models\[',
        r'mdb\.Model\(',
        r'm\._buf\.emit\(',
        r'session\.',
        r'from abaqus import',
        r'from abaqusConstants import',
    ]
    for pat in native_patterns:
        if re.search(pat, code):
            return True
    return False


def validate_api_calls(code):
    """Validate all API calls in generated code against real signatures.

    When native Abaqus commands are detected (mdb.models[...], m._buf.emit),
    only simplified-API calls are validated — native commands are passed
    through with basic safety checks only.

    Args:
        code: Generated Python code string.

    Returns:
        List of error/warning strings. Empty list means code is valid.
    """
    has_native = _uses_native_abaqus_code(code)
    signatures = _build_api_signatures()
    calls = _extract_api_calls(code)
    errors = []

    for builder, method, kwargs, n_positional, lineno, args, keywords in calls:
        key = (builder, method)

        # 1. Check method exists
        if key not in signatures:
            # Skip AbaqusModel constructor and known non-builder calls
            if method in ("AbaqusModel", "max_values", "field_output"):
                continue
            # When native Abaqus code is present, skip unknown method errors
            # (the LLM may be using m._buf.emit or other hybrid patterns)
            if has_native:
                continue
            if builder in _BUILDER_MAP or builder == "model":
                errors.append(
                    f"第{lineno}行: m.{builder}.{method}() 方法不存在。"
                    f"请检查 API 文档中 {builder} 的可用方法。"
                )
            continue

        sig = signatures[key]

        # 2. Check for unknown keyword arguments
        for param_name in kwargs:
            if param_name not in sig["all_params"]:
                errors.append(
                    f"第{lineno}行: m.{builder}.{method}() 没有参数 '{param_name}'。"
                    f"可用参数: {', '.join(sig['all_params'])}"
                )

        # 3. Check required parameters are covered
        # Positional args fill required params in order
        uncovered_required = sig["required"][n_positional:]
        for req in uncovered_required:
            if req not in kwargs:
                errors.append(
                    f"第{lineno}行: m.{builder}.{method}() 缺少必需参数 '{req}'。"
                    f"必需参数: {', '.join(sig['required'])}"
                )

    # 4. Check semantic constraint rules
    for rule in _CONSTRAINT_RULES:
        rule_builder, rule_method = rule["method"]
        for builder, method, kwargs, n_positional, lineno, args, keywords in calls:
            if builder == rule_builder and method == rule_method:
                for forbidden in rule.get("forbidden_params", []):
                    if forbidden in kwargs:
                        errors.append(f"第{lineno}行: {rule['message']}")

    # 5. Check concentrated_force set_name was created with correct method
    #    (must use create_vertex_set or create_node_set_by_bounding_box, NOT
    #     create_set_by_bounding_box or create_face_set)
    cf_sets = {}  # set_name -> lineno
    for builder, method, kwargs, n_positional, lineno, args, keywords in calls:
        if builder == "load" and method == "concentrated_force":
            set_name = None
            # Check keyword arg
            for kw in keywords:
                if kw.arg == "set_name":
                    set_name = _ast_str_value(kw.value)
            # Check 4th positional arg (index 3) if not found as keyword
            if set_name is None and len(args) >= 4:
                set_name = _ast_str_value(args[3])
            if set_name:
                cf_sets[set_name] = lineno

    if cf_sets:
        # Collect sets created by correct methods (vertex/node sets)
        vertex_sets = set()
        for builder, method, kwargs, n_positional, lineno, args, keywords in calls:
            if builder == "part" and method in ("create_vertex_set", "create_node_set_by_bounding_box"):
                sn = None
                for kw in keywords:
                    if kw.arg == "set_name":
                        sn = _ast_str_value(kw.value)
                if sn is None and len(args) >= 2:
                    sn = _ast_str_value(args[1])
                if sn:
                    vertex_sets.add(sn)

        # Collect sets created by wrong methods (cell/face sets)
        wrong_sets = {}  # set_name -> method_name
        for builder, method, kwargs, n_positional, lineno, args, keywords in calls:
            if builder == "part" and method in ("create_set_by_bounding_box", "create_face_set"):
                sn = None
                for kw in keywords:
                    if kw.arg == "set_name":
                        sn = _ast_str_value(kw.value)
                if sn is None and len(args) >= 2:
                    sn = _ast_str_value(args[1])
                if sn:
                    wrong_sets[sn] = method

        for sn, ln in cf_sets.items():
            if sn in vertex_sets:
                continue
            if sn in wrong_sets:
                errors.append(
                    f"第{ln}行: 集中力 '{sn}' 的加载点集合使用了 {wrong_sets[sn]}() 创建，"
                    f"这是错误的。必须使用 m.part.create_vertex_set() 或 "
                    f"m.part.create_node_set_by_bounding_box() 创建顶点/节点集合。"
                )
            else:
                errors.append(
                    f"第{ln}行: 集中力 '{sn}' 的加载点集合未正确创建。"
                    f"必须使用 m.part.create_vertex_set() 创建顶点集合，"
                    f"不能使用 create_set_by_bounding_box() 或 create_face_set()。"
                )

    if errors:
        logger.warning("API validation found %d issues", len(errors))
    else:
        logger.debug("API validation passed for %d calls", len(calls))

    return errors
