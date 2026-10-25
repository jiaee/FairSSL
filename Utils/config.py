import ast
import os
from pathlib import Path
from types import SimpleNamespace


def _parse_scalar(value):
    value = value.strip()
    if value in {"", "null", "None", "~"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return os.path.expandvars(value[1:-1])
    try:
        return ast.literal_eval(value)
    except Exception:
        return os.path.expandvars(value)


def load_yaml_config(path):
    root = {}
    stack = [(-1, root)]
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        key, sep, value = raw.strip().partition(":")
        if not sep:
            raise ValueError(f"Invalid config line: {raw}")
        while indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip():
            parent[key] = _parse_scalar(value)
        else:
            child = {}
            parent[key] = child
            stack.append((indent, child))
    return root


def to_namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{k: to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [to_namespace(v) for v in value]
    return value


def is_configured(value):
    if value is None:
        return False
    if isinstance(value, str):
        value = value.strip()
        return bool(value) and not (value.startswith("${") and value.endswith("}"))
    return True
