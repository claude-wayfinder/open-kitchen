"""
@tool Decorator — Auto-Generate OpenAI Function-Calling Schemas
Extracted from OpenMythos (Build Small Hackathon 2026).

Pattern: Write a normal Python function with type hints and a docstring.
The @tool decorator generates the OpenAI function-calling schema automatically.
Supports str/int/float/bool/list[X]/Optional[X]/Enum. Code IS the schema.
"""

from __future__ import annotations

import inspect
import json
from enum import Enum
from typing import Any, Callable, get_args, get_origin, Union


# ---------------------------------------------------------------------------
# 1. TOOL REGISTRY — Stores decorated functions and their schemas
# ---------------------------------------------------------------------------

_TOOL_REGISTRY: dict[str, dict] = {}


def get_tools() -> list[dict]:
    """Return all registered tools as OpenAI function-calling schema objects."""
    return [
        {"type": "function", "function": schema}
        for schema in _TOOL_REGISTRY.values()
    ]


def get_tool_map() -> dict[str, Callable]:
    """Return name -> callable map for dispatching tool calls."""
    return {name: schema["_callable"] for name, schema in _TOOL_REGISTRY.items()}


# ---------------------------------------------------------------------------
# 2. TYPE MAPPING — Python types to JSON Schema types
# ---------------------------------------------------------------------------

def _python_type_to_json_schema(annotation) -> dict:
    """Convert a Python type hint to a JSON Schema fragment."""

    # Handle None / missing annotations
    if annotation is inspect.Parameter.empty or annotation is None:
        return {"type": "string"}

    # Handle Enum subclasses
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return {"type": "string", "enum": [e.value for e in annotation]}

    # Handle basic types
    type_map = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        dict: {"type": "object"},
        list: {"type": "array"},
    }
    if annotation in type_map:
        return type_map[annotation]

    # Handle generic types: list[str], dict[str, int], Optional[str], etc.
    origin = get_origin(annotation)
    args = get_args(annotation)

    # list[X] -> {"type": "array", "items": X_schema}
    if origin is list and args:
        return {"type": "array", "items": _python_type_to_json_schema(args[0])}

    # Optional[X] is Union[X, None] — just return X's schema
    if origin is Union:
        non_none = [a for a in args if a is not type(None)]
        if non_none:
            return _python_type_to_json_schema(non_none[0])

    return {"type": "string"}


# ---------------------------------------------------------------------------
# 3. DOCSTRING PARSER — Extract parameter descriptions (Google-style)
# ---------------------------------------------------------------------------

def _parse_param_docs(docstring: str | None) -> tuple[str, dict[str, str]]:
    """Extract function description and per-param descriptions from docstring."""
    if not docstring:
        return "", {}
    lines = docstring.strip().split("\n")
    desc_lines, param_docs, in_args = [], {}, False
    for line in lines:
        s = line.strip()
        if s.lower().startswith(("args:", "parameters:", "params:")):
            in_args = True; continue
        if s.lower().startswith(("returns:", "raises:", "yields:")):
            in_args = False; continue
        if in_args and ":" in s:
            name, desc = s.split(":", 1)
            param_docs[name.strip()] = desc.strip()
        elif not in_args and s:
            desc_lines.append(s)
    return " ".join(desc_lines), param_docs


# ---------------------------------------------------------------------------
# 4. THE DECORATOR — Where it all comes together
# ---------------------------------------------------------------------------

def tool(fn: Callable) -> Callable:
    """Decorator that registers a function as an OpenAI-compatible tool."""
    sig = inspect.signature(fn)
    description, param_docs = _parse_param_docs(fn.__doc__)
    properties, required = {}, []
    for name, param in sig.parameters.items():
        schema = _python_type_to_json_schema(param.annotation)
        if name in param_docs:
            schema["description"] = param_docs[name]
        properties[name] = schema
        if param.default is inspect.Parameter.empty:
            required.append(name)

    _TOOL_REGISTRY[fn.__name__] = {
        "name": fn.__name__,
        "description": description or f"Call the {fn.__name__} function",
        "parameters": {"type": "object", "properties": properties, "required": required},
        "_callable": fn,
    }
    return fn


# ---------------------------------------------------------------------------
# 5. DISPATCH — Execute a tool call from the model's response
# ---------------------------------------------------------------------------

def dispatch_tool_call(tool_call: dict) -> str:
    """
    Execute a tool call returned by the model.

    Args:
        tool_call: {"name": "search", "arguments": '{"query": "hello"}'}

    Returns:
        String result to feed back to the model.
    """
    name = tool_call["name"]
    fn_map = get_tool_map()

    if name not in fn_map:
        return json.dumps({"error": f"Unknown tool: {name}"})

    args = tool_call.get("arguments", "{}")
    if isinstance(args, str):
        args = json.loads(args)

    try:
        result = fn_map[name](**args)
        return json.dumps(result) if not isinstance(result, str) else result
    except Exception as e:
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# EXAMPLE USAGE
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    class Priority(Enum):
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"

    @tool
    def create_task(title: str, description: str,
                    priority: Priority = Priority.MEDIUM,
                    tags: list[str] | None = None) -> dict:
        """Create a new task in the project board.

        Args:
            title: Short title for the task
            description: Detailed description of what needs to be done
            priority: Task priority level
            tags: Optional list of tags to categorize the task
        """
        return {"created": True, "title": title, "priority": priority.value}

    @tool
    def search_docs(query: str, max_results: int = 10) -> str:
        """Search the documentation.

        Args:
            query: Search query string
            max_results: Maximum number of results
        """
        return f"Found results for: {query}"

    # Print the generated schemas
    tools = get_tools()
    print(json.dumps(tools, indent=2, default=str))

    # Simulate dispatching a tool call from the model
    result = dispatch_tool_call({
        "name": "create_task",
        "arguments": json.dumps({
            "title": "Fix the bug",
            "description": "The login page crashes on mobile",
            "priority": "high",
        }),
    })
    print(f"\nTool result: {result}")
