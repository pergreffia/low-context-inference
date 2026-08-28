"""Structural validation of chat-completions payloads (M0–M6 review P1).

Minimal, OpenAI-compatible structural checks: malformed SHAPES must become
client errors (400) instead of internal 500s, while unknown fields and
unknown content-part types stay untouched (transparency, M6 §13.1).

Only structure is validated — semantics (which roles are legal, whether a
tool_call matches a tool definition) remain the upstream endpoint's business.
"""

from __future__ import annotations

from typing import Any


class PayloadValidationError(ValueError):
    """Raised for structurally invalid request payloads (maps to 400)."""


def _reject(message: str) -> PayloadValidationError:
    return PayloadValidationError(message)


def validate_chat_payload(payload: dict[str, Any]) -> None:
    """Validate the structural subset the proxy relies upon.

    Raises PayloadValidationError on malformed shapes; returns silently
    otherwise. Absent optional keys are fine; present-but-wrong-typed keys
    are client errors.
    """
    if not isinstance(payload, dict):
        raise _reject("request body must be a JSON object")

    messages = payload.get("messages")
    if not isinstance(messages, list) or isinstance(messages, bool):
        raise _reject("'messages' must be an array of message objects")
    if not messages:
        raise _reject("'messages' must not be empty")
    for index, message in enumerate(messages):
        _validate_message(message, index)

    tools = payload.get("tools")
    if tools is not None:
        if not isinstance(tools, list):
            raise _reject("'tools' must be an array when present")
        for index, tool in enumerate(tools):
            _validate_tool(tool, f"tools[{index}]")

    stream = payload.get("stream")
    if stream is not None and not isinstance(stream, bool):
        raise _reject("'stream' must be a boolean when present")

    n = payload.get("n")
    if n is not None:
        if isinstance(n, bool) or not isinstance(n, int):
            raise _reject("'n' must be an integer when present")
        if n > 1:
            raise _reject("only n=1 is supported by this proxy")


def _validate_tool(tool: Any, where: str) -> None:
    if not isinstance(tool, dict):
        raise _reject(f"{where} must be an object")
    tool_type = tool.get("type")
    if not isinstance(tool_type, str) or not tool_type:
        raise _reject(f"{where}.type must be a non-empty string")
    if tool_type != "function" and tool_type != "custom":
        return
    container_key = "function" if tool_type == "function" else "custom"
    container = tool.get(container_key)
    if not isinstance(container, dict):
        raise _reject(f"{where}.{container_key} must be an object for {tool_type} tools")
    name = container.get("name")
    if not isinstance(name, str) or not name:
        raise _reject(f"{where}.{container_key}.name must be a non-empty string")
    description = container.get("description")
    if description is not None and not isinstance(description, str):
        raise _reject(f"{where}.{container_key}.description must be a string when present")
    parameters = function_parameters_or_none(tool_type, container)
    if parameters is not None and not isinstance(parameters, dict):
        raise _reject(f"{where}.{container_key}.parameters must be an object when present")


def function_parameters_or_none(tool_type: str, container: dict) -> Any:
    return container.get("parameters") if tool_type == "function" else None


def _validate_tool_calls(tool_calls: Any, where: str) -> None:
    for index, call in enumerate(tool_calls):
        call_where = f"{where}[{index}]"
        if not isinstance(call, dict):
            raise _reject(f"{call_where} must be an object")
        call_id = call.get("id")
        if call_id is not None and not isinstance(call_id, str):
            raise _reject(f"{call_where}.id must be a string when present")
        call_type = call.get("type")
        if isinstance(call_type, str) and call_type not in ("function", "custom"):
            continue
        if call_type == "custom":
            container, key = call.get("custom"), "custom"
        elif call_type == "function":
            container, key = call.get("function"), "function"
        elif "function" in call:
            container, key = call["function"], "function"
        elif "custom" in call:
            container, key = call["custom"], "custom"
        else:
            raise _reject(f"{call_where} must carry a 'function' or 'custom' payload")
        if not isinstance(container, dict):
            raise _reject(f"{call_where}.{key} must be an object")
        name = container.get("name")
        if not isinstance(name, str) or not name:
            raise _reject(f"{call_where}.{key}.name must be a non-empty string")
        arguments_or_input = container.get("arguments") if key == "function" else container.get("input")
        if arguments_or_input is not None and not isinstance(arguments_or_input, str):
            raise _reject(f"{call_where}.{key}.{'arguments' if key == 'function' else 'input'} must be a string when present")


def _validate_message(message: Any, index: int) -> None:
    where = f"messages[{index}]"
    if not isinstance(message, dict):
        raise _reject(f"{where} must be an object")
    role = message.get("role")
    if not isinstance(role, str) or not role:
        raise _reject(f"{where}.role must be a non-empty string")
    if "content" in message:
        content = message["content"]
        if content is not None and not isinstance(content, str | list):
            raise _reject(f"{where}.content must be a string, array, or null")
        if isinstance(content, list):
            for part_index, part in enumerate(content):
                if not isinstance(part, dict):
                    raise _reject(f"{where}.content[{part_index}] must be an object")
    tool_calls = message.get("tool_calls")
    if tool_calls is not None:
        if not isinstance(tool_calls, list):
            raise _reject(f"{where}.tool_calls must be an array when present")
        _validate_tool_calls(tool_calls, where)
