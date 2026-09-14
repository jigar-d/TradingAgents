"""LangChain adapter for the local Codex App Server.

The App Server owns ChatGPT authentication. TradingAgents supplies its prompts
and tools; Codex runs the turn and calls the supplied tools over JSON-RPC.
"""

from __future__ import annotations

import atexit
import json
import queue
import shutil
import subprocess
import threading
import time
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langchain_core.utils.function_calling import convert_to_openai_tool

from .base_client import BaseLLMClient
from .validators import validate_model


def _message_text(messages: list[BaseMessage]) -> str:
    """Flatten a LangChain prompt into the text input accepted by App Server."""
    parts = []
    for message in messages:
        if isinstance(message, SystemMessage):
            continue
        elif isinstance(message, HumanMessage):
            role = "User"
        elif message.type == "tool":
            role = f"Tool {getattr(message, 'name', '')}".strip()
        else:
            role = "Assistant"
        content = message.content
        if isinstance(content, list):
            content = "\n".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content
            )
        parts.append(f"{role}:\n{content}")
    return "\n\n".join(parts)


def _system_text(messages: list[BaseMessage]) -> str:
    return "\n\n".join(
        str(message.content)
        for message in messages
        if isinstance(message, SystemMessage)
    )


def _schema_for(schema: Any) -> dict[str, Any]:
    if isinstance(schema, dict):
        return schema
    if hasattr(schema, "model_json_schema"):
        return schema.model_json_schema()
    if hasattr(schema, "schema"):
        return schema.schema()
    raise TypeError(f"Unsupported structured output schema: {schema!r}")


def _parse_schema(schema: Any, value: str) -> Any:
    try:
        data = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Codex returned invalid JSON for structured output: {exc}") from exc
    if isinstance(schema, type) and hasattr(schema, "model_validate"):
        return schema.model_validate(data)
    if isinstance(schema, type) and hasattr(schema, "parse_obj"):
        return schema.parse_obj(data)
    return data


class _CodexAppServer:
    """One serialized stdio connection shared by the quick and deep models."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._process: subprocess.Popen[str] | None = None
        self._messages: queue.Queue[dict[str, Any] | BaseException] = queue.Queue()
        self._reader: threading.Thread | None = None
        self._next_id = 0
        self._initialized = False
        self._active_tools: dict[str, Any] = {}

    def _start(self) -> None:
        if self._initialized and self._process and self._process.poll() is None:
            return
        executable = shutil.which("codex")
        if not executable:
            raise RuntimeError(
                "Codex CLI was not found. Install Codex and sign in with ChatGPT, "
                "then retry the Codex provider."
            )
        process = subprocess.Popen(
            [executable, "app-server", "--stdio"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )
        self._process = process
        self._messages = queue.Queue()
        self._reader = threading.Thread(
            target=self._read_stdout, args=(process, self._messages), daemon=True
        )
        self._reader.start()

        try:
            self._request(
                "initialize",
                {
                    "clientInfo": {
                        "name": "tradingagents",
                        "title": "TradingAgents",
                        "version": "0.4.0",
                    },
                    "capabilities": {"experimentalApi": True},
                },
            )
            self._send({"method": "initialized", "params": {}})
            account = self._request("account/read", {"refreshToken": False}).get("account")
            if not account or account.get("type") != "chatgpt":
                raise RuntimeError(
                    "Codex is not signed in with a ChatGPT account. Run `codex login` "
                    "and choose ChatGPT sign-in, then retry."
                )
            self._initialized = True
        except Exception:
            self._process.terminate()
            self._process = None
            raise

    @staticmethod
    def _read_stdout(
        process: subprocess.Popen[str], messages: queue.Queue[dict[str, Any] | BaseException]
    ) -> None:
        assert process.stdout
        try:
            for line in process.stdout:
                if line.strip():
                    messages.put(json.loads(line))
        except BaseException as exc:  # forwarded to the waiting caller
            messages.put(exc)
        finally:
            messages.put(EOFError("Codex App Server closed its output stream"))

    def _send(self, message: dict[str, Any]) -> None:
        if not self._process or not self._process.stdin or self._process.poll() is not None:
            raise RuntimeError("Codex App Server is not running")
        self._process.stdin.write(json.dumps(message, separators=(",", ":")) + "\n")
        self._process.stdin.flush()

    def _next_message(self, timeout: float) -> dict[str, Any]:
        try:
            message = self._messages.get(timeout=timeout)
        except queue.Empty as exc:
            raise TimeoutError("Timed out waiting for Codex App Server") from exc
        if isinstance(message, BaseException):
            raise RuntimeError(f"Codex App Server communication failed: {message}") from message
        return message

    def _next_request_id(self) -> str:
        self._next_id += 1
        return f"tradingagents-{self._next_id}"

    def _request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float = 30,
    ) -> dict[str, Any]:
        request_id = self._next_request_id()
        request: dict[str, Any] = {"method": method, "id": request_id}
        if params is not None:
            request["params"] = params
        self._send(request)
        while True:
            message = self._next_message(timeout)
            if message.get("method") == "item/tool/call" and "id" in message:
                self._handle_tool_call(message)
                continue
            if message.get("id") != request_id:
                continue
            if "error" in message:
                detail = message["error"].get("message", str(message["error"]))
                raise RuntimeError(f"Codex App Server {method} failed: {detail}")
            return message.get("result", {})

    def _handle_tool_call(self, request: dict[str, Any]) -> None:
        params = request.get("params", {})
        tool = self._active_tools.get(params.get("tool"))
        success = tool is not None
        try:
            if tool is None:
                raise ValueError(f"Unrecognized TradingAgents tool: {params.get('tool')}")
            result = tool.invoke(params.get("arguments", {}))
            if hasattr(result, "content"):
                result = result.content
            if not isinstance(result, str):
                result = json.dumps(result, default=str, ensure_ascii=False)
        except Exception as exc:
            success = False
            result = f"Tool error: {exc}"
        self._send(
            {
                "id": request["id"],
                "result": {
                    "success": success,
                    "contentItems": [{"type": "inputText", "text": result}],
                },
            }
        )

    @staticmethod
    def _dynamic_tools(tools: list[Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        specs = []
        by_name = {}
        for tool in tools:
            converted = convert_to_openai_tool(tool)
            function = converted["function"]
            name = function["name"]
            if name in by_name:
                raise ValueError(f"Duplicate Codex tool name: {name}")
            by_name[name] = tool
            specs.append(
                {
                    "type": "function",
                    "name": name,
                    "description": function.get("description", ""),
                    "inputSchema": function.get("parameters", {"type": "object"}),
                }
            )
        return specs, by_name

    def invoke(
        self,
        *,
        model: str,
        messages: list[BaseMessage],
        tools: list[Any] | None = None,
        output_schema: dict[str, Any] | None = None,
        reasoning_effort: str | None = None,
        timeout: float = 600,
    ) -> str:
        with self._lock:
            self._start()
            dynamic_tools, self._active_tools = self._dynamic_tools(tools or [])
            start = {
                "model": model,
                "ephemeral": True,
                "approvalPolicy": "never",
                "sandbox": "read-only",
                "runtimeWorkspaceRoots": [],
                "serviceName": "tradingagents",
                "developerInstructions": (
                    "You are a stateless language model used by TradingAgents. "
                    "Use only the dynamic tools supplied by the caller. Do not "
                    "inspect or modify files, run shell commands, or browse the web.\n\n"
                    "TradingAgents system instructions:\n"
                    + _system_text(messages)
                ),
            }
            if dynamic_tools:
                start["dynamicTools"] = dynamic_tools
            thread = self._request("thread/start", start)
            thread_id = thread["thread"]["id"]

            turn_params: dict[str, Any] = {
                "threadId": thread_id,
                "approvalPolicy": "never",
                "sandboxPolicy": {"type": "readOnly", "networkAccess": False},
                "input": [
                    {
                        "type": "text",
                        "text": (
                            "Act as the language model for TradingAgents. Answer the "
                            "provided messages directly. Use only the supplied tools "
                            "when needed; do not use shell, filesystem, or web tools.\n\n"
                            + _message_text(messages)
                        ),
                    }
                ],
            }
            if output_schema:
                turn_params["outputSchema"] = output_schema
            if reasoning_effort:
                turn_params["effort"] = reasoning_effort
            self._request("turn/start", turn_params)

            final_text = ""
            end_at = time.monotonic() + timeout
            while True:
                remaining = end_at - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Codex model turn timed out")
                message = self._next_message(remaining)
                method = message.get("method")
                if method == "item/tool/call" and "id" in message:
                    self._handle_tool_call(message)
                elif method == "item/completed":
                    item = message.get("params", {}).get("item", {})
                    if item.get("type") == "agentMessage":
                        final_text = item.get("text", final_text)
                elif method == "turn/completed":
                    turn = message.get("params", {}).get("turn", {})
                    if turn.get("status") != "completed":
                        error = turn.get("error", {}).get("message", turn.get("status"))
                        raise RuntimeError(f"Codex model turn {turn.get('status')}: {error}")
                    if not final_text:
                        for item in turn.get("items", []):
                            if item.get("type") == "agentMessage":
                                final_text = item.get("text", "")
                    return final_text


_APP_SERVER = _CodexAppServer()


def _stop_app_server() -> None:
    if _APP_SERVER._process and _APP_SERVER._process.poll() is None:
        _APP_SERVER._process.terminate()


atexit.register(_stop_app_server)


class CodexAppServerChatModel(BaseChatModel):
    """Chat model that routes prompts through a ChatGPT-authenticated Codex CLI."""

    model: str
    reasoning_effort: str | None = None
    timeout: float = 600

    @property
    def _llm_type(self) -> str:
        return "codex-app-server"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.model}

    def bind_tools(self, tools: list[Any], **kwargs: Any):
        del kwargs
        return self.bind(tools=tools)

    def with_structured_output(self, schema: Any, *, include_raw: bool = False, **kwargs: Any):
        del kwargs
        output_schema = _schema_for(schema)

        def invoke(input: Any) -> Any:
            raw = self.invoke(input, output_schema=output_schema)
            try:
                parsed = _parse_schema(schema, raw.content)
                if include_raw:
                    return {"raw": raw, "parsed": parsed, "parsing_error": None}
                return parsed
            except Exception as exc:
                if include_raw:
                    return {"raw": raw, "parsed": None, "parsing_error": exc}
                raise

        return RunnableLambda(invoke)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager
        content = _APP_SERVER.invoke(
            model=self.model,
            messages=messages,
            tools=kwargs.get("tools"),
            output_schema=kwargs.get("output_schema"),
            reasoning_effort=self.reasoning_effort,
            timeout=self.timeout,
        )
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])


class CodexClient(BaseLLMClient):
    """Client for the Codex App Server using the user's ChatGPT subscription."""

    def get_llm(self) -> CodexAppServerChatModel:
        self.warn_if_unknown_model()
        return CodexAppServerChatModel(
            model=self.model,
            reasoning_effort=self.kwargs.get("reasoning_effort"),
            timeout=float(self.kwargs.get("timeout", 600)),
        )

    def validate_model(self) -> bool:
        return validate_model("codex", self.model)
