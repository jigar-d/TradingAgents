from langchain_core.tools import tool
from pydantic import BaseModel

from tradingagents.llm_clients import codex_client
from tradingagents.llm_clients.codex_client import CodexAppServerChatModel, _CodexAppServer


def test_codex_model_uses_app_server_and_preserves_plain_text(monkeypatch):
    calls = []

    def invoke(**kwargs):
        calls.append(kwargs)
        return "analysis result"

    monkeypatch.setattr(codex_client._APP_SERVER, "invoke", invoke)
    result = CodexAppServerChatModel(model="gpt-5.6-terra").invoke("analyze this")

    assert result.content == "analysis result"
    assert calls[0]["model"] == "gpt-5.6-terra"
    assert calls[0]["output_schema"] is None


def test_codex_structured_output_passes_schema_and_parses_response(monkeypatch):
    class Decision(BaseModel):
        action: str

    calls = []

    def invoke(**kwargs):
        calls.append(kwargs)
        return '{"action":"HOLD"}'

    monkeypatch.setattr(codex_client._APP_SERVER, "invoke", invoke)
    model = CodexAppServerChatModel(model="gpt-5.6-terra")
    result = model.with_structured_output(Decision).invoke("choose")

    assert result == Decision(action="HOLD")
    assert calls[0]["output_schema"]["properties"]["action"]["type"] == "string"


def test_codex_dynamic_tools_map_schema_and_handler():
    @tool
    def lookup_price(ticker: str) -> str:
        """Look up a price."""
        return ticker

    specs, handlers = _CodexAppServer._dynamic_tools([lookup_price])

    assert specs[0]["type"] == "function"
    assert specs[0]["name"] == "lookup_price"
    assert specs[0]["inputSchema"]["properties"]["ticker"]["type"] == "string"
    assert handlers["lookup_price"] is lookup_price
