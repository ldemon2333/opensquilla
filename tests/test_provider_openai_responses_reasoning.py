from opensquilla.engine.types import ThinkingLevel
from opensquilla.provider.openai_responses import OpenAIResponsesProvider
from opensquilla.provider.types import ChatConfig


def _payload(cfg: ChatConfig) -> dict:
    provider = OpenAIResponsesProvider(api_key="test", model="gpt-test", base_url="http://127.0.0.1:1/v1")
    return provider._build_payload([], None, cfg)


def test_thinking_level_maps_to_reasoning_effort() -> None:
    for level, effort in (
        (ThinkingLevel.LOW, "low"),
        (ThinkingLevel.MEDIUM, "medium"),
        (ThinkingLevel.HIGH, "high"),
    ):
        assert _payload(ChatConfig(thinking=True, thinking_level=level))["reasoning"] == {"effort": effort}


def test_thinking_off_sends_no_reasoning_field() -> None:
    assert "reasoning" not in _payload(ChatConfig(thinking=False))
