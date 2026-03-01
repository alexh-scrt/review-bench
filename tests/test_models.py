"""Tests for the review_bench.models module.

Covers:
- parse_issues: bullet/numbered/fallback parsing
- _build_user_prompt: prompt construction
- BaseReviewAdapter: abstract interface enforcement
- GPT4Adapter: response parsing from mocked OpenAI API responses
- ClaudeAdapter: response parsing from mocked Anthropic API responses
- GeminiAdapter: construction and interface
- OllamaAdapter: HTTP interaction mocked with respx
- ModelAdapterError: exception attributes
- get_adapter / list_adapters: registry helpers
- adapter_id and model_name attributes

HTTP-level mocking uses ``respx`` for OllamaAdapter tests.
OpenAI and Anthropic adapters are tested by monkey-patching the underlying
client objects so the tests do not require real API keys.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import respx
import httpx

from review_bench.models import (
    ADAPTER_REGISTRY,
    BaseReviewAdapter,
    ClaudeAdapter,
    GPT4Adapter,
    GeminiAdapter,
    ModelAdapterError,
    OllamaAdapter,
    _build_user_prompt,
    get_adapter,
    list_adapters,
    parse_issues,
)


# ---------------------------------------------------------------------------
# parse_issues
# ---------------------------------------------------------------------------


class TestParseIssues:
    """Tests for the parse_issues function."""

    def test_empty_string_returns_empty_list(self) -> None:
        assert parse_issues("") == []

    def test_whitespace_only_returns_empty_list(self) -> None:
        assert parse_issues("   \n  \n  ") == []

    def test_dash_bullet_points_extracted(self) -> None:
        text = "- null dereference found\n- missing null check\n- add a none guard"
        result = parse_issues(text)
        assert result == ["null dereference found", "missing null check", "add a none guard"]

    def test_asterisk_bullet_points_extracted(self) -> None:
        text = "* sql injection risk\n* use parameterized queries"
        result = parse_issues(text)
        assert result == ["sql injection risk", "use parameterized queries"]

    def test_numbered_list_extracted(self) -> None:
        text = "1. off by one error\n2. index out of bounds\n3. wrong loop bound"
        result = parse_issues(text)
        assert result == ["off by one error", "index out of bounds", "wrong loop bound"]

    def test_numbered_list_with_parenthesis(self) -> None:
        text = "1) race condition detected\n2) add a lock"
        result = parse_issues(text)
        assert result == ["race condition detected", "add a lock"]

    def test_mixed_bullets_and_numbered(self) -> None:
        text = "- first issue\n2. second issue"
        result = parse_issues(text)
        assert len(result) == 2
        assert "first issue" in result
        assert "second issue" in result

    def test_fallback_to_lines_when_no_bullets(self) -> None:
        text = "null dereference issue\nresource leak present"
        result = parse_issues(text)
        assert "null dereference issue" in result
        assert "resource leak present" in result

    def test_blank_lines_ignored_in_fallback(self) -> None:
        text = "issue one\n\nissue two\n"
        result = parse_issues(text)
        assert "issue one" in result
        assert "issue two" in result
        assert "" not in result

    def test_single_bullet_returns_single_issue(self) -> None:
        result = parse_issues("- resource leak: file not closed")
        assert result == ["resource leak: file not closed"]

    def test_bullet_with_leading_whitespace(self) -> None:
        text = "  - indented issue here"
        result = parse_issues(text)
        assert result == ["indented issue here"]

    def test_returns_list_type(self) -> None:
        result = parse_issues("- some issue")
        assert isinstance(result, list)

    def test_all_items_are_strings(self) -> None:
        text = "- issue one\n- issue two\n- issue three"
        result = parse_issues(text)
        assert all(isinstance(item, str) for item in result)

    def test_real_openai_response_format(self, mock_openai_chat_response: dict) -> None:
        """parse_issues handles the content from a realistic OpenAI response."""
        content = mock_openai_chat_response["choices"][0]["message"]["content"]
        result = parse_issues(content)
        assert isinstance(result, list)
        # Content mentions null dereference — should appear somewhere
        combined = " ".join(result).lower()
        assert "null" in combined

    def test_real_anthropic_response_format(self, mock_anthropic_response: dict) -> None:
        """parse_issues handles the text from a realistic Anthropic response."""
        content = mock_anthropic_response["content"][0]["text"]
        result = parse_issues(content)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_real_gemini_response_format(self, mock_gemini_response: dict) -> None:
        """parse_issues handles the text from a realistic Gemini response."""
        content = mock_gemini_response["candidates"][0]["content"]["parts"][0]["text"]
        result = parse_issues(content)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_real_ollama_response_format(self, mock_ollama_response: dict) -> None:
        """parse_issues handles the response text from a realistic Ollama response."""
        content = mock_ollama_response["response"]
        result = parse_issues(content)
        assert isinstance(result, list)
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# _build_user_prompt
# ---------------------------------------------------------------------------


class TestBuildUserPrompt:
    """Tests for the _build_user_prompt helper."""

    def test_returns_string(self) -> None:
        result = _build_user_prompt("x = None\nprint(x.name)", "python")
        assert isinstance(result, str)

    def test_contains_language(self) -> None:
        result = _build_user_prompt("let x = null;\nconsole.log(x.foo)", "javascript")
        assert "javascript" in result

    def test_contains_code(self) -> None:
        code = "def f(x):\n    return x.name\n"
        result = _build_user_prompt(code, "python")
        assert code in result

    def test_contains_code_fence(self) -> None:
        result = _build_user_prompt("x = 1", "python")
        assert "```python" in result

    def test_nonempty_for_empty_code(self) -> None:
        result = _build_user_prompt("", "python")
        assert len(result) > 0


# ---------------------------------------------------------------------------
# ModelAdapterError
# ---------------------------------------------------------------------------


class TestModelAdapterError:
    """Tests for the ModelAdapterError exception class."""

    def test_is_runtime_error(self) -> None:
        err = ModelAdapterError("something went wrong")
        assert isinstance(err, RuntimeError)

    def test_message_stored(self) -> None:
        err = ModelAdapterError("test message")
        assert str(err) == "test message"

    def test_adapter_id_default_empty(self) -> None:
        err = ModelAdapterError("msg")
        assert err.adapter_id == ""

    def test_adapter_id_set(self) -> None:
        err = ModelAdapterError("msg", adapter_id="gpt4")
        assert err.adapter_id == "gpt4"

    def test_status_code_default_none(self) -> None:
        err = ModelAdapterError("msg")
        assert err.status_code is None

    def test_status_code_set(self) -> None:
        err = ModelAdapterError("msg", status_code=429)
        assert err.status_code == 429

    def test_can_be_raised_and_caught(self) -> None:
        with pytest.raises(ModelAdapterError) as exc_info:
            raise ModelAdapterError("api failure", adapter_id="claude", status_code=500)
        assert exc_info.value.adapter_id == "claude"
        assert exc_info.value.status_code == 500


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


class TestAdapterRegistry:
    """Tests for get_adapter and list_adapters."""

    def test_list_adapters_returns_list(self) -> None:
        result = list_adapters()
        assert isinstance(result, list)

    def test_list_adapters_sorted(self) -> None:
        result = list_adapters()
        assert result == sorted(result)

    def test_list_adapters_contains_known_ids(self) -> None:
        result = list_adapters()
        for adapter_id in ("gpt4", "claude", "gemini", "ollama"):
            assert adapter_id in result

    def test_get_adapter_gpt4(self) -> None:
        adapter = get_adapter("gpt4")
        assert isinstance(adapter, GPT4Adapter)

    def test_get_adapter_claude(self) -> None:
        adapter = get_adapter("claude")
        assert isinstance(adapter, ClaudeAdapter)

    def test_get_adapter_gemini(self) -> None:
        adapter = get_adapter("gemini")
        assert isinstance(adapter, GeminiAdapter)

    def test_get_adapter_ollama(self) -> None:
        adapter = get_adapter("ollama")
        assert isinstance(adapter, OllamaAdapter)

    def test_get_adapter_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown adapter"):
            get_adapter("nonexistent_model")

    def test_get_adapter_passes_kwargs(self) -> None:
        adapter = get_adapter("ollama", model="codellama", base_url="http://host:11434")
        assert isinstance(adapter, OllamaAdapter)
        assert adapter.model_name == "codellama"

    def test_adapter_registry_dict(self) -> None:
        assert isinstance(ADAPTER_REGISTRY, dict)
        assert "gpt4" in ADAPTER_REGISTRY
        assert "claude" in ADAPTER_REGISTRY
        assert "gemini" in ADAPTER_REGISTRY
        assert "ollama" in ADAPTER_REGISTRY


# ---------------------------------------------------------------------------
# BaseReviewAdapter — abstract interface
# ---------------------------------------------------------------------------


class TestBaseReviewAdapter:
    """Tests that BaseReviewAdapter enforces the abstract interface."""

    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            BaseReviewAdapter()  # type: ignore[abstract]

    def test_concrete_subclass_must_implement_review(self) -> None:
        class IncompleteAdapter(BaseReviewAdapter):
            adapter_id = "incomplete"

            @property
            def model_name(self) -> str:
                return "incomplete"

            # review is NOT implemented

        with pytest.raises(TypeError):
            IncompleteAdapter()  # type: ignore[abstract]

    def test_concrete_subclass_must_implement_model_name(self) -> None:
        class IncompleteAdapter2(BaseReviewAdapter):
            adapter_id = "incomplete2"

            async def review(self, code: str, language: str) -> list[str]:
                return []

            # model_name property NOT implemented

        with pytest.raises(TypeError):
            IncompleteAdapter2()  # type: ignore[abstract]

    def test_minimal_concrete_subclass(self) -> None:
        class MinimalAdapter(BaseReviewAdapter):
            adapter_id = "minimal"

            @property
            def model_name(self) -> str:
                return "minimal-model"

            async def review(self, code: str, language: str) -> list[str]:
                return ["issue one"]

        adapter = MinimalAdapter()
        assert adapter.adapter_id == "minimal"
        assert adapter.model_name == "minimal-model"

    def test_repr_format(self) -> None:
        adapter = GPT4Adapter()
        r = repr(adapter)
        assert "GPT4Adapter" in r
        assert "gpt4" in r

    @pytest.mark.asyncio
    async def test_default_review_raw_joins_issues(self) -> None:
        """BaseReviewAdapter.review_raw default joins issues with bullet prefix."""

        class SimpleAdapter(BaseReviewAdapter):
            adapter_id = "simple"

            @property
            def model_name(self) -> str:
                return "simple-model"

            async def review(self, code: str, language: str) -> list[str]:
                return ["issue one", "issue two"]

        adapter = SimpleAdapter()
        raw = await adapter.review_raw("x = 1", "python")
        assert "issue one" in raw
        assert "issue two" in raw


# ---------------------------------------------------------------------------
# GPT4Adapter — mocked client
# ---------------------------------------------------------------------------


class TestGPT4Adapter:
    """Tests for GPT4Adapter using a mocked OpenAI AsyncClient."""

    def _make_mock_response(self, content: str) -> MagicMock:
        """Build a mock object that mimics an OpenAI ChatCompletion."""
        mock_message = MagicMock()
        mock_message.content = content

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        return mock_response

    def test_adapter_id(self) -> None:
        adapter = GPT4Adapter()
        assert adapter.adapter_id == "gpt4"

    def test_model_name_default(self) -> None:
        adapter = GPT4Adapter()
        assert adapter.model_name == "gpt-4o"

    def test_model_name_custom(self) -> None:
        adapter = GPT4Adapter(model="gpt-4-turbo")
        assert adapter.model_name == "gpt-4-turbo"

    @pytest.mark.asyncio
    async def test_review_returns_list(self) -> None:
        adapter = GPT4Adapter()
        mock_response = self._make_mock_response(
            "- null dereference found\n- add a null check"
        )
        adapter._client.chat = MagicMock()
        adapter._client.chat.completions = MagicMock()
        adapter._client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await adapter.review("def f(x): return x.name", "python")
        assert isinstance(result, list)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_review_parses_bullet_points(self) -> None:
        adapter = GPT4Adapter()
        mock_response = self._make_mock_response(
            "- null dereference\n- missing null check"
        )
        adapter._client.chat = MagicMock()
        adapter._client.chat.completions = MagicMock()
        adapter._client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await adapter.review("code", "python")
        assert "null dereference" in result
        assert "missing null check" in result

    @pytest.mark.asyncio
    async def test_review_empty_content_returns_empty_list(self) -> None:
        adapter = GPT4Adapter()
        mock_response = self._make_mock_response("")
        adapter._client.chat = MagicMock()
        adapter._client.chat.completions = MagicMock()
        adapter._client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await adapter.review("code", "python")
        assert result == []

    @pytest.mark.asyncio
    async def test_review_none_content_returns_empty_list(self) -> None:
        adapter = GPT4Adapter()
        mock_response = self._make_mock_response(None)
        adapter._client.chat = MagicMock()
        adapter._client.chat.completions = MagicMock()
        adapter._client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await adapter.review("code", "python")
        assert result == []

    @pytest.mark.asyncio
    async def test_review_raises_model_adapter_error_on_exception(self) -> None:
        adapter = GPT4Adapter()
        adapter._client.chat = MagicMock()
        adapter._client.chat.completions = MagicMock()
        adapter._client.chat.completions.create = AsyncMock(
            side_effect=RuntimeError("network failure")
        )

        with pytest.raises(ModelAdapterError) as exc_info:
            await adapter.review("code", "python")
        assert exc_info.value.adapter_id == "gpt4"

    @pytest.mark.asyncio
    async def test_review_raw_returns_string(self) -> None:
        adapter = GPT4Adapter()
        raw_content = "- null dereference\n- resource leak"
        mock_response = self._make_mock_response(raw_content)
        adapter._client.chat = MagicMock()
        adapter._client.chat.completions = MagicMock()
        adapter._client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await adapter.review_raw("code", "python")
        assert isinstance(result, str)
        assert "null dereference" in result

    @pytest.mark.asyncio
    async def test_review_raw_raises_model_adapter_error_on_exception(self) -> None:
        adapter = GPT4Adapter()
        adapter._client.chat = MagicMock()
        adapter._client.chat.completions = MagicMock()
        adapter._client.chat.completions.create = AsyncMock(
            side_effect=Exception("timeout")
        )

        with pytest.raises(ModelAdapterError):
            await adapter.review_raw("code", "python")

    @pytest.mark.asyncio
    async def test_review_passes_model_to_create(self) -> None:
        adapter = GPT4Adapter(model="gpt-4-turbo")
        mock_response = self._make_mock_response("- issue")
        create_mock = AsyncMock(return_value=mock_response)
        adapter._client.chat = MagicMock()
        adapter._client.chat.completions = MagicMock()
        adapter._client.chat.completions.create = create_mock

        await adapter.review("code", "python")
        call_kwargs = create_mock.call_args
        assert call_kwargs.kwargs["model"] == "gpt-4-turbo"

    @pytest.mark.asyncio
    async def test_review_includes_system_prompt(self) -> None:
        adapter = GPT4Adapter()
        mock_response = self._make_mock_response("- issue")
        create_mock = AsyncMock(return_value=mock_response)
        adapter._client.chat = MagicMock()
        adapter._client.chat.completions = MagicMock()
        adapter._client.chat.completions.create = create_mock

        await adapter.review("code", "python")
        messages = create_mock.call_args.kwargs["messages"]
        roles = [m["role"] for m in messages]
        assert "system" in roles
        assert "user" in roles


# ---------------------------------------------------------------------------
# ClaudeAdapter — mocked client
# ---------------------------------------------------------------------------


class TestClaudeAdapter:
    """Tests for ClaudeAdapter using a mocked Anthropic AsyncClient."""

    def _make_mock_response(self, text: str) -> MagicMock:
        """Build a mock Anthropic Messages response."""
        mock_block = MagicMock()
        mock_block.text = text

        mock_response = MagicMock()
        mock_response.content = [mock_block]
        return mock_response

    def test_adapter_id(self) -> None:
        adapter = ClaudeAdapter()
        assert adapter.adapter_id == "claude"

    def test_model_name_default(self) -> None:
        adapter = ClaudeAdapter()
        assert adapter.model_name == "claude-3-opus-20240229"

    def test_model_name_custom(self) -> None:
        adapter = ClaudeAdapter(model="claude-3-5-sonnet-20240620")
        assert adapter.model_name == "claude-3-5-sonnet-20240620"

    @pytest.mark.asyncio
    async def test_review_returns_list(self) -> None:
        adapter = ClaudeAdapter()
        mock_response = self._make_mock_response(
            "- null dereference\n- missing null check"
        )
        adapter._client.messages = MagicMock()
        adapter._client.messages.create = AsyncMock(return_value=mock_response)

        result = await adapter.review("def f(x): return x.name", "python")
        assert isinstance(result, list)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_review_extracts_text_blocks(self) -> None:
        adapter = ClaudeAdapter()
        mock_response = self._make_mock_response(
            "- sql injection risk\n- use parameterized query"
        )
        adapter._client.messages = MagicMock()
        adapter._client.messages.create = AsyncMock(return_value=mock_response)

        result = await adapter.review("code", "python")
        assert "sql injection risk" in result
        assert "use parameterized query" in result

    @pytest.mark.asyncio
    async def test_review_empty_content_blocks(self) -> None:
        adapter = ClaudeAdapter()
        mock_block = MagicMock(spec=[])  # no 'text' attribute
        mock_response = MagicMock()
        mock_response.content = [mock_block]
        adapter._client.messages = MagicMock()
        adapter._client.messages.create = AsyncMock(return_value=mock_response)

        result = await adapter.review("code", "python")
        assert result == []

    @pytest.mark.asyncio
    async def test_review_multiple_content_blocks(self) -> None:
        adapter = ClaudeAdapter()
        block1 = MagicMock()
        block1.text = "- null dereference"
        block2 = MagicMock()
        block2.text = "\n- resource leak"
        mock_response = MagicMock()
        mock_response.content = [block1, block2]
        adapter._client.messages = MagicMock()
        adapter._client.messages.create = AsyncMock(return_value=mock_response)

        result = await adapter.review("code", "python")
        assert "null dereference" in result
        assert "resource leak" in result

    @pytest.mark.asyncio
    async def test_review_raises_model_adapter_error_on_exception(self) -> None:
        adapter = ClaudeAdapter()
        adapter._client.messages = MagicMock()
        adapter._client.messages.create = AsyncMock(
            side_effect=RuntimeError("api error")
        )

        with pytest.raises(ModelAdapterError) as exc_info:
            await adapter.review("code", "python")
        assert exc_info.value.adapter_id == "claude"

    @pytest.mark.asyncio
    async def test_review_raw_returns_string(self) -> None:
        adapter = ClaudeAdapter()
        mock_response = self._make_mock_response("- null dereference issue")
        adapter._client.messages = MagicMock()
        adapter._client.messages.create = AsyncMock(return_value=mock_response)

        result = await adapter.review_raw("code", "python")
        assert isinstance(result, str)
        assert "null dereference" in result

    @pytest.mark.asyncio
    async def test_review_raw_raises_model_adapter_error_on_exception(self) -> None:
        adapter = ClaudeAdapter()
        adapter._client.messages = MagicMock()
        adapter._client.messages.create = AsyncMock(
            side_effect=Exception("rate limited")
        )

        with pytest.raises(ModelAdapterError):
            await adapter.review_raw("code", "python")

    @pytest.mark.asyncio
    async def test_review_passes_system_prompt(self) -> None:
        adapter = ClaudeAdapter()
        mock_response = self._make_mock_response("- issue")
        create_mock = AsyncMock(return_value=mock_response)
        adapter._client.messages = MagicMock()
        adapter._client.messages.create = create_mock

        await adapter.review("code", "python")
        call_kwargs = create_mock.call_args.kwargs
        assert "system" in call_kwargs
        assert "code reviewer" in call_kwargs["system"].lower()


# ---------------------------------------------------------------------------
# GeminiAdapter — construction and interface tests
# ---------------------------------------------------------------------------


class TestGeminiAdapter:
    """Tests for GeminiAdapter.

    Full API calls are not mocked at the HTTP level here because the
    google-generativeai SDK doesn't use httpx directly. Instead we patch
    the model's generate_content_async method.
    """

    def test_adapter_id(self) -> None:
        adapter = GeminiAdapter()
        assert adapter.adapter_id == "gemini"

    def test_model_name_default(self) -> None:
        adapter = GeminiAdapter()
        assert adapter.model_name == "gemini-1.5-pro"

    def test_model_name_custom(self) -> None:
        adapter = GeminiAdapter(model="gemini-1.0-pro")
        assert adapter.model_name == "gemini-1.0-pro"

    def test_is_base_adapter_subclass(self) -> None:
        adapter = GeminiAdapter()
        assert isinstance(adapter, BaseReviewAdapter)

    @pytest.mark.asyncio
    async def test_review_returns_list(self) -> None:
        adapter = GeminiAdapter()
        mock_response = MagicMock()
        mock_response.text = "- sql injection vulnerability\n- use parameterized query"

        mock_model_instance = MagicMock()
        mock_model_instance.generate_content_async = AsyncMock(return_value=mock_response)

        with patch.object(adapter, "_get_model", return_value=mock_model_instance):
            result = await adapter.review("code", "python")

        assert isinstance(result, list)
        assert "sql injection vulnerability" in result
        assert "use parameterized query" in result

    @pytest.mark.asyncio
    async def test_review_empty_response(self) -> None:
        adapter = GeminiAdapter()
        mock_response = MagicMock()
        mock_response.text = ""

        mock_model_instance = MagicMock()
        mock_model_instance.generate_content_async = AsyncMock(return_value=mock_response)

        with patch.object(adapter, "_get_model", return_value=mock_model_instance):
            result = await adapter.review("code", "python")

        assert result == []

    @pytest.mark.asyncio
    async def test_review_no_text_attribute(self) -> None:
        adapter = GeminiAdapter()
        mock_response = MagicMock(spec=[])  # no 'text' attribute

        mock_model_instance = MagicMock()
        mock_model_instance.generate_content_async = AsyncMock(return_value=mock_response)

        with patch.object(adapter, "_get_model", return_value=mock_model_instance):
            result = await adapter.review("code", "python")

        assert result == []

    @pytest.mark.asyncio
    async def test_review_raises_model_adapter_error_on_exception(self) -> None:
        adapter = GeminiAdapter()
        mock_model_instance = MagicMock()
        mock_model_instance.generate_content_async = AsyncMock(
            side_effect=RuntimeError("quota exceeded")
        )

        with patch.object(adapter, "_get_model", return_value=mock_model_instance):
            with pytest.raises(ModelAdapterError) as exc_info:
                await adapter.review("code", "python")
        assert exc_info.value.adapter_id == "gemini"

    @pytest.mark.asyncio
    async def test_review_raw_returns_string(self) -> None:
        adapter = GeminiAdapter()
        raw_text = "- security issue detected\n- sql injection present"
        mock_response = MagicMock()
        mock_response.text = raw_text

        mock_model_instance = MagicMock()
        mock_model_instance.generate_content_async = AsyncMock(return_value=mock_response)

        with patch.object(adapter, "_get_model", return_value=mock_model_instance):
            result = await adapter.review_raw("code", "python")

        assert isinstance(result, str)
        assert "sql injection" in result

    @pytest.mark.asyncio
    async def test_review_raw_raises_model_adapter_error_on_exception(self) -> None:
        adapter = GeminiAdapter()
        mock_model_instance = MagicMock()
        mock_model_instance.generate_content_async = AsyncMock(
            side_effect=Exception("network error")
        )

        with patch.object(adapter, "_get_model", return_value=mock_model_instance):
            with pytest.raises(ModelAdapterError):
                await adapter.review_raw("code", "python")


# ---------------------------------------------------------------------------
# OllamaAdapter — respx HTTP mocking
# ---------------------------------------------------------------------------


class TestOllamaAdapter:
    """Tests for OllamaAdapter using respx to mock HTTP calls."""

    _BASE_URL = "http://localhost:11434"
    _GENERATE_URL = f"{_BASE_URL}/api/generate"

    def test_adapter_id(self) -> None:
        adapter = OllamaAdapter()
        assert adapter.adapter_id == "ollama"

    def test_model_name_default(self) -> None:
        adapter = OllamaAdapter()
        assert adapter.model_name == "llama3"

    def test_model_name_custom(self) -> None:
        adapter = OllamaAdapter(model="codellama")
        assert adapter.model_name == "codellama"

    def test_is_base_adapter_subclass(self) -> None:
        adapter = OllamaAdapter()
        assert isinstance(adapter, BaseReviewAdapter)

    def test_custom_base_url(self) -> None:
        adapter = OllamaAdapter(base_url="http://myhost:11434")
        assert "myhost" in adapter._base_url

    def test_trailing_slash_stripped_from_base_url(self) -> None:
        adapter = OllamaAdapter(base_url="http://localhost:11434/")
        assert not adapter._base_url.endswith("/")

    @pytest.mark.asyncio
    async def test_review_returns_list(self, mock_ollama_response: dict) -> None:
        adapter = OllamaAdapter(base_url=self._BASE_URL)
        with respx.mock(base_url=self._BASE_URL, assert_all_called=False) as mock_router:
            mock_router.post("/api/generate").mock(
                return_value=httpx.Response(200, json=mock_ollama_response)
            )
            result = await adapter.review("code", "python")

        assert isinstance(result, list)
        assert len(result) >= 1

    @pytest.mark.asyncio
    async def test_review_parses_response_field(self, mock_ollama_response: dict) -> None:
        adapter = OllamaAdapter(base_url=self._BASE_URL)
        with respx.mock(base_url=self._BASE_URL, assert_all_called=False) as mock_router:
            mock_router.post("/api/generate").mock(
                return_value=httpx.Response(200, json=mock_ollama_response)
            )
            result = await adapter.review("code", "python")

        combined = " ".join(result).lower()
        assert "race condition" in combined

    @pytest.mark.asyncio
    async def test_review_empty_response_field(self) -> None:
        adapter = OllamaAdapter(base_url=self._BASE_URL)
        payload = {"model": "llama3", "response": "", "done": True}
        with respx.mock(base_url=self._BASE_URL, assert_all_called=False) as mock_router:
            mock_router.post("/api/generate").mock(
                return_value=httpx.Response(200, json=payload)
            )
            result = await adapter.review("code", "python")

        assert result == []

    @pytest.mark.asyncio
    async def test_review_missing_response_field(self) -> None:
        adapter = OllamaAdapter(base_url=self._BASE_URL)
        payload = {"model": "llama3", "done": True}  # no 'response' key
        with respx.mock(base_url=self._BASE_URL, assert_all_called=False) as mock_router:
            mock_router.post("/api/generate").mock(
                return_value=httpx.Response(200, json=payload)
            )
            result = await adapter.review("code", "python")

        assert result == []

    @pytest.mark.asyncio
    async def test_review_http_error_raises_model_adapter_error(self) -> None:
        adapter = OllamaAdapter(base_url=self._BASE_URL)
        with respx.mock(base_url=self._BASE_URL, assert_all_called=False) as mock_router:
            mock_router.post("/api/generate").mock(
                return_value=httpx.Response(500, text="internal server error")
            )
            with pytest.raises(ModelAdapterError) as exc_info:
                await adapter.review("code", "python")

        assert exc_info.value.adapter_id == "ollama"
        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_review_404_raises_model_adapter_error(self) -> None:
        adapter = OllamaAdapter(base_url=self._BASE_URL)
        with respx.mock(base_url=self._BASE_URL, assert_all_called=False) as mock_router:
            mock_router.post("/api/generate").mock(
                return_value=httpx.Response(404, text="not found")
            )
            with pytest.raises(ModelAdapterError) as exc_info:
                await adapter.review("code", "python")

        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_review_connection_error_raises_model_adapter_error(self) -> None:
        adapter = OllamaAdapter(base_url="http://nonexistent-host:11434")
        with respx.mock() as mock_router:
            mock_router.post("http://nonexistent-host:11434/api/generate").mock(
                side_effect=httpx.ConnectError("connection refused")
            )
            with pytest.raises(ModelAdapterError):
                await adapter.review("code", "python")

    @pytest.mark.asyncio
    async def test_review_sends_model_name_in_body(self, mock_ollama_response: dict) -> None:
        adapter = OllamaAdapter(base_url=self._BASE_URL, model="codellama")
        captured_request: list[httpx.Request] = []

        def capture_and_respond(request: httpx.Request) -> httpx.Response:
            captured_request.append(request)
            return httpx.Response(200, json=mock_ollama_response)

        with respx.mock(base_url=self._BASE_URL, assert_all_called=False) as mock_router:
            mock_router.post("/api/generate").mock(side_effect=capture_and_respond)
            await adapter.review("code", "python")

        body = json.loads(captured_request[0].content)
        assert body["model"] == "codellama"

    @pytest.mark.asyncio
    async def test_review_sends_stream_false(self, mock_ollama_response: dict) -> None:
        adapter = OllamaAdapter(base_url=self._BASE_URL)
        captured_request: list[httpx.Request] = []

        def capture_and_respond(request: httpx.Request) -> httpx.Response:
            captured_request.append(request)
            return httpx.Response(200, json=mock_ollama_response)

        with respx.mock(base_url=self._BASE_URL, assert_all_called=False) as mock_router:
            mock_router.post("/api/generate").mock(side_effect=capture_and_respond)
            await adapter.review("code", "python")

        body = json.loads(captured_request[0].content)
        assert body["stream"] is False

    @pytest.mark.asyncio
    async def test_review_raw_returns_string(self, mock_ollama_response: dict) -> None:
        adapter = OllamaAdapter(base_url=self._BASE_URL)
        with respx.mock(base_url=self._BASE_URL, assert_all_called=False) as mock_router:
            mock_router.post("/api/generate").mock(
                return_value=httpx.Response(200, json=mock_ollama_response)
            )
            result = await adapter.review_raw("code", "python")

        assert isinstance(result, str)
        assert "race condition" in result.lower()

    @pytest.mark.asyncio
    async def test_review_raw_http_error_raises(self) -> None:
        adapter = OllamaAdapter(base_url=self._BASE_URL)
        with respx.mock(base_url=self._BASE_URL, assert_all_called=False) as mock_router:
            mock_router.post("/api/generate").mock(
                return_value=httpx.Response(503, text="service unavailable")
            )
            with pytest.raises(ModelAdapterError):
                await adapter.review_raw("code", "python")

    @pytest.mark.asyncio
    async def test_review_prompt_contains_code(self, mock_ollama_response: dict) -> None:
        adapter = OllamaAdapter(base_url=self._BASE_URL)
        code_snippet = "def my_unique_function_xyz(): pass"
        captured_request: list[httpx.Request] = []

        def capture_and_respond(request: httpx.Request) -> httpx.Response:
            captured_request.append(request)
            return httpx.Response(200, json=mock_ollama_response)

        with respx.mock(base_url=self._BASE_URL, assert_all_called=False) as mock_router:
            mock_router.post("/api/generate").mock(side_effect=capture_and_respond)
            await adapter.review(code_snippet, "python")

        body = json.loads(captured_request[0].content)
        assert code_snippet in body["prompt"]

    @pytest.mark.asyncio
    async def test_review_with_different_language(self, mock_ollama_response: dict) -> None:
        adapter = OllamaAdapter(base_url=self._BASE_URL)
        captured_request: list[httpx.Request] = []

        def capture_and_respond(request: httpx.Request) -> httpx.Response:
            captured_request.append(request)
            return httpx.Response(200, json=mock_ollama_response)

        with respx.mock(base_url=self._BASE_URL, assert_all_called=False) as mock_router:
            mock_router.post("/api/generate").mock(side_effect=capture_and_respond)
            result = await adapter.review("function f() {}", "javascript")

        body = json.loads(captured_request[0].content)
        assert "javascript" in body["prompt"]
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Integration: parse_issues with real-world-style responses
# ---------------------------------------------------------------------------


class TestParseIssuesIntegration:
    """Integration tests verifying parse_issues on realistic LLM response formats."""

    def test_gpt4_style_numbered_response(self) -> None:
        response = (
            "1. The variable `user` may be `None` — null dereference risk.\n"
            "2. Missing null check before accessing `user.name`.\n"
            "3. Consider returning an Optional type to signal absence."
        )
        result = parse_issues(response)
        assert len(result) == 3
        combined = " ".join(result).lower()
        assert "null" in combined

    def test_claude_style_dash_response(self) -> None:
        response = (
            "Here are the issues I found:\n"
            "- Potential null dereference: `find_user()` can return None.\n"
            "- Missing null check before attribute access on `user`.\n"
            "- Should use early return or guard clause for safety."
        )
        result = parse_issues(response)
        assert len(result) == 3

    def test_gemini_style_response(self) -> None:
        response = (
            "**Security Issue**: SQL Injection vulnerability detected.\n"
            "- The query is built using string concatenation with unsanitized user input.\n"
            "- Use parameterized queries or prepared statements instead.\n"
            "- Never trust user-controlled data in SQL queries."
        )
        result = parse_issues(response)
        # At minimum the 3 bullet lines should be extracted
        assert len(result) >= 3
        combined = " ".join(result).lower()
        assert "parameterized" in combined

    def test_ollama_style_prose_response(self) -> None:
        response = (
            "I can see a race condition in this code.\n"
            "The shared counter is not protected by a lock.\n"
            "Thread safety is not ensured for the increment operation."
        )
        result = parse_issues(response)
        assert len(result) >= 1
        combined = " ".join(result).lower()
        assert "race condition" in combined or "shared" in combined
