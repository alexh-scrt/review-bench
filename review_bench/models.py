"""Model adapter classes for review_bench.

Provides a unified async interface for querying different LLM providers
(Claude, GPT-4, Gemini, Ollama) to perform code reviews. Each adapter
exposes a common ``review(code, language) -> list[str]`` async method
that returns a list of issue strings extracted from the model's response.

Adapter classes
---------------
- :class:`BaseReviewAdapter` — abstract base defining the interface.
- :class:`GPT4Adapter` — OpenAI GPT-4o via the ``openai`` SDK.
- :class:`ClaudeAdapter` — Anthropic Claude via the ``anthropic`` SDK.
- :class:`GeminiAdapter` — Google Gemini via the ``google-generativeai`` SDK.
- :class:`OllamaAdapter` — Local Ollama models via the HTTP REST API (``httpx``).

All adapters share a system prompt that instructs the model to produce a
structured bullet-point list of code issues, which the :func:`parse_issues`
function then splits into individual issue strings.
"""

from __future__ import annotations

import abc
import logging
import re
import textwrap
from typing import Any

import httpx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are an expert code reviewer specialising in identifying software bugs.
    When given a code snippet, you must:
    1. Identify ALL bugs and potential issues in the code.
    2. List each issue as a separate bullet point starting with a dash (-).
    3. Be concise — one sentence per issue.
    4. Focus on correctness, security, resource management, and concurrency.
    5. Do NOT suggest style improvements; only report actual bugs or risks.
    """
)


def _build_user_prompt(code: str, language: str) -> str:
    """Build the user-facing prompt for a code review request.

    Args:
        code: The source code snippet to review.
        language: The programming language of the snippet.

    Returns:
        Formatted prompt string ready to send to the model.
    """
    return (
        f"Please review the following {language} code and list all bugs or issues:\n\n"
        f"```{language}\n{code}\n```\n\n"
        "List each issue as a bullet point starting with a dash (-)."
    )


# ---------------------------------------------------------------------------
# Issue parsing
# ---------------------------------------------------------------------------

# Match lines that start with common bullet markers after optional whitespace.
_BULLET_RE = re.compile(r"^\s*[-*•·‣⁃▪▸>]\s+", re.MULTILINE)
# Match numbered list items like "1. " or "1) "
_NUMBERED_RE = re.compile(r"^\s*\d+[.)\]]\s+", re.MULTILINE)


def parse_issues(response_text: str) -> list[str]:
    """Extract individual issue strings from a model's free-text response.

    The function looks for bullet-point lines (dash, asterisk, or numbered)
    and extracts the text after the bullet marker. If no bullets are found,
    it falls back to splitting on newlines and returning non-empty lines.

    Args:
        response_text: Raw text returned by the model.

    Returns:
        List of cleaned issue strings. Empty list if response is empty.
    """
    if not response_text or not response_text.strip():
        return []

    lines = response_text.splitlines()
    issues: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Check for bullet markers
        if _BULLET_RE.match(line):
            issue = _BULLET_RE.sub("", line, count=1).strip()
            if issue:
                issues.append(issue)
        elif _NUMBERED_RE.match(line):
            issue = _NUMBERED_RE.sub("", line, count=1).strip()
            if issue:
                issues.append(issue)

    # Fallback: no structured bullets found — treat each non-empty line as an issue.
    if not issues:
        issues = [ln.strip() for ln in lines if ln.strip()]

    return issues


# ---------------------------------------------------------------------------
# Abstract base adapter
# ---------------------------------------------------------------------------


class BaseReviewAdapter(abc.ABC):
    """Abstract base class defining the common interface for all model adapters.

    Subclasses must implement :meth:`review` and :meth:`model_name`.

    Attributes:
        adapter_id: Short identifier string for this adapter (e.g. ``"gpt4"``),
            used in result records and log messages.
    """

    adapter_id: str = "base"

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        """Full provider model name (e.g. ``"gpt-4o"`` or ``"claude-3-opus-20240229"``)."""

    @abc.abstractmethod
    async def review(self, code: str, language: str) -> list[str]:
        """Perform an async code review and return a list of issue strings.

        Args:
            code: The source code snippet to review.
            language: The programming language of the snippet (e.g. ``"python"``),
                used to construct the prompt.

        Returns:
            List of issue strings extracted from the model's response.
            Returns an empty list if the model returns no content or the
            request fails.

        Raises:
            ModelAdapterError: If the API call fails unrecoverably.
        """

    async def review_raw(self, code: str, language: str) -> str:
        """Return the raw text response from the model without parsing.

        Default implementation calls :meth:`review` and joins the issues;
        subclasses may override to cache the raw response more efficiently.

        Args:
            code: The source code snippet to review.
            language: The programming language of the snippet.

        Returns:
            Raw response text as a single string.
        """
        issues = await self.review(code, language)
        return "\n".join(f"- {issue}" for issue in issues)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(adapter_id={self.adapter_id!r}, model_name={self.model_name!r})"


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class ModelAdapterError(RuntimeError):
    """Raised when a model adapter encounters an unrecoverable API error.

    Attributes:
        adapter_id: The ID of the adapter that raised the error.
        status_code: HTTP status code if applicable, else ``None``.
    """

    def __init__(
        self,
        message: str,
        adapter_id: str = "",
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.adapter_id = adapter_id
        self.status_code = status_code


# ---------------------------------------------------------------------------
# GPT-4 adapter (OpenAI)
# ---------------------------------------------------------------------------


class GPT4Adapter(BaseReviewAdapter):
    """Code review adapter for OpenAI GPT-4o.

    Uses the ``openai`` Python SDK.  Requires ``OPENAI_API_KEY`` to be set
    in the environment (the SDK picks this up automatically).

    Args:
        api_key: Optional explicit API key; if ``None`` the SDK reads
            ``OPENAI_API_KEY`` from the environment.
        model: The OpenAI model identifier to use. Defaults to ``"gpt-4o"``.
        temperature: Sampling temperature. Defaults to ``0.2`` for
            deterministic code-review outputs.
        max_tokens: Maximum tokens to generate. Defaults to ``1024``.
        timeout: HTTP request timeout in seconds. Defaults to ``60``.
    """

    adapter_id: str = "gpt4"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        timeout: float = 60.0,
    ) -> None:
        try:
            from openai import AsyncOpenAI  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for GPT4Adapter. "
                "Install it with: pip install openai"
            ) from exc

        self._client = AsyncOpenAI(api_key=api_key, timeout=timeout)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    @property
    def model_name(self) -> str:
        """Full OpenAI model identifier."""
        return self._model

    async def review(self, code: str, language: str) -> list[str]:
        """Send code to GPT-4o for review and return parsed issues.

        Args:
            code: Source code snippet.
            language: Programming language name.

        Returns:
            List of issue strings parsed from the model response.

        Raises:
            ModelAdapterError: On API errors.
        """
        user_prompt = _build_user_prompt(code, language)
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response.choices[0].message.content or ""
            logger.debug(
                "GPT4Adapter: received %d chars for language=%s",
                len(content),
                language,
            )
            return parse_issues(content)
        except Exception as exc:
            # Catch-all: convert SDK exceptions to our custom error type.
            logger.error("GPT4Adapter error: %s", exc)
            raise ModelAdapterError(
                f"GPT-4 API call failed: {exc}",
                adapter_id=self.adapter_id,
            ) from exc

    async def review_raw(self, code: str, language: str) -> str:
        """Return raw GPT-4o response text without parsing.

        Args:
            code: Source code snippet.
            language: Programming language name.

        Returns:
            Raw model response string.

        Raises:
            ModelAdapterError: On API errors.
        """
        user_prompt = _build_user_prompt(code, language)
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            logger.error("GPT4Adapter raw error: %s", exc)
            raise ModelAdapterError(
                f"GPT-4 API call failed: {exc}",
                adapter_id=self.adapter_id,
            ) from exc


# ---------------------------------------------------------------------------
# Claude adapter (Anthropic)
# ---------------------------------------------------------------------------


class ClaudeAdapter(BaseReviewAdapter):
    """Code review adapter for Anthropic Claude models.

    Uses the ``anthropic`` Python SDK.  Requires ``ANTHROPIC_API_KEY`` to be
    set in the environment (the SDK picks this up automatically).

    Args:
        api_key: Optional explicit API key.
        model: The Anthropic model identifier. Defaults to
            ``"claude-3-opus-20240229"``.
        temperature: Sampling temperature. Defaults to ``0.2``.
        max_tokens: Maximum tokens to generate. Defaults to ``1024``.
        timeout: HTTP request timeout in seconds. Defaults to ``60``.
    """

    adapter_id: str = "claude"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-3-opus-20240229",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        timeout: float = 60.0,
    ) -> None:
        try:
            from anthropic import AsyncAnthropic  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "The 'anthropic' package is required for ClaudeAdapter. "
                "Install it with: pip install anthropic"
            ) from exc

        self._client = AsyncAnthropic(api_key=api_key, timeout=timeout)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    @property
    def model_name(self) -> str:
        """Full Anthropic model identifier."""
        return self._model

    async def review(self, code: str, language: str) -> list[str]:
        """Send code to Claude for review and return parsed issues.

        Args:
            code: Source code snippet.
            language: Programming language name.

        Returns:
            List of issue strings parsed from the model response.

        Raises:
            ModelAdapterError: On API errors.
        """
        user_prompt = _build_user_prompt(code, language)
        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                system=_SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": user_prompt},
                ],
            )
            # Content is a list of ContentBlock objects; extract text blocks.
            content = ""
            for block in response.content:
                if hasattr(block, "text"):
                    content += block.text
            logger.debug(
                "ClaudeAdapter: received %d chars for language=%s",
                len(content),
                language,
            )
            return parse_issues(content)
        except Exception as exc:
            logger.error("ClaudeAdapter error: %s", exc)
            raise ModelAdapterError(
                f"Claude API call failed: {exc}",
                adapter_id=self.adapter_id,
            ) from exc

    async def review_raw(self, code: str, language: str) -> str:
        """Return raw Claude response text without parsing.

        Args:
            code: Source code snippet.
            language: Programming language name.

        Returns:
            Raw model response string.

        Raises:
            ModelAdapterError: On API errors.
        """
        user_prompt = _build_user_prompt(code, language)
        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                system=_SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = ""
            for block in response.content:
                if hasattr(block, "text"):
                    content += block.text
            return content
        except Exception as exc:
            logger.error("ClaudeAdapter raw error: %s", exc)
            raise ModelAdapterError(
                f"Claude API call failed: {exc}",
                adapter_id=self.adapter_id,
            ) from exc


# ---------------------------------------------------------------------------
# Gemini adapter (Google)
# ---------------------------------------------------------------------------


class GeminiAdapter(BaseReviewAdapter):
    """Code review adapter for Google Gemini via the ``google-generativeai`` SDK.

    Requires ``GOOGLE_API_KEY`` to be set in the environment, or passed
    explicitly via *api_key*.

    Args:
        api_key: Optional explicit Google API key. If ``None``, the SDK reads
            ``GOOGLE_API_KEY`` from the environment.
        model: Gemini model name. Defaults to ``"gemini-1.5-pro"``.
        temperature: Sampling temperature. Defaults to ``0.2``.
        max_tokens: Maximum output tokens. Defaults to ``1024``.
    """

    adapter_id: str = "gemini"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-1.5-pro",
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> None:
        try:
            import google.generativeai as genai  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "The 'google-generativeai' package is required for GeminiAdapter. "
                "Install it with: pip install google-generativeai"
            ) from exc

        if api_key:
            genai.configure(api_key=api_key)

        self._genai = genai
        self._model_name = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._generation_config = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

    @property
    def model_name(self) -> str:
        """Full Google Gemini model identifier."""
        return self._model_name

    def _get_model(self) -> Any:
        """Instantiate and return the GenerativeModel with system instruction."""
        return self._genai.GenerativeModel(
            model_name=self._model_name,
            system_instruction=_SYSTEM_PROMPT,
            generation_config=self._generation_config,
        )

    async def review(self, code: str, language: str) -> list[str]:
        """Send code to Gemini for review and return parsed issues.

        .. note::
            The google-generativeai SDK's async support uses
            ``generate_content_async``; this method wraps that call.

        Args:
            code: Source code snippet.
            language: Programming language name.

        Returns:
            List of issue strings parsed from the model response.

        Raises:
            ModelAdapterError: On API errors.
        """
        user_prompt = _build_user_prompt(code, language)
        try:
            model_instance = self._get_model()
            response = await model_instance.generate_content_async(user_prompt)
            content = response.text if hasattr(response, "text") else ""
            logger.debug(
                "GeminiAdapter: received %d chars for language=%s",
                len(content),
                language,
            )
            return parse_issues(content)
        except Exception as exc:
            logger.error("GeminiAdapter error: %s", exc)
            raise ModelAdapterError(
                f"Gemini API call failed: {exc}",
                adapter_id=self.adapter_id,
            ) from exc

    async def review_raw(self, code: str, language: str) -> str:
        """Return raw Gemini response text without parsing.

        Args:
            code: Source code snippet.
            language: Programming language name.

        Returns:
            Raw model response string.

        Raises:
            ModelAdapterError: On API errors.
        """
        user_prompt = _build_user_prompt(code, language)
        try:
            model_instance = self._get_model()
            response = await model_instance.generate_content_async(user_prompt)
            return response.text if hasattr(response, "text") else ""
        except Exception as exc:
            logger.error("GeminiAdapter raw error: %s", exc)
            raise ModelAdapterError(
                f"Gemini API call failed: {exc}",
                adapter_id=self.adapter_id,
            ) from exc


# ---------------------------------------------------------------------------
# Ollama adapter (local HTTP)
# ---------------------------------------------------------------------------


class OllamaAdapter(BaseReviewAdapter):
    """Code review adapter for local Ollama models via its REST API.

    Communicates with a running Ollama server using ``httpx`` async HTTP.
    No API key is required; the server must be reachable at *base_url*.

    Args:
        model: Ollama model name (e.g. ``"llama3"`` or ``"codellama"``). Defaults to
            ``"llama3"``.
        base_url: Base URL of the Ollama server. Defaults to
            ``"http://localhost:11434"``.
        timeout: HTTP request timeout in seconds. Defaults to ``120.0`` because
            local models can be slow.
        temperature: Sampling temperature passed in the request options. Defaults
            to ``0.2``.
        max_tokens: Maximum tokens for the response (``num_predict``). Defaults to
            ``1024``.
    """

    adapter_id: str = "ollama"
    _GENERATE_PATH = "/api/generate"

    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._temperature = temperature
        self._max_tokens = max_tokens

    @property
    def model_name(self) -> str:
        """Ollama model name."""
        return self._model

    def _build_request_body(self, prompt: str) -> dict[str, Any]:
        """Construct the JSON body for ``POST /api/generate``.

        Args:
            prompt: The full prompt text to send.

        Returns:
            Dictionary suitable for JSON serialisation.
        """
        return {
            "model": self._model,
            "prompt": f"{_SYSTEM_PROMPT}\n\n{prompt}",
            "stream": False,
            "options": {
                "temperature": self._temperature,
                "num_predict": self._max_tokens,
            },
        }

    async def review(self, code: str, language: str) -> list[str]:
        """Send code to a local Ollama model for review and return parsed issues.

        Args:
            code: Source code snippet.
            language: Programming language name.

        Returns:
            List of issue strings parsed from the model response.

        Raises:
            ModelAdapterError: If the HTTP request fails or returns a non-200 status.
        """
        user_prompt = _build_user_prompt(code, language)
        body = self._build_request_body(user_prompt)
        url = f"{self._base_url}{self._GENERATE_PATH}"

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(url, json=body)
                response.raise_for_status()
                data = response.json()
                content: str = data.get("response", "")
                logger.debug(
                    "OllamaAdapter: received %d chars for language=%s model=%s",
                    len(content),
                    language,
                    self._model,
                )
                return parse_issues(content)
        except httpx.HTTPStatusError as exc:
            raise ModelAdapterError(
                f"Ollama HTTP error {exc.response.status_code}: {exc}",
                adapter_id=self.adapter_id,
                status_code=exc.response.status_code,
            ) from exc
        except httpx.RequestError as exc:
            raise ModelAdapterError(
                f"Ollama request failed (is Ollama running at {self._base_url}?): {exc}",
                adapter_id=self.adapter_id,
            ) from exc
        except Exception as exc:
            logger.error("OllamaAdapter unexpected error: %s", exc)
            raise ModelAdapterError(
                f"Ollama adapter error: {exc}",
                adapter_id=self.adapter_id,
            ) from exc

    async def review_raw(self, code: str, language: str) -> str:
        """Return raw Ollama response text without parsing.

        Args:
            code: Source code snippet.
            language: Programming language name.

        Returns:
            Raw model response string from Ollama.

        Raises:
            ModelAdapterError: If the HTTP request fails.
        """
        user_prompt = _build_user_prompt(code, language)
        body = self._build_request_body(user_prompt)
        url = f"{self._base_url}{self._GENERATE_PATH}"

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(url, json=body)
                response.raise_for_status()
                data = response.json()
                return data.get("response", "")
        except httpx.HTTPStatusError as exc:
            raise ModelAdapterError(
                f"Ollama HTTP error {exc.response.status_code}: {exc}",
                adapter_id=self.adapter_id,
                status_code=exc.response.status_code,
            ) from exc
        except httpx.RequestError as exc:
            raise ModelAdapterError(
                f"Ollama request failed (is Ollama running at {self._base_url}?): {exc}",
                adapter_id=self.adapter_id,
            ) from exc


# ---------------------------------------------------------------------------
# Registry / factory
# ---------------------------------------------------------------------------

#: Mapping from short adapter ID to the concrete adapter class.
ADAPTER_REGISTRY: dict[str, type[BaseReviewAdapter]] = {
    "gpt4": GPT4Adapter,
    "claude": ClaudeAdapter,
    "gemini": GeminiAdapter,
    "ollama": OllamaAdapter,
}


def get_adapter(adapter_id: str, **kwargs: Any) -> BaseReviewAdapter:
    """Instantiate a registered adapter by its short identifier.

    Args:
        adapter_id: One of the keys in :data:`ADAPTER_REGISTRY` (e.g. ``"gpt4"``
            ``"claude"``, ``"gemini"``, ``"ollama"``).
        **kwargs: Keyword arguments forwarded to the adapter's ``__init__``.

    Returns:
        Instantiated :class:`BaseReviewAdapter` subclass.

    Raises:
        ValueError: If *adapter_id* is not found in the registry.
    """
    adapter_cls = ADAPTER_REGISTRY.get(adapter_id)
    if adapter_cls is None:
        known = ", ".join(sorted(ADAPTER_REGISTRY))
        raise ValueError(
            f"Unknown adapter id {adapter_id!r}. Known adapters: {known}"
        )
    return adapter_cls(**kwargs)


def list_adapters() -> list[str]:
    """Return a sorted list of registered adapter IDs.

    Returns:
        Sorted list of adapter identifier strings.
    """
    return sorted(ADAPTER_REGISTRY.keys())
