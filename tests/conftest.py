"""Shared pytest fixtures for the review_bench test suite.

Provides reusable fixtures covering:
- Minimal and full BenchmarkSample instances
- Temporary JSONL files for loader tests
- Representative raw JSONL record dicts
- Mock API response payloads for model adapter tests
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

import pytest

from review_bench.samples import BenchmarkSample


# ---------------------------------------------------------------------------
# Raw record dicts (as they appear in JSONL files)
# ---------------------------------------------------------------------------


@pytest.fixture()
def raw_null_dereference_record() -> dict[str, Any]:
    """Return a raw JSON record dict for a null-dereference Python sample."""
    return {
        "id": "fixture_001",
        "language": "python",
        "category": "null_dereference",
        "description": "Function dereferences a potentially None return value without checking.",
        "code": "def get_name(user_id):\n    user = find_user(user_id)\n    return user.name\n",
        "bug_labels": ["null_dereference", "missing_null_check", "attribute_error"],
        "expected_issues": ["null dereference", "none check", "missing null check"],
    }


@pytest.fixture()
def raw_sql_injection_record() -> dict[str, Any]:
    """Return a raw JSON record dict for a SQL-injection Python sample."""
    return {
        "id": "fixture_002",
        "language": "python",
        "category": "sql_injection",
        "description": "User input directly interpolated into a SQL query string.",
        "code": "def get_user(username):\n    query = \"SELECT * FROM users WHERE username = '\" + username + \"'\"\n    return db.execute(query)\n",
        "bug_labels": ["sql_injection", "injection", "security", "unsanitized_input"],
        "expected_issues": ["sql injection", "parameterized query", "unsanitized"],
    }


@pytest.fixture()
def raw_off_by_one_record() -> dict[str, Any]:
    """Return a raw JSON record dict for an off-by-one Python sample."""
    return {
        "id": "fixture_003",
        "language": "python",
        "category": "off_by_one",
        "description": "Loop iterates one index past the end of the list.",
        "code": "def sum_elements(lst):\n    total = 0\n    for i in range(len(lst) + 1):\n        total += lst[i]\n    return total\n",
        "bug_labels": ["off_by_one", "index_out_of_bounds", "loop_error"],
        "expected_issues": ["off by one", "index out of range", "loop bound"],
    }


@pytest.fixture()
def raw_resource_leak_record() -> dict[str, Any]:
    """Return a raw JSON record dict for a resource-leak Python sample."""
    return {
        "id": "fixture_004",
        "language": "python",
        "category": "resource_leak",
        "description": "File handle opened but never closed.",
        "code": "def read_config(path):\n    f = open(path, 'r')\n    data = f.read()\n    return data\n",
        "bug_labels": ["resource_leak", "file_not_closed", "missing_context_manager"],
        "expected_issues": ["resource leak", "file not closed", "context manager"],
    }


@pytest.fixture()
def raw_race_condition_record() -> dict[str, Any]:
    """Return a raw JSON record dict for a race-condition Python sample."""
    return {
        "id": "fixture_005",
        "language": "python",
        "category": "race_condition",
        "description": "Shared counter incremented without synchronization.",
        "code": (
            "import threading\n\ncounter = 0\n\n"
            "def increment():\n    global counter\n    counter += 1\n"
        ),
        "bug_labels": ["race_condition", "thread_safety", "missing_lock"],
        "expected_issues": ["race condition", "thread safety", "lock"],
    }


@pytest.fixture()
def raw_javascript_record() -> dict[str, Any]:
    """Return a raw JSON record dict for a JavaScript null-dereference sample."""
    return {
        "id": "fixture_006",
        "language": "javascript",
        "category": "null_dereference",
        "description": "Accessing a property on a value that may be undefined.",
        "code": "function getUserEmail(userId) {\n    const user = users.find(u => u.id === userId);\n    return user.email;\n}\n",
        "bug_labels": ["null_dereference", "undefined_access", "missing_null_check"],
        "expected_issues": ["null dereference", "undefined", "optional chaining"],
    }


# ---------------------------------------------------------------------------
# Typed BenchmarkSample fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_null_dereference() -> BenchmarkSample:
    """Return a typed BenchmarkSample for a null-dereference bug."""
    return BenchmarkSample(
        id="fixture_001",
        language="python",
        category="null_dereference",
        description="Function dereferences a potentially None return value without checking.",
        code="def get_name(user_id):\n    user = find_user(user_id)\n    return user.name\n",
        bug_labels=("null_dereference", "missing_null_check", "attribute_error"),
        expected_issues=("null dereference", "none check", "missing null check"),
    )


@pytest.fixture()
def sample_sql_injection() -> BenchmarkSample:
    """Return a typed BenchmarkSample for a SQL-injection bug."""
    return BenchmarkSample(
        id="fixture_002",
        language="python",
        category="sql_injection",
        description="User input directly interpolated into a SQL query string.",
        code="def get_user(username):\n    query = \"SELECT * FROM users WHERE username = '\" + username + \"'\"\n    return db.execute(query)\n",
        bug_labels=("sql_injection", "injection", "security", "unsanitized_input"),
        expected_issues=("sql injection", "parameterized query", "unsanitized"),
    )


@pytest.fixture()
def sample_off_by_one() -> BenchmarkSample:
    """Return a typed BenchmarkSample for an off-by-one bug."""
    return BenchmarkSample(
        id="fixture_003",
        language="python",
        category="off_by_one",
        description="Loop iterates one index past the end of the list.",
        code="def sum_elements(lst):\n    total = 0\n    for i in range(len(lst) + 1):\n        total += lst[i]\n    return total\n",
        bug_labels=("off_by_one", "index_out_of_bounds", "loop_error"),
        expected_issues=("off by one", "index out of range", "loop bound"),
    )


@pytest.fixture()
def sample_resource_leak() -> BenchmarkSample:
    """Return a typed BenchmarkSample for a resource-leak bug."""
    return BenchmarkSample(
        id="fixture_004",
        language="python",
        category="resource_leak",
        description="File handle opened but never closed.",
        code="def read_config(path):\n    f = open(path, 'r')\n    data = f.read()\n    return data\n",
        bug_labels=("resource_leak", "file_not_closed", "missing_context_manager"),
        expected_issues=("resource leak", "file not closed", "context manager"),
    )


@pytest.fixture()
def sample_race_condition() -> BenchmarkSample:
    """Return a typed BenchmarkSample for a race-condition bug."""
    return BenchmarkSample(
        id="fixture_005",
        language="python",
        category="race_condition",
        description="Shared counter incremented without synchronization.",
        code=(
            "import threading\n\ncounter = 0\n\n"
            "def increment():\n    global counter\n    counter += 1\n"
        ),
        bug_labels=("race_condition", "thread_safety", "missing_lock"),
        expected_issues=("race condition", "thread safety", "lock"),
    )


@pytest.fixture()
def all_fixture_samples(
    sample_null_dereference: BenchmarkSample,
    sample_sql_injection: BenchmarkSample,
    sample_off_by_one: BenchmarkSample,
    sample_resource_leak: BenchmarkSample,
    sample_race_condition: BenchmarkSample,
) -> list[BenchmarkSample]:
    """Return all five fixture BenchmarkSample objects as a list."""
    return [
        sample_null_dereference,
        sample_sql_injection,
        sample_off_by_one,
        sample_resource_leak,
        sample_race_condition,
    ]


# ---------------------------------------------------------------------------
# Temporary JSONL file fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def valid_jsonl_file(tmp_path: Path, raw_null_dereference_record: dict, raw_sql_injection_record: dict, raw_off_by_one_record: dict) -> Path:
    """Create a temporary JSONL file with three valid sample records.

    Returns:
        Path to the written JSONL file.
    """
    jsonl_path = tmp_path / "test_samples.jsonl"
    records = [
        raw_null_dereference_record,
        raw_sql_injection_record,
        raw_off_by_one_record,
    ]
    lines = [json.dumps(rec) for rec in records]
    jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return jsonl_path


@pytest.fixture()
def jsonl_file_with_comments(tmp_path: Path, raw_null_dereference_record: dict) -> Path:
    """Create a temporary JSONL file containing blank lines and comment lines.

    Returns:
        Path to the written JSONL file.
    """
    jsonl_path = tmp_path / "commented_samples.jsonl"
    content = textwrap.dedent(
        f"""\
        # This is a comment line
        {json.dumps(raw_null_dereference_record)}

        # Another comment

        """
    )
    jsonl_path.write_text(content, encoding="utf-8")
    return jsonl_path


@pytest.fixture()
def jsonl_file_with_invalid_json(tmp_path: Path, raw_null_dereference_record: dict) -> Path:
    """Create a JSONL file where the second line has malformed JSON.

    Returns:
        Path to the written JSONL file.
    """
    jsonl_path = tmp_path / "invalid_json_samples.jsonl"
    lines = [
        json.dumps(raw_null_dereference_record),
        "{this is not valid json",
    ]
    jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return jsonl_path


@pytest.fixture()
def jsonl_file_missing_field(tmp_path: Path) -> Path:
    """Create a JSONL file where a record is missing the required 'code' field.

    Returns:
        Path to the written JSONL file.
    """
    jsonl_path = tmp_path / "missing_field_samples.jsonl"
    record = {
        "id": "bad_sample",
        "language": "python",
        "category": "null_dereference",
        "description": "Missing code field.",
        # 'code' is intentionally absent
        "bug_labels": ["null_dereference"],
        "expected_issues": ["null dereference"],
    }
    jsonl_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return jsonl_path


@pytest.fixture()
def jsonl_file_empty_labels(tmp_path: Path) -> Path:
    """Create a JSONL file where a record has an empty bug_labels list.

    Returns:
        Path to the written JSONL file.
    """
    jsonl_path = tmp_path / "empty_labels_samples.jsonl"
    record = {
        "id": "empty_labels",
        "language": "python",
        "category": "null_dereference",
        "description": "Has empty bug_labels.",
        "code": "x = None; print(x.name)",
        "bug_labels": [],
        "expected_issues": ["null dereference"],
    }
    jsonl_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return jsonl_path


@pytest.fixture()
def jsonl_file_mixed_valid_invalid(
    tmp_path: Path,
    raw_null_dereference_record: dict,
    raw_sql_injection_record: dict,
) -> Path:
    """Create a JSONL file with one valid record, one invalid JSON, and another valid record.

    Returns:
        Path to the written JSONL file.
    """
    jsonl_path = tmp_path / "mixed_samples.jsonl"
    lines = [
        json.dumps(raw_null_dereference_record),
        "{broken",
        json.dumps(raw_sql_injection_record),
    ]
    jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return jsonl_path


@pytest.fixture()
def empty_jsonl_file(tmp_path: Path) -> Path:
    """Create a completely empty JSONL file.

    Returns:
        Path to the empty JSONL file.
    """
    jsonl_path = tmp_path / "empty.jsonl"
    jsonl_path.write_text("", encoding="utf-8")
    return jsonl_path


@pytest.fixture()
def jsonl_file_multi_language(
    tmp_path: Path,
    raw_null_dereference_record: dict,
    raw_javascript_record: dict,
    raw_sql_injection_record: dict,
) -> Path:
    """Create a JSONL file with Python, JavaScript, and Python records.

    Returns:
        Path to the written JSONL file.
    """
    jsonl_path = tmp_path / "multi_lang_samples.jsonl"
    records = [
        raw_null_dereference_record,   # python / null_dereference
        raw_javascript_record,          # javascript / null_dereference
        raw_sql_injection_record,       # python / sql_injection
    ]
    lines = [json.dumps(rec) for rec in records]
    jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return jsonl_path


# ---------------------------------------------------------------------------
# Mock API response fixtures (used in tests/test_models.py)
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_openai_chat_response() -> dict[str, Any]:
    """Return a minimal OpenAI chat completion API response payload."""
    return {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1715000000,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": (
                        "This code has a null dereference issue. "
                        "The user variable could be None if find_user returns None. "
                        "You should add a null check before accessing user.name."
                    ),
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 50, "completion_tokens": 40, "total_tokens": 90},
    }


@pytest.fixture()
def mock_anthropic_response() -> dict[str, Any]:
    """Return a minimal Anthropic Messages API response payload."""
    return {
        "id": "msg_test123",
        "type": "message",
        "role": "assistant",
        "model": "claude-3-opus-20240229",
        "content": [
            {
                "type": "text",
                "text": (
                    "I found a potential null dereference. "
                    "The result of find_user(user_id) may be None, and accessing "
                    ".name on None will raise AttributeError. "
                    "Add a missing null check before the attribute access."
                ),
            }
        ],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 50, "output_tokens": 45},
    }


@pytest.fixture()
def mock_gemini_response() -> dict[str, Any]:
    """Return a minimal Google Gemini generateContent API response payload."""
    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": (
                                "Security issue detected: sql injection vulnerability. "
                                "The query uses string concatenation with unsanitized user input. "
                                "Use a parameterized query instead."
                            )
                        }
                    ],
                    "role": "model",
                },
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": {"promptTokenCount": 60, "candidatesTokenCount": 35},
    }


@pytest.fixture()
def mock_ollama_response() -> dict[str, Any]:
    """Return a minimal Ollama /api/generate API response payload."""
    return {
        "model": "llama3",
        "created_at": "2024-01-01T00:00:00Z",
        "response": (
            "I can see a race condition in this code. "
            "The counter variable is shared across threads without a lock or synchronization. "
            "This is a thread safety issue that could cause incorrect counts."
        ),
        "done": True,
        "total_duration": 1234567890,
        "prompt_eval_count": 40,
        "eval_count": 38,
    }
