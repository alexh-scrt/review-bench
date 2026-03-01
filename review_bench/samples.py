"""Sample loader and typed dataclasses for the review_bench benchmark dataset.

This module defines the core data model for benchmark samples and scoring results,
and provides a loader that reads, parses, and validates the bundled JSONL dataset
into strongly-typed Python objects.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

# Path to the bundled dataset relative to this file's parent package root.
_DATA_DIR = Path(__file__).parent.parent / "data"
_DEFAULT_SAMPLES_PATH = _DATA_DIR / "samples.jsonl"

# Valid categories recognised by the benchmark.
VALID_CATEGORIES: frozenset[str] = frozenset(
    {
        "null_dereference",
        "off_by_one",
        "sql_injection",
        "race_condition",
        "resource_leak",
    }
)

# Valid programming languages present in the dataset.
VALID_LANGUAGES: frozenset[str] = frozenset(
    {"python", "javascript", "java", "go", "typescript", "rust", "c", "cpp"}
)


class SampleValidationError(ValueError):
    """Raised when a benchmark sample record fails schema validation."""


class SamplesLoadError(OSError):
    """Raised when the samples JSONL file cannot be opened or read."""


@dataclass(frozen=True)
class BenchmarkSample:
    """A single buggy code sample from the benchmark dataset.

    Attributes:
        id: Unique string identifier for the sample (e.g. ``"sample_001"``).
        language: Programming language of the snippet (e.g. ``"python"``).
        category: Bug category this sample belongs to (e.g. ``"null_dereference"``).
        description: Human-readable description of the bug present in the code.
        code: The buggy source code snippet.
        bug_labels: Canonical keyword labels for the bug (used for scoring).
        expected_issues: Natural-language phrases the model's review should mention.
    """

    id: str
    language: str
    category: str
    description: str
    code: str
    bug_labels: tuple[str, ...]
    expected_issues: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate field types and value constraints after construction."""
        if not self.id or not isinstance(self.id, str):
            raise SampleValidationError(f"Sample 'id' must be a non-empty string; got {self.id!r}")
        if not self.language or not isinstance(self.language, str):
            raise SampleValidationError(
                f"Sample '{self.id}': 'language' must be a non-empty string; got {self.language!r}"
            )
        if not self.category or not isinstance(self.category, str):
            raise SampleValidationError(
                f"Sample '{self.id}': 'category' must be a non-empty string; got {self.category!r}"
            )
        if not self.code or not isinstance(self.code, str):
            raise SampleValidationError(
                f"Sample '{self.id}': 'code' must be a non-empty string."
            )
        if not self.bug_labels:
            raise SampleValidationError(
                f"Sample '{self.id}': 'bug_labels' must contain at least one label."
            )
        if not self.expected_issues:
            raise SampleValidationError(
                f"Sample '{self.id}': 'expected_issues' must contain at least one issue."
            )


@dataclass
class CategoryScore:
    """Precision/recall/F1 metrics for a single bug category.

    Attributes:
        category: The bug category name.
        precision: Fraction of model-identified issues that match ground-truth labels.
        recall: Fraction of ground-truth labels identified by the model.
        f1: Harmonic mean of precision and recall.
        sample_count: Number of samples evaluated in this category.
    """

    category: str
    precision: float
    recall: float
    f1: float
    sample_count: int


@dataclass
class ModelScore:
    """Aggregate scoring results for a single model across all benchmark samples.

    Attributes:
        model_id: Identifier string for the model adapter (e.g. ``"gpt4"``).
        model_name: Full model name as reported by the provider.
        precision: Aggregate precision across all evaluated samples.
        recall: Aggregate recall across all evaluated samples.
        f1: Aggregate F1 across all evaluated samples.
        sample_count: Total number of samples evaluated.
        elapsed_seconds: Wall-clock time taken for the full run in seconds.
        category_scores: Per-category breakdown of precision/recall/F1.
    """

    model_id: str
    model_name: str
    precision: float
    recall: float
    f1: float
    sample_count: int
    elapsed_seconds: float
    category_scores: list[CategoryScore] = field(default_factory=list)


@dataclass
class SampleResult:
    """Raw result from running a single model against a single benchmark sample.

    Attributes:
        sample_id: ID of the :class:`BenchmarkSample` that was evaluated.
        model_id: Identifier of the model adapter that produced the review.
        identified_issues: List of issue strings extracted from the model's response.
        precision: Precision for this individual sample.
        recall: Recall for this individual sample.
        f1: F1 for this individual sample.
        raw_response: The unprocessed text returned by the model.
    """

    sample_id: str
    model_id: str
    identified_issues: list[str]
    precision: float
    recall: float
    f1: float
    raw_response: str = ""


@dataclass
class BenchmarkRun:
    """Complete results from a full benchmark execution.

    Attributes:
        run_id: Unique identifier for this run (typically a timestamp string).
        model_scores: Per-model aggregate scoring results.
        sample_results: Flat list of all per-sample, per-model results.
        total_samples: Total number of distinct samples in the dataset used.
        timestamp: ISO-8601 timestamp string of when the run was started.
    """

    run_id: str
    model_scores: list[ModelScore]
    sample_results: list[SampleResult]
    total_samples: int
    timestamp: str


# ---------------------------------------------------------------------------
# Internal parsing helpers
# ---------------------------------------------------------------------------


def _require_str(record: dict, key: str, sample_id: str = "<unknown>") -> str:
    """Extract a required non-empty string field from a raw record dict.

    Args:
        record: The parsed JSON object for one JSONL line.
        key: The field name to extract.
        sample_id: Sample identifier used in error messages.

    Returns:
        The string value of the field.

    Raises:
        SampleValidationError: If the key is absent or the value is not a non-empty string.
    """
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SampleValidationError(
            f"Sample '{sample_id}': field '{key}' must be a non-empty string; got {value!r}"
        )
    return value


def _require_str_list(record: dict, key: str, sample_id: str = "<unknown>") -> list[str]:
    """Extract a required list-of-strings field from a raw record dict.

    Args:
        record: The parsed JSON object for one JSONL line.
        key: The field name to extract.
        sample_id: Sample identifier used in error messages.

    Returns:
        A list of strings (guaranteed non-empty list, each element a non-empty string).

    Raises:
        SampleValidationError: If the key is absent, the value is not a list, the list
            is empty, or any element is not a non-empty string.
    """
    value = record.get(key)
    if not isinstance(value, list) or len(value) == 0:
        raise SampleValidationError(
            f"Sample '{sample_id}': field '{key}' must be a non-empty list; got {value!r}"
        )
    cleaned: list[str] = []
    for idx, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise SampleValidationError(
                f"Sample '{sample_id}': field '{key}[{idx}]' must be a non-empty string; "
                f"got {item!r}"
            )
        cleaned.append(item)
    return cleaned


def _parse_record(record: dict, line_number: int) -> BenchmarkSample:
    """Parse and validate a single JSON record into a :class:`BenchmarkSample`.

    Args:
        record: Decoded JSON object from one JSONL line.
        line_number: 1-based line number in the JSONL file (used in error messages).

    Returns:
        A validated :class:`BenchmarkSample` instance.

    Raises:
        SampleValidationError: If any required field is missing or invalid.
    """
    # Extract id first so subsequent messages can reference it.
    raw_id = record.get("id", f"<line {line_number}>")
    sample_id = raw_id if isinstance(raw_id, str) and raw_id.strip() else f"<line {line_number}>"

    sample_id_str = _require_str(record, "id", sample_id)
    language = _require_str(record, "language", sample_id_str).lower()
    category = _require_str(record, "category", sample_id_str).lower()
    description = _require_str(record, "description", sample_id_str)
    code = _require_str(record, "code", sample_id_str)
    bug_labels = _require_str_list(record, "bug_labels", sample_id_str)
    expected_issues = _require_str_list(record, "expected_issues", sample_id_str)

    return BenchmarkSample(
        id=sample_id_str,
        language=language,
        category=category,
        description=description,
        code=code,
        bug_labels=tuple(bug_labels),
        expected_issues=tuple(expected_issues),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def iter_samples(
    path: Path | str | None = None,
    *,
    skip_invalid: bool = False,
) -> Iterator[BenchmarkSample]:
    """Yield :class:`BenchmarkSample` objects parsed from a JSONL file.

    Lines that are blank or start with ``#`` are silently skipped.
    Lines with JSON syntax errors or failing schema validation raise
    :class:`SampleValidationError` unless *skip_invalid* is ``True``.

    Args:
        path: Path to the JSONL file.  Defaults to the bundled
            ``data/samples.jsonl`` when ``None``.
        skip_invalid: When ``True``, log a warning and continue on per-line
            errors instead of raising.

    Yields:
        Parsed and validated :class:`BenchmarkSample` instances.

    Raises:
        SamplesLoadError: If the file cannot be opened or read.
        SampleValidationError: If a record is invalid and *skip_invalid* is ``False``.
    """
    resolved = Path(path) if path is not None else _DEFAULT_SAMPLES_PATH
    try:
        file_handle = resolved.open(encoding="utf-8")
    except OSError as exc:
        raise SamplesLoadError(f"Cannot open samples file '{resolved}': {exc}") from exc

    with file_handle as fh:
        for line_number, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                msg = f"JSON parse error on line {line_number} of '{resolved}': {exc}"
                if skip_invalid:
                    logger.warning(msg)
                    continue
                raise SampleValidationError(msg) from exc

            if not isinstance(record, dict):
                msg = (
                    f"Line {line_number} of '{resolved}' is not a JSON object; "
                    f"got {type(record).__name__}"
                )
                if skip_invalid:
                    logger.warning(msg)
                    continue
                raise SampleValidationError(msg)

            try:
                sample = _parse_record(record, line_number)
            except SampleValidationError as exc:
                if skip_invalid:
                    logger.warning("Skipping invalid record: %s", exc)
                    continue
                raise

            yield sample


def load_samples(
    path: Path | str | None = None,
    *,
    category: str | None = None,
    language: str | None = None,
    skip_invalid: bool = False,
) -> list[BenchmarkSample]:
    """Load all benchmark samples from a JSONL file into a list.

    Optionally filter by *category* and/or *language*.

    Args:
        path: Path to the JSONL file.  Defaults to the bundled dataset.
        category: If given, only return samples whose ``category`` matches
            this value (case-insensitive).
        language: If given, only return samples whose ``language`` matches
            this value (case-insensitive).
        skip_invalid: When ``True``, silently skip malformed records.

    Returns:
        A list of :class:`BenchmarkSample` instances matching the filters.

    Raises:
        SamplesLoadError: If the file cannot be opened.
        SampleValidationError: If any record is invalid and *skip_invalid* is ``False``.
    """
    category_filter = category.lower() if category else None
    language_filter = language.lower() if language else None

    samples: list[BenchmarkSample] = []
    for sample in iter_samples(path, skip_invalid=skip_invalid):
        if category_filter and sample.category != category_filter:
            continue
        if language_filter and sample.language != language_filter:
            continue
        samples.append(sample)

    logger.debug(
        "Loaded %d sample(s) (category=%s, language=%s) from '%s'",
        len(samples),
        category_filter or "*",
        language_filter or "*",
        path or _DEFAULT_SAMPLES_PATH,
    )
    return samples


def get_categories(samples: list[BenchmarkSample]) -> list[str]:
    """Return a sorted list of unique category names present in *samples*.

    Args:
        samples: List of loaded benchmark samples.

    Returns:
        Sorted list of unique category strings.
    """
    return sorted({s.category for s in samples})


def get_languages(samples: list[BenchmarkSample]) -> list[str]:
    """Return a sorted list of unique language names present in *samples*.

    Args:
        samples: List of loaded benchmark samples.

    Returns:
        Sorted list of unique language strings.
    """
    return sorted({s.language for s in samples})


def samples_by_category(samples: list[BenchmarkSample]) -> dict[str, list[BenchmarkSample]]:
    """Group *samples* by their ``category`` field.

    Args:
        samples: List of loaded benchmark samples.

    Returns:
        Dictionary mapping category name to the list of samples in that category.
        Keys appear in sorted order.
    """
    result: dict[str, list[BenchmarkSample]] = {}
    for sample in samples:
        result.setdefault(sample.category, []).append(sample)
    return dict(sorted(result.items()))
