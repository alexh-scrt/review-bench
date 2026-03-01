"""Tests for the review_bench.samples module.

Covers:
- BenchmarkSample dataclass construction and validation
- _parse_record internal helper
- iter_samples iterator with valid, invalid, and edge-case JSONL files
- load_samples with category and language filters
- Utility functions: get_categories, get_languages, samples_by_category
- SamplesLoadError for missing files
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from review_bench.samples import (
    BenchmarkSample,
    BenchmarkRun,
    CategoryScore,
    ModelScore,
    SampleResult,
    SampleValidationError,
    SamplesLoadError,
    _parse_record,
    get_categories,
    get_languages,
    iter_samples,
    load_samples,
    samples_by_category,
)


# ---------------------------------------------------------------------------
# BenchmarkSample construction
# ---------------------------------------------------------------------------


class TestBenchmarkSampleConstruction:
    """Tests for the BenchmarkSample frozen dataclass."""

    def test_valid_construction(self, sample_null_dereference: BenchmarkSample) -> None:
        """A fully-specified sample is constructed without error."""
        assert sample_null_dereference.id == "fixture_001"
        assert sample_null_dereference.language == "python"
        assert sample_null_dereference.category == "null_dereference"
        assert isinstance(sample_null_dereference.bug_labels, tuple)
        assert isinstance(sample_null_dereference.expected_issues, tuple)

    def test_bug_labels_is_tuple(self, sample_null_dereference: BenchmarkSample) -> None:
        """bug_labels is stored as a tuple (frozen dataclass)."""
        assert isinstance(sample_null_dereference.bug_labels, tuple)
        assert len(sample_null_dereference.bug_labels) >= 1

    def test_expected_issues_is_tuple(self, sample_null_dereference: BenchmarkSample) -> None:
        """expected_issues is stored as a tuple."""
        assert isinstance(sample_null_dereference.expected_issues, tuple)
        assert len(sample_null_dereference.expected_issues) >= 1

    def test_frozen_immutability(self, sample_null_dereference: BenchmarkSample) -> None:
        """BenchmarkSample is frozen; attribute assignment raises FrozenInstanceError."""
        with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
            sample_null_dereference.id = "new_id"  # type: ignore[misc]

    def test_empty_id_raises(self) -> None:
        """An empty id string raises SampleValidationError."""
        with pytest.raises(SampleValidationError, match="id"):
            BenchmarkSample(
                id="",
                language="python",
                category="null_dereference",
                description="desc",
                code="x = 1",
                bug_labels=("label",),
                expected_issues=("issue",),
            )

    def test_empty_code_raises(self) -> None:
        """An empty code string raises SampleValidationError."""
        with pytest.raises(SampleValidationError, match="code"):
            BenchmarkSample(
                id="test",
                language="python",
                category="null_dereference",
                description="desc",
                code="",
                bug_labels=("label",),
                expected_issues=("issue",),
            )

    def test_empty_bug_labels_raises(self) -> None:
        """An empty bug_labels tuple raises SampleValidationError."""
        with pytest.raises(SampleValidationError, match="bug_labels"):
            BenchmarkSample(
                id="test",
                language="python",
                category="null_dereference",
                description="desc",
                code="x = 1",
                bug_labels=(),
                expected_issues=("issue",),
            )

    def test_empty_expected_issues_raises(self) -> None:
        """An empty expected_issues tuple raises SampleValidationError."""
        with pytest.raises(SampleValidationError, match="expected_issues"):
            BenchmarkSample(
                id="test",
                language="python",
                category="null_dereference",
                description="desc",
                code="x = 1",
                bug_labels=("label",),
                expected_issues=(),
            )

    def test_equality_and_hash(self) -> None:
        """Two samples with identical fields are equal and have the same hash."""
        s1 = BenchmarkSample(
            id="x",
            language="python",
            category="null_dereference",
            description="d",
            code="c",
            bug_labels=("a",),
            expected_issues=("b",),
        )
        s2 = BenchmarkSample(
            id="x",
            language="python",
            category="null_dereference",
            description="d",
            code="c",
            bug_labels=("a",),
            expected_issues=("b",),
        )
        assert s1 == s2
        assert hash(s1) == hash(s2)


# ---------------------------------------------------------------------------
# Scoring dataclass smoke tests
# ---------------------------------------------------------------------------


class TestScoringDataclasses:
    """Smoke tests to ensure scoring dataclasses can be instantiated."""

    def test_category_score_creation(self) -> None:
        cs = CategoryScore(category="null_dereference", precision=0.8, recall=0.7, f1=0.747, sample_count=5)
        assert cs.category == "null_dereference"
        assert cs.precision == pytest.approx(0.8)
        assert cs.recall == pytest.approx(0.7)
        assert cs.f1 == pytest.approx(0.747)
        assert cs.sample_count == 5

    def test_model_score_creation(self) -> None:
        cs = CategoryScore(category="sql_injection", precision=0.9, recall=0.85, f1=0.874, sample_count=10)
        ms = ModelScore(
            model_id="gpt4",
            model_name="gpt-4o",
            precision=0.85,
            recall=0.80,
            f1=0.824,
            sample_count=50,
            elapsed_seconds=120.5,
            category_scores=[cs],
        )
        assert ms.model_id == "gpt4"
        assert ms.model_name == "gpt-4o"
        assert len(ms.category_scores) == 1

    def test_sample_result_creation(self) -> None:
        sr = SampleResult(
            sample_id="sample_001",
            model_id="claude",
            identified_issues=["null dereference", "none check"],
            precision=1.0,
            recall=0.67,
            f1=0.8,
            raw_response="The code has a null dereference issue.",
        )
        assert sr.sample_id == "sample_001"
        assert len(sr.identified_issues) == 2

    def test_benchmark_run_creation(self) -> None:
        br = BenchmarkRun(
            run_id="20240101_120000",
            model_scores=[],
            sample_results=[],
            total_samples=50,
            timestamp="2024-01-01T12:00:00",
        )
        assert br.run_id == "20240101_120000"
        assert br.total_samples == 50

    def test_model_score_default_category_scores(self) -> None:
        """ModelScore.category_scores defaults to an empty list."""
        ms = ModelScore(
            model_id="ollama",
            model_name="llama3",
            precision=0.7,
            recall=0.65,
            f1=0.674,
            sample_count=50,
            elapsed_seconds=90.0,
        )
        assert ms.category_scores == []


# ---------------------------------------------------------------------------
# _parse_record internal helper
# ---------------------------------------------------------------------------


class TestParseRecord:
    """Tests for the _parse_record internal helper function."""

    def test_valid_record(self, raw_null_dereference_record: dict[str, Any]) -> None:
        """A valid record dict is parsed into a BenchmarkSample."""
        sample = _parse_record(raw_null_dereference_record, line_number=1)
        assert sample.id == "fixture_001"
        assert sample.language == "python"
        assert sample.category == "null_dereference"
        assert "null_dereference" in sample.bug_labels
        assert "null dereference" in sample.expected_issues

    def test_language_lowercased(self) -> None:
        """Language field is normalised to lowercase."""
        record = {
            "id": "t1",
            "language": "Python",
            "category": "null_dereference",
            "description": "d",
            "code": "x = 1",
            "bug_labels": ["null_dereference"],
            "expected_issues": ["null dereference"],
        }
        sample = _parse_record(record, line_number=1)
        assert sample.language == "python"

    def test_category_lowercased(self) -> None:
        """Category field is normalised to lowercase."""
        record = {
            "id": "t2",
            "language": "python",
            "category": "SQL_Injection",
            "description": "d",
            "code": "x = 1",
            "bug_labels": ["sql_injection"],
            "expected_issues": ["sql injection"],
        }
        sample = _parse_record(record, line_number=1)
        assert sample.category == "sql_injection"

    def test_missing_id_raises(self) -> None:
        """A record without 'id' raises SampleValidationError."""
        record = {
            "language": "python",
            "category": "null_dereference",
            "description": "d",
            "code": "x",
            "bug_labels": ["label"],
            "expected_issues": ["issue"],
        }
        with pytest.raises(SampleValidationError):
            _parse_record(record, line_number=5)

    def test_missing_code_raises(self) -> None:
        """A record without 'code' raises SampleValidationError."""
        record = {
            "id": "t3",
            "language": "python",
            "category": "null_dereference",
            "description": "d",
            "bug_labels": ["label"],
            "expected_issues": ["issue"],
        }
        with pytest.raises(SampleValidationError, match="code"):
            _parse_record(record, line_number=1)

    def test_empty_bug_labels_list_raises(self) -> None:
        """A record with an empty bug_labels list raises SampleValidationError."""
        record = {
            "id": "t4",
            "language": "python",
            "category": "null_dereference",
            "description": "d",
            "code": "x = 1",
            "bug_labels": [],
            "expected_issues": ["issue"],
        }
        with pytest.raises(SampleValidationError, match="bug_labels"):
            _parse_record(record, line_number=1)

    def test_non_string_label_raises(self) -> None:
        """A bug_labels entry that is not a string raises SampleValidationError."""
        record = {
            "id": "t5",
            "language": "python",
            "category": "null_dereference",
            "description": "d",
            "code": "x = 1",
            "bug_labels": [42, "null_dereference"],
            "expected_issues": ["issue"],
        }
        with pytest.raises(SampleValidationError):
            _parse_record(record, line_number=1)

    def test_bug_labels_stored_as_tuple(self, raw_null_dereference_record: dict[str, Any]) -> None:
        """Parsed bug_labels are stored as a tuple on the resulting sample."""
        sample = _parse_record(raw_null_dereference_record, line_number=1)
        assert isinstance(sample.bug_labels, tuple)

    def test_expected_issues_stored_as_tuple(self, raw_null_dereference_record: dict[str, Any]) -> None:
        """Parsed expected_issues are stored as a tuple on the resulting sample."""
        sample = _parse_record(raw_null_dereference_record, line_number=1)
        assert isinstance(sample.expected_issues, tuple)


# ---------------------------------------------------------------------------
# iter_samples
# ---------------------------------------------------------------------------


class TestIterSamples:
    """Tests for the iter_samples generator."""

    def test_iter_valid_file(self, valid_jsonl_file: Path) -> None:
        """Three valid records are yielded from a well-formed JSONL file."""
        samples = list(iter_samples(valid_jsonl_file))
        assert len(samples) == 3
        assert all(isinstance(s, BenchmarkSample) for s in samples)

    def test_iter_ids_in_order(self, valid_jsonl_file: Path) -> None:
        """Samples are yielded in the order they appear in the file."""
        samples = list(iter_samples(valid_jsonl_file))
        assert samples[0].id == "fixture_001"
        assert samples[1].id == "fixture_002"
        assert samples[2].id == "fixture_003"

    def test_iter_skips_blank_and_comment_lines(self, jsonl_file_with_comments: Path) -> None:
        """Blank lines and lines starting with '#' are silently skipped."""
        samples = list(iter_samples(jsonl_file_with_comments))
        assert len(samples) == 1
        assert samples[0].id == "fixture_001"

    def test_iter_empty_file_yields_nothing(self, empty_jsonl_file: Path) -> None:
        """An empty file yields zero samples."""
        samples = list(iter_samples(empty_jsonl_file))
        assert samples == []

    def test_iter_invalid_json_raises_by_default(self, jsonl_file_with_invalid_json: Path) -> None:
        """A malformed JSON line raises SampleValidationError when skip_invalid=False."""
        with pytest.raises(SampleValidationError):
            list(iter_samples(jsonl_file_with_invalid_json))

    def test_iter_invalid_json_skipped_when_flag_set(self, jsonl_file_with_invalid_json: Path) -> None:
        """A malformed JSON line is skipped when skip_invalid=True."""
        samples = list(iter_samples(jsonl_file_with_invalid_json, skip_invalid=True))
        assert len(samples) == 1  # Only the valid first record
        assert samples[0].id == "fixture_001"

    def test_iter_missing_field_raises_by_default(self, jsonl_file_missing_field: Path) -> None:
        """A record missing a required field raises SampleValidationError."""
        with pytest.raises(SampleValidationError):
            list(iter_samples(jsonl_file_missing_field))

    def test_iter_missing_field_skipped_when_flag_set(self, jsonl_file_missing_field: Path) -> None:
        """A record missing a required field is skipped when skip_invalid=True."""
        samples = list(iter_samples(jsonl_file_missing_field, skip_invalid=True))
        assert samples == []

    def test_iter_mixed_valid_invalid_skip(self, jsonl_file_mixed_valid_invalid: Path) -> None:
        """With skip_invalid=True, two valid records are returned despite one broken line."""
        samples = list(iter_samples(jsonl_file_mixed_valid_invalid, skip_invalid=True))
        assert len(samples) == 2
        ids = {s.id for s in samples}
        assert "fixture_001" in ids
        assert "fixture_002" in ids

    def test_iter_nonexistent_file_raises(self, tmp_path: Path) -> None:
        """A SamplesLoadError is raised when the file does not exist."""
        missing = tmp_path / "does_not_exist.jsonl"
        with pytest.raises(SamplesLoadError):
            list(iter_samples(missing))

    def test_iter_accepts_string_path(self, valid_jsonl_file: Path) -> None:
        """iter_samples accepts a plain string path in addition to a Path object."""
        samples = list(iter_samples(str(valid_jsonl_file)))
        assert len(samples) == 3

    def test_iter_yields_benchmark_sample_instances(self, valid_jsonl_file: Path) -> None:
        """Every yielded object is a BenchmarkSample instance."""
        for sample in iter_samples(valid_jsonl_file):
            assert isinstance(sample, BenchmarkSample)


# ---------------------------------------------------------------------------
# load_samples
# ---------------------------------------------------------------------------


class TestLoadSamples:
    """Tests for the load_samples convenience function."""

    def test_load_all_samples(self, valid_jsonl_file: Path) -> None:
        """load_samples returns all records when no filters are given."""
        samples = load_samples(valid_jsonl_file)
        assert len(samples) == 3

    def test_load_returns_list(self, valid_jsonl_file: Path) -> None:
        """load_samples always returns a list."""
        result = load_samples(valid_jsonl_file)
        assert isinstance(result, list)

    def test_filter_by_category(self, valid_jsonl_file: Path) -> None:
        """Filtering by category returns only matching samples."""
        samples = load_samples(valid_jsonl_file, category="sql_injection")
        assert len(samples) == 1
        assert samples[0].category == "sql_injection"

    def test_filter_by_category_case_insensitive(self, valid_jsonl_file: Path) -> None:
        """Category filter is case-insensitive."""
        samples_lower = load_samples(valid_jsonl_file, category="null_dereference")
        samples_upper = load_samples(valid_jsonl_file, category="NULL_DEREFERENCE")
        assert len(samples_lower) == len(samples_upper)

    def test_filter_by_language(self, jsonl_file_multi_language: Path) -> None:
        """Filtering by language returns only matching samples."""
        python_samples = load_samples(jsonl_file_multi_language, language="python")
        assert len(python_samples) == 2
        assert all(s.language == "python" for s in python_samples)

    def test_filter_by_language_javascript(self, jsonl_file_multi_language: Path) -> None:
        """Filtering by language='javascript' returns the JavaScript sample."""
        js_samples = load_samples(jsonl_file_multi_language, language="javascript")
        assert len(js_samples) == 1
        assert js_samples[0].language == "javascript"

    def test_filter_by_language_case_insensitive(self, jsonl_file_multi_language: Path) -> None:
        """Language filter is case-insensitive."""
        samples_lower = load_samples(jsonl_file_multi_language, language="python")
        samples_upper = load_samples(jsonl_file_multi_language, language="PYTHON")
        assert len(samples_lower) == len(samples_upper)

    def test_filter_by_category_and_language(self, jsonl_file_multi_language: Path) -> None:
        """Combined category + language filter returns only matching samples."""
        samples = load_samples(
            jsonl_file_multi_language,
            category="null_dereference",
            language="javascript",
        )
        assert len(samples) == 1
        assert samples[0].language == "javascript"
        assert samples[0].category == "null_dereference"

    def test_filter_no_match_returns_empty(self, valid_jsonl_file: Path) -> None:
        """A filter that matches nothing returns an empty list."""
        samples = load_samples(valid_jsonl_file, category="race_condition")
        assert samples == []

    def test_load_empty_file_returns_empty_list(self, empty_jsonl_file: Path) -> None:
        """Loading an empty file returns an empty list."""
        samples = load_samples(empty_jsonl_file)
        assert samples == []

    def test_load_skip_invalid_propagated(self, jsonl_file_mixed_valid_invalid: Path) -> None:
        """skip_invalid=True is propagated to iter_samples."""
        samples = load_samples(jsonl_file_mixed_valid_invalid, skip_invalid=True)
        assert len(samples) == 2

    def test_load_default_path_exists(self) -> None:
        """The default bundled samples.jsonl file exists and loads at least one sample."""
        samples = load_samples()  # no path arg → uses bundled data/samples.jsonl
        assert len(samples) >= 1
        assert all(isinstance(s, BenchmarkSample) for s in samples)

    def test_load_bundled_samples_count(self) -> None:
        """The bundled dataset contains exactly 50 samples."""
        samples = load_samples()
        assert len(samples) == 50

    def test_load_bundled_samples_categories(self) -> None:
        """All five expected categories are present in the bundled dataset."""
        samples = load_samples()
        categories = {s.category for s in samples}
        expected = {"null_dereference", "off_by_one", "sql_injection", "race_condition", "resource_leak"}
        assert expected == categories


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


class TestGetCategories:
    """Tests for the get_categories utility."""

    def test_returns_sorted_list(self, all_fixture_samples: list[BenchmarkSample]) -> None:
        """get_categories returns a sorted list of unique category names."""
        cats = get_categories(all_fixture_samples)
        assert cats == sorted(cats)

    def test_returns_unique_values(self, all_fixture_samples: list[BenchmarkSample]) -> None:
        """get_categories returns no duplicates."""
        cats = get_categories(all_fixture_samples)
        assert len(cats) == len(set(cats))

    def test_all_five_categories(self, all_fixture_samples: list[BenchmarkSample]) -> None:
        """All five fixture categories are returned."""
        cats = get_categories(all_fixture_samples)
        assert "null_dereference" in cats
        assert "sql_injection" in cats
        assert "off_by_one" in cats
        assert "resource_leak" in cats
        assert "race_condition" in cats

    def test_empty_list_returns_empty(self) -> None:
        """An empty sample list returns an empty category list."""
        assert get_categories([]) == []


class TestGetLanguages:
    """Tests for the get_languages utility."""

    def test_returns_sorted_list(self, all_fixture_samples: list[BenchmarkSample]) -> None:
        """get_languages returns a sorted list."""
        langs = get_languages(all_fixture_samples)
        assert langs == sorted(langs)

    def test_returns_unique_values(self, all_fixture_samples: list[BenchmarkSample]) -> None:
        """get_languages returns no duplicates."""
        langs = get_languages(all_fixture_samples)
        assert len(langs) == len(set(langs))

    def test_python_present(self, all_fixture_samples: list[BenchmarkSample]) -> None:
        """'python' is present in the fixture sample languages."""
        langs = get_languages(all_fixture_samples)
        assert "python" in langs

    def test_empty_list_returns_empty(self) -> None:
        """An empty sample list returns an empty language list."""
        assert get_languages([]) == []


class TestSamplesByCategory:
    """Tests for the samples_by_category utility."""

    def test_groups_by_category(self, all_fixture_samples: list[BenchmarkSample]) -> None:
        """Samples are grouped under their respective category keys."""
        grouped = samples_by_category(all_fixture_samples)
        assert isinstance(grouped, dict)
        for category, group in grouped.items():
            assert all(s.category == category for s in group)

    def test_all_samples_accounted_for(self, all_fixture_samples: list[BenchmarkSample]) -> None:
        """The total number of grouped samples equals the total input count."""
        grouped = samples_by_category(all_fixture_samples)
        total = sum(len(v) for v in grouped.values())
        assert total == len(all_fixture_samples)

    def test_keys_are_sorted(self, all_fixture_samples: list[BenchmarkSample]) -> None:
        """The returned dictionary keys are in sorted order."""
        grouped = samples_by_category(all_fixture_samples)
        keys = list(grouped.keys())
        assert keys == sorted(keys)

    def test_empty_list_returns_empty_dict(self) -> None:
        """An empty input list returns an empty dict."""
        assert samples_by_category([]) == {}

    def test_single_category(self, sample_null_dereference: BenchmarkSample) -> None:
        """A list of samples from one category produces a single-key dict."""
        grouped = samples_by_category([sample_null_dereference])
        assert list(grouped.keys()) == ["null_dereference"]
        assert grouped["null_dereference"] == [sample_null_dereference]

    def test_multiple_samples_same_category(
        self,
        sample_null_dereference: BenchmarkSample,
    ) -> None:
        """Multiple samples with the same category are grouped together."""
        dup = BenchmarkSample(
            id="fixture_001b",
            language="java",
            category="null_dereference",
            description="Another null deref.",
            code="User u = repo.find(id); u.getName();",
            bug_labels=("null_dereference", "NullPointerException"),
            expected_issues=("null check", "NullPointerException"),
        )
        grouped = samples_by_category([sample_null_dereference, dup])
        assert len(grouped["null_dereference"]) == 2
