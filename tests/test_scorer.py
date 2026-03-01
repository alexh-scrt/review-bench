"""Unit tests for the review_bench.scorer module.

Covers:
- Text normalisation helper (_normalise)
- Phrase-in-text matching (_phrase_in_text)
- Issue-to-label alignment (_issue_matches_any_label)
- Per-sample scoring (score_sample) with a variety of response scenarios
- Aggregate scoring (compute_aggregate_score) micro-averaging
- Full model run scoring (score_model_run)
- Per-category score breakdown
- Edge cases: empty responses, perfect scores, zero scores, single sample
- Utility function extract_issues_from_response
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from review_bench.samples import (
    BenchmarkSample,
    CategoryScore,
    ModelScore,
    SampleResult,
)
from review_bench.scorer import (
    _issue_matches_any_label,
    _normalise,
    _phrase_in_text,
    compute_aggregate_score,
    extract_issues_from_response,
    score_model_run,
    score_sample,
)


# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------


def _make_sample(
    sample_id: str = "s001",
    category: str = "null_dereference",
    bug_labels: tuple[str, ...] = ("null_dereference", "missing_null_check"),
    expected_issues: tuple[str, ...] = ("null dereference", "none check", "missing null check"),
    language: str = "python",
) -> BenchmarkSample:
    """Build a minimal BenchmarkSample for use in scorer tests."""
    return BenchmarkSample(
        id=sample_id,
        language=language,
        category=category,
        description="Test sample.",
        code="def f(x): return x.name",
        bug_labels=bug_labels,
        expected_issues=expected_issues,
    )


PERFECT_RESPONSE = (
    "This code has a null dereference issue. "
    "There is a missing null check — the value could be None. "
    "You should add a none check before accessing the attribute."
)

PARTIAL_RESPONSE = (
    "This code has a null dereference issue but I see nothing else wrong."
)

EMPTY_RESPONSE = ""

IRRELEVANT_RESPONSE = "The code looks fine to me. No issues detected."


# ---------------------------------------------------------------------------
# _normalise
# ---------------------------------------------------------------------------


class TestNormalise:
    """Tests for the _normalise text helper."""

    def test_lowercases_text(self) -> None:
        assert _normalise("Hello World") == "hello world"

    def test_replaces_punctuation_with_space(self) -> None:
        result = _normalise("null-dereference!")
        assert "null" in result
        assert "dereference" in result

    def test_collapses_whitespace(self) -> None:
        result = _normalise("  multiple   spaces  ")
        assert "  " not in result
        assert result == "multiple spaces"

    def test_strips_leading_trailing_whitespace(self) -> None:
        assert _normalise("  hello  ") == "hello"

    def test_empty_string(self) -> None:
        assert _normalise("") == ""

    def test_underscores_become_spaces(self) -> None:
        result = _normalise("null_dereference")
        assert result == "null dereference"

    def test_mixed_separators(self) -> None:
        result = _normalise("off-by-one (index)")
        # All non-word chars become spaces
        assert "off" in result
        assert "by" in result
        assert "one" in result
        assert "index" in result

    def test_unicode_text(self) -> None:
        result = _normalise("naïve café")
        # Should not crash and should lowercase
        assert isinstance(result, str)

    def test_numbers_preserved(self) -> None:
        result = _normalise("version 3.11")
        assert "3" in result
        assert "11" in result


# ---------------------------------------------------------------------------
# _phrase_in_text
# ---------------------------------------------------------------------------


class TestPhraseInText:
    """Tests for the _phrase_in_text helper."""

    def test_exact_match(self) -> None:
        assert _phrase_in_text("null dereference", "this code has a null dereference issue")

    def test_phrase_not_present(self) -> None:
        assert not _phrase_in_text("sql injection", "null dereference present")

    def test_empty_phrase(self) -> None:
        # Empty string is always a substring
        assert _phrase_in_text("", "any text")

    def test_empty_text(self) -> None:
        assert not _phrase_in_text("null dereference", "")

    def test_both_empty(self) -> None:
        assert _phrase_in_text("", "")

    def test_partial_word_match(self) -> None:
        # "none" is in "nonexistent" — substring match is intentional
        assert _phrase_in_text("none", "nonexistent variable")

    def test_case_sensitive_after_normalisation(self) -> None:
        # Caller is expected to pass already-normalised strings;
        # both lower-case should match
        assert _phrase_in_text("null dereference", "there is a null dereference here")


# ---------------------------------------------------------------------------
# _issue_matches_any_label
# ---------------------------------------------------------------------------


class TestIssueMatchesAnyLabel:
    """Tests for the _issue_matches_any_label helper."""

    def test_exact_label_in_issue(self) -> None:
        assert _issue_matches_any_label(
            "null dereference", ("null_dereference", "missing_null_check")
        )

    def test_issue_in_label(self) -> None:
        # "null" is a substring of "null_dereference" after normalisation
        assert _issue_matches_any_label("null", ("null_dereference",))

    def test_no_match(self) -> None:
        assert not _issue_matches_any_label(
            "sql injection", ("null_dereference", "missing_null_check")
        )

    def test_empty_labels(self) -> None:
        assert not _issue_matches_any_label("null dereference", ())

    def test_underscore_normalisation(self) -> None:
        # Labels with underscores normalise to spaces before comparison
        assert _issue_matches_any_label(
            "missing null check", ("missing_null_check",)
        )

    def test_case_insensitive(self) -> None:
        assert _issue_matches_any_label(
            "NULL DEREFERENCE", ("null_dereference",)
        )

    def test_partial_label_match(self) -> None:
        assert _issue_matches_any_label(
            "race condition detected", ("race_condition",)
        )

    def test_multi_label_first_matches(self) -> None:
        assert _issue_matches_any_label(
            "resource leak",
            ("null_dereference", "resource_leak", "off_by_one"),
        )

    def test_multi_label_last_matches(self) -> None:
        assert _issue_matches_any_label(
            "off by one error",
            ("null_dereference", "resource_leak", "off_by_one"),
        )


# ---------------------------------------------------------------------------
# score_sample — basic cases
# ---------------------------------------------------------------------------


class TestScoreSampleBasic:
    """Basic correctness tests for score_sample."""

    def test_returns_sample_result(self) -> None:
        sample = _make_sample()
        result = score_sample(sample, PERFECT_RESPONSE)
        assert isinstance(result, SampleResult)

    def test_sample_id_propagated(self) -> None:
        sample = _make_sample(sample_id="test_001")
        result = score_sample(sample, PERFECT_RESPONSE, model_id="gpt4")
        assert result.sample_id == "test_001"

    def test_model_id_propagated(self) -> None:
        sample = _make_sample()
        result = score_sample(sample, PERFECT_RESPONSE, model_id="claude")
        assert result.model_id == "claude"

    def test_raw_response_preserved(self) -> None:
        sample = _make_sample()
        result = score_sample(sample, PERFECT_RESPONSE)
        assert result.raw_response == PERFECT_RESPONSE

    def test_identified_issues_is_list(self) -> None:
        sample = _make_sample()
        result = score_sample(sample, PERFECT_RESPONSE)
        assert isinstance(result.identified_issues, list)

    def test_precision_in_range(self) -> None:
        sample = _make_sample()
        result = score_sample(sample, PERFECT_RESPONSE)
        assert 0.0 <= result.precision <= 1.0

    def test_recall_in_range(self) -> None:
        sample = _make_sample()
        result = score_sample(sample, PERFECT_RESPONSE)
        assert 0.0 <= result.recall <= 1.0

    def test_f1_in_range(self) -> None:
        sample = _make_sample()
        result = score_sample(sample, PERFECT_RESPONSE)
        assert 0.0 <= result.f1 <= 1.0

    def test_default_model_id_empty_string(self) -> None:
        sample = _make_sample()
        result = score_sample(sample, PERFECT_RESPONSE)
        assert result.model_id == ""


# ---------------------------------------------------------------------------
# score_sample — perfect score
# ---------------------------------------------------------------------------


class TestScoreSamplePerfect:
    """Tests where the model response contains all expected issues."""

    def test_recall_is_one_when_all_issues_mentioned(self) -> None:
        sample = _make_sample(
            expected_issues=("null dereference",),
            bug_labels=("null_dereference",),
        )
        response = "This code has a null dereference problem."
        result = score_sample(sample, response)
        assert result.recall == pytest.approx(1.0)

    def test_precision_is_one_when_all_issues_correct(self) -> None:
        sample = _make_sample(
            expected_issues=("null dereference",),
            bug_labels=("null_dereference",),
        )
        response = "This code has a null dereference problem."
        result = score_sample(sample, response)
        assert result.precision == pytest.approx(1.0)

    def test_f1_is_one_when_perfect(self) -> None:
        sample = _make_sample(
            expected_issues=("null dereference",),
            bug_labels=("null_dereference",),
        )
        response = "This code has a null dereference problem."
        result = score_sample(sample, response)
        assert result.f1 == pytest.approx(1.0)

    def test_all_issues_in_identified(self) -> None:
        sample = _make_sample(
            expected_issues=("null dereference", "none check"),
            bug_labels=("null_dereference", "missing_null_check"),
        )
        response = "null dereference detected; add a none check"
        result = score_sample(sample, response)
        assert set(result.identified_issues) == {"null dereference", "none check"}


# ---------------------------------------------------------------------------
# score_sample — zero score
# ---------------------------------------------------------------------------


class TestScoreSampleZero:
    """Tests where the model response matches nothing."""

    def test_empty_response_recall_zero(self) -> None:
        sample = _make_sample()
        result = score_sample(sample, EMPTY_RESPONSE)
        assert result.recall == pytest.approx(0.0)

    def test_empty_response_precision_zero(self) -> None:
        sample = _make_sample()
        result = score_sample(sample, EMPTY_RESPONSE)
        assert result.precision == pytest.approx(0.0)

    def test_empty_response_f1_zero(self) -> None:
        sample = _make_sample()
        result = score_sample(sample, EMPTY_RESPONSE)
        assert result.f1 == pytest.approx(0.0)

    def test_empty_response_no_identified_issues(self) -> None:
        sample = _make_sample()
        result = score_sample(sample, EMPTY_RESPONSE)
        assert result.identified_issues == []

    def test_irrelevant_response_recall_zero(self) -> None:
        sample = _make_sample(
            expected_issues=("null dereference", "none check", "missing null check"),
            bug_labels=("null_dereference",),
        )
        result = score_sample(sample, IRRELEVANT_RESPONSE)
        assert result.recall == pytest.approx(0.0)

    def test_irrelevant_response_f1_zero(self) -> None:
        sample = _make_sample(
            expected_issues=("null dereference", "none check", "missing null check"),
            bug_labels=("null_dereference",),
        )
        result = score_sample(sample, IRRELEVANT_RESPONSE)
        assert result.f1 == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# score_sample — partial score
# ---------------------------------------------------------------------------


class TestScoreSamplePartial:
    """Tests where the model response matches some but not all expected issues."""

    def test_partial_recall(self) -> None:
        # PARTIAL_RESPONSE only mentions "null dereference" (1 of 3 issues)
        sample = _make_sample(
            expected_issues=("null dereference", "none check", "missing null check"),
            bug_labels=("null_dereference", "missing_null_check"),
        )
        result = score_sample(sample, PARTIAL_RESPONSE)
        assert 0.0 < result.recall < 1.0

    def test_partial_recall_exact_value(self) -> None:
        sample = _make_sample(
            expected_issues=("null dereference", "none check", "missing null check"),
            bug_labels=("null_dereference", "missing_null_check"),
        )
        result = score_sample(sample, PARTIAL_RESPONSE)
        # 1 out of 3 expected issues matched
        assert result.recall == pytest.approx(1 / 3, rel=1e-4)

    def test_partial_f1_between_zero_and_one(self) -> None:
        sample = _make_sample(
            expected_issues=("null dereference", "none check"),
            bug_labels=("null_dereference",),
        )
        result = score_sample(sample, PARTIAL_RESPONSE)
        assert 0.0 <= result.f1 <= 1.0

    def test_f1_is_harmonic_mean(self) -> None:
        sample = _make_sample(
            expected_issues=("null dereference",),
            bug_labels=("null_dereference",),
        )
        result = score_sample(sample, PARTIAL_RESPONSE)
        p, r = result.precision, result.recall
        if p + r > 0:
            expected_f1 = 2 * p * r / (p + r)
            assert result.f1 == pytest.approx(expected_f1, rel=1e-4)

    def test_identified_issues_subset(self) -> None:
        sample = _make_sample(
            expected_issues=("null dereference", "none check", "missing null check"),
            bug_labels=("null_dereference", "missing_null_check"),
        )
        result = score_sample(sample, PARTIAL_RESPONSE)
        # Only "null dereference" appears in PARTIAL_RESPONSE
        assert "null dereference" in result.identified_issues
        assert "none check" not in result.identified_issues


# ---------------------------------------------------------------------------
# score_sample — SQL injection sample (from conftest fixture)
# ---------------------------------------------------------------------------


class TestScoreSampleSqlInjection:
    """Tests using the SQL injection sample fixture."""

    def test_sql_injection_perfect_response(self, sample_sql_injection: BenchmarkSample) -> None:
        response = (
            "This code is vulnerable to sql injection because the query uses "
            "string concatenation with unsanitized user input. "
            "Use a parameterized query instead."
        )
        result = score_sample(sample_sql_injection, response, model_id="gpt4")
        assert result.recall > 0.5

    def test_sql_injection_empty_response(self, sample_sql_injection: BenchmarkSample) -> None:
        result = score_sample(sample_sql_injection, "", model_id="gpt4")
        assert result.recall == 0.0
        assert result.precision == 0.0
        assert result.f1 == 0.0

    def test_sql_injection_sample_id_correct(self, sample_sql_injection: BenchmarkSample) -> None:
        result = score_sample(sample_sql_injection, "sql injection found", model_id="claude")
        assert result.sample_id == sample_sql_injection.id


# ---------------------------------------------------------------------------
# score_sample — race condition sample
# ---------------------------------------------------------------------------


class TestScoreSampleRaceCondition:
    """Tests using the race condition sample fixture."""

    def test_race_condition_identified(self, sample_race_condition: BenchmarkSample) -> None:
        response = "There is a race condition here. The counter lacks thread safety and a lock."
        result = score_sample(sample_race_condition, response)
        assert "race condition" in result.identified_issues
        assert result.recall > 0.0

    def test_no_race_mentioned(self, sample_race_condition: BenchmarkSample) -> None:
        result = score_sample(sample_race_condition, "The code looks fine.")
        assert result.recall == 0.0


# ---------------------------------------------------------------------------
# score_sample — resource leak sample
# ---------------------------------------------------------------------------


class TestScoreSampleResourceLeak:
    """Tests using the resource leak sample fixture."""

    def test_resource_leak_identified(self, sample_resource_leak: BenchmarkSample) -> None:
        response = "There is a resource leak because the file is not closed; use a context manager."
        result = score_sample(sample_resource_leak, response)
        assert result.recall > 0.0
        assert "resource leak" in result.identified_issues


# ---------------------------------------------------------------------------
# score_sample — off-by-one sample
# ---------------------------------------------------------------------------


class TestScoreSampleOffByOne:
    """Tests using the off-by-one sample fixture."""

    def test_off_by_one_identified(self, sample_off_by_one: BenchmarkSample) -> None:
        response = "This is an off by one error; the loop goes out of range due to the wrong loop bound."
        result = score_sample(sample_off_by_one, response)
        assert "off by one" in result.identified_issues
        assert result.recall == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compute_aggregate_score
# ---------------------------------------------------------------------------


class TestComputeAggregateScore:
    """Tests for the compute_aggregate_score function."""

    def _build_results_and_map(
        self, samples: list[BenchmarkSample], responses: dict[str, str], model_id: str = "m1"
    ) -> tuple[list[SampleResult], dict[str, BenchmarkSample]]:
        """Helper: build SampleResult list and samples_by_id mapping."""
        samples_by_id = {s.id: s for s in samples}
        results = [score_sample(s, responses.get(s.id, ""), model_id=model_id) for s in samples]
        return results, samples_by_id

    def test_returns_model_score(self, all_fixture_samples: list[BenchmarkSample]) -> None:
        responses = {s.id: "" for s in all_fixture_samples}
        results, smap = self._build_results_and_map(all_fixture_samples, responses)
        ms = compute_aggregate_score(results, smap, model_id="m1", model_name="Model1")
        assert isinstance(ms, ModelScore)

    def test_model_id_propagated(self, all_fixture_samples: list[BenchmarkSample]) -> None:
        responses = {s.id: "" for s in all_fixture_samples}
        results, smap = self._build_results_and_map(all_fixture_samples, responses)
        ms = compute_aggregate_score(results, smap, model_id="gpt4", model_name="gpt-4o")
        assert ms.model_id == "gpt4"

    def test_model_name_propagated(self, all_fixture_samples: list[BenchmarkSample]) -> None:
        responses = {s.id: "" for s in all_fixture_samples}
        results, smap = self._build_results_and_map(all_fixture_samples, responses)
        ms = compute_aggregate_score(results, smap, model_id="gpt4", model_name="gpt-4o")
        assert ms.model_name == "gpt-4o"

    def test_elapsed_seconds_propagated(self, all_fixture_samples: list[BenchmarkSample]) -> None:
        responses = {s.id: "" for s in all_fixture_samples}
        results, smap = self._build_results_and_map(all_fixture_samples, responses)
        ms = compute_aggregate_score(
            results, smap, model_id="m", model_name="M", elapsed_seconds=99.5
        )
        assert ms.elapsed_seconds == pytest.approx(99.5)

    def test_sample_count_correct(self, all_fixture_samples: list[BenchmarkSample]) -> None:
        responses = {s.id: "" for s in all_fixture_samples}
        results, smap = self._build_results_and_map(all_fixture_samples, responses)
        ms = compute_aggregate_score(results, smap, model_id="m", model_name="M")
        assert ms.sample_count == len(all_fixture_samples)

    def test_all_empty_responses_zero_scores(self, all_fixture_samples: list[BenchmarkSample]) -> None:
        responses = {s.id: "" for s in all_fixture_samples}
        results, smap = self._build_results_and_map(all_fixture_samples, responses)
        ms = compute_aggregate_score(results, smap, model_id="m", model_name="M")
        assert ms.precision == pytest.approx(0.0)
        assert ms.recall == pytest.approx(0.0)
        assert ms.f1 == pytest.approx(0.0)

    def test_perfect_responses_high_scores(
        self, all_fixture_samples: list[BenchmarkSample]
    ) -> None:
        # Build responses that mention all expected issues for each sample.
        responses: dict[str, str] = {}
        for sample in all_fixture_samples:
            responses[sample.id] = " ".join(sample.expected_issues)
        results, smap = self._build_results_and_map(all_fixture_samples, responses)
        ms = compute_aggregate_score(results, smap, model_id="m", model_name="M")
        assert ms.recall == pytest.approx(1.0)
        assert ms.f1 > 0.0

    def test_category_scores_present(self, all_fixture_samples: list[BenchmarkSample]) -> None:
        responses = {s.id: " ".join(s.expected_issues) for s in all_fixture_samples}
        results, smap = self._build_results_and_map(all_fixture_samples, responses)
        ms = compute_aggregate_score(results, smap, model_id="m", model_name="M")
        assert len(ms.category_scores) > 0
        assert all(isinstance(cs, CategoryScore) for cs in ms.category_scores)

    def test_category_scores_sorted(self, all_fixture_samples: list[BenchmarkSample]) -> None:
        responses = {s.id: "" for s in all_fixture_samples}
        results, smap = self._build_results_and_map(all_fixture_samples, responses)
        ms = compute_aggregate_score(results, smap, model_id="m", model_name="M")
        cats = [cs.category for cs in ms.category_scores]
        assert cats == sorted(cats)

    def test_category_scores_cover_all_categories(
        self, all_fixture_samples: list[BenchmarkSample]
    ) -> None:
        responses = {s.id: "" for s in all_fixture_samples}
        results, smap = self._build_results_and_map(all_fixture_samples, responses)
        ms = compute_aggregate_score(results, smap, model_id="m", model_name="M")
        cat_names = {cs.category for cs in ms.category_scores}
        expected_cats = {s.category for s in all_fixture_samples}
        assert cat_names == expected_cats

    def test_category_sample_counts_correct(
        self, all_fixture_samples: list[BenchmarkSample]
    ) -> None:
        responses = {s.id: "" for s in all_fixture_samples}
        results, smap = self._build_results_and_map(all_fixture_samples, responses)
        ms = compute_aggregate_score(results, smap, model_id="m", model_name="M")
        # Each fixture sample has a unique category -> each category bucket = 1 sample
        for cs in ms.category_scores:
            assert cs.sample_count == 1

    def test_precision_recall_in_range(
        self, all_fixture_samples: list[BenchmarkSample]
    ) -> None:
        responses = {s.id: " ".join(s.expected_issues) for s in all_fixture_samples}
        results, smap = self._build_results_and_map(all_fixture_samples, responses)
        ms = compute_aggregate_score(results, smap, model_id="m", model_name="M")
        assert 0.0 <= ms.precision <= 1.0
        assert 0.0 <= ms.recall <= 1.0
        assert 0.0 <= ms.f1 <= 1.0

    def test_empty_results_list(self) -> None:
        ms = compute_aggregate_score([], {}, model_id="m", model_name="M")
        assert ms.sample_count == 0
        assert ms.precision == 0.0
        assert ms.recall == 0.0
        assert ms.f1 == 0.0
        assert ms.category_scores == []

    def test_unknown_sample_id_skipped(self) -> None:
        """A SampleResult whose sample_id is absent from samples_by_id is skipped."""
        orphan = SampleResult(
            sample_id="unknown_999",
            model_id="m",
            identified_issues=["null dereference"],
            precision=1.0,
            recall=1.0,
            f1=1.0,
            raw_response="null dereference",
        )
        ms = compute_aggregate_score([orphan], {}, model_id="m", model_name="M")
        # sample is skipped -> no contribution
        assert ms.sample_count == 0

    def test_single_sample_aggregate_matches_individual(
        self, sample_null_dereference: BenchmarkSample
    ) -> None:
        response = "null dereference none check missing null check"
        individual = score_sample(sample_null_dereference, response, model_id="m")
        results = [individual]
        smap = {sample_null_dereference.id: sample_null_dereference}
        ms = compute_aggregate_score(results, smap, model_id="m", model_name="M")
        # Aggregate recall should equal individual recall when there's one sample
        assert ms.recall == pytest.approx(individual.recall, rel=1e-3)


# ---------------------------------------------------------------------------
# score_model_run
# ---------------------------------------------------------------------------


class TestScoreModelRun:
    """Tests for the high-level score_model_run function."""

    def test_returns_tuple(self, all_fixture_samples: list[BenchmarkSample]) -> None:
        responses = {s.id: "" for s in all_fixture_samples}
        result = score_model_run(
            all_fixture_samples, responses, model_id="gpt4", model_name="gpt-4o"
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_sample_results_list(self, all_fixture_samples: list[BenchmarkSample]) -> None:
        responses = {s.id: "" for s in all_fixture_samples}
        sample_results, _ = score_model_run(
            all_fixture_samples, responses, model_id="gpt4", model_name="gpt-4o"
        )
        assert isinstance(sample_results, list)
        assert all(isinstance(r, SampleResult) for r in sample_results)

    def test_returns_model_score(self, all_fixture_samples: list[BenchmarkSample]) -> None:
        responses = {s.id: "" for s in all_fixture_samples}
        _, model_score = score_model_run(
            all_fixture_samples, responses, model_id="gpt4", model_name="gpt-4o"
        )
        assert isinstance(model_score, ModelScore)

    def test_sample_result_count(self, all_fixture_samples: list[BenchmarkSample]) -> None:
        responses = {s.id: "" for s in all_fixture_samples}
        sample_results, _ = score_model_run(
            all_fixture_samples, responses, model_id="m", model_name="M"
        )
        assert len(sample_results) == len(all_fixture_samples)

    def test_missing_response_defaults_to_empty(self, sample_null_dereference: BenchmarkSample) -> None:
        # Provide no response for the sample -> empty response -> zero scores
        sample_results, ms = score_model_run(
            [sample_null_dereference], {}, model_id="m", model_name="M"
        )
        assert sample_results[0].recall == 0.0

    def test_elapsed_seconds_propagated(self, all_fixture_samples: list[BenchmarkSample]) -> None:
        responses = {s.id: "" for s in all_fixture_samples}
        _, ms = score_model_run(
            all_fixture_samples,
            responses,
            model_id="m",
            model_name="M",
            elapsed_seconds=42.7,
        )
        assert ms.elapsed_seconds == pytest.approx(42.7)

    def test_model_id_in_results(self, all_fixture_samples: list[BenchmarkSample]) -> None:
        responses = {s.id: "" for s in all_fixture_samples}
        sample_results, ms = score_model_run(
            all_fixture_samples, responses, model_id="claude", model_name="claude-3-opus"
        )
        assert all(r.model_id == "claude" for r in sample_results)
        assert ms.model_id == "claude"

    def test_perfect_run_full_recall(self, all_fixture_samples: list[BenchmarkSample]) -> None:
        responses = {s.id: " ".join(s.expected_issues) for s in all_fixture_samples}
        _, ms = score_model_run(
            all_fixture_samples, responses, model_id="m", model_name="M"
        )
        assert ms.recall == pytest.approx(1.0)

    def test_empty_samples_list(self) -> None:
        sample_results, ms = score_model_run([], {}, model_id="m", model_name="M")
        assert sample_results == []
        assert ms.sample_count == 0
        assert ms.f1 == 0.0

    def test_category_scores_in_model_score(self, all_fixture_samples: list[BenchmarkSample]) -> None:
        responses = {s.id: " ".join(s.expected_issues) for s in all_fixture_samples}
        _, ms = score_model_run(
            all_fixture_samples, responses, model_id="m", model_name="M"
        )
        assert len(ms.category_scores) == len({s.category for s in all_fixture_samples})


# ---------------------------------------------------------------------------
# extract_issues_from_response
# ---------------------------------------------------------------------------


class TestExtractIssuesFromResponse:
    """Tests for the extract_issues_from_response utility function."""

    def test_returns_list(self) -> None:
        result = extract_issues_from_response("some text", ["null dereference"])
        assert isinstance(result, list)

    def test_matches_present_phrase(self) -> None:
        result = extract_issues_from_response(
            "There is a null dereference here.",
            ["null dereference", "sql injection"],
        )
        assert "null dereference" in result
        assert "sql injection" not in result

    def test_empty_response_returns_empty(self) -> None:
        result = extract_issues_from_response("", ["null dereference"])
        assert result == []

    def test_empty_vocabulary_returns_empty(self) -> None:
        result = extract_issues_from_response("null dereference found", [])
        assert result == []

    def test_preserves_vocabulary_order(self) -> None:
        vocab = ["race condition", "null dereference", "sql injection"]
        response = "sql injection and null dereference and race condition"
        result = extract_issues_from_response(response, vocab)
        # All three found; order should match vocab order
        assert result == vocab

    def test_no_matches_returns_empty(self) -> None:
        result = extract_issues_from_response(
            "Everything looks great!", ["null dereference", "sql injection"]
        )
        assert result == []

    def test_case_insensitive_matching(self) -> None:
        result = extract_issues_from_response(
            "NULL DEREFERENCE detected", ["null dereference"]
        )
        assert "null dereference" in result

    def test_partial_phrase_match(self) -> None:
        # "null" alone should match "null dereference" as a substring
        result = extract_issues_from_response(
            "null pointer access", ["null"]
        )
        assert "null" in result

    def test_duplicate_vocabulary_entries(self) -> None:
        # Duplicates in vocab are preserved as-is (each checked independently)
        result = extract_issues_from_response(
            "null dereference", ["null dereference", "null dereference"]
        )
        assert len(result) == 2

    def test_multiword_phrase_exact(self) -> None:
        result = extract_issues_from_response(
            "You should add a parameterized query",
            ["parameterized query", "sql injection"],
        )
        assert "parameterized query" in result
        assert "sql injection" not in result


# ---------------------------------------------------------------------------
# F1 mathematical property tests
# ---------------------------------------------------------------------------


class TestF1MathematicalProperties:
    """Verify that F1 satisfies its mathematical definition."""

    @pytest.mark.parametrize(
        "expected_issues, response",
        [
            (("null dereference",), "null dereference present"),
            (("sql injection", "parameterized"), "sql injection risk; use parameterized query"),
            (("race condition", "thread safety", "lock"), "race condition: add lock for thread safety"),
        ],
    )
    def test_f1_equals_harmonic_mean(self, expected_issues: tuple, response: str) -> None:
        """F1 must equal the harmonic mean of precision and recall."""
        bug_labels = tuple(e.replace(" ", "_") for e in expected_issues)
        sample = _make_sample(expected_issues=expected_issues, bug_labels=bug_labels)
        result = score_sample(sample, response)
        p, r, f1 = result.precision, result.recall, result.f1
        if p + r > 0:
            expected_f1 = 2 * p * r / (p + r)
            assert f1 == pytest.approx(expected_f1, rel=1e-4)
        else:
            assert f1 == pytest.approx(0.0)

    def test_f1_zero_when_precision_zero(self) -> None:
        sample = _make_sample(
            expected_issues=("null dereference",),
            bug_labels=("null_dereference",),
        )
        result = score_sample(sample, "the code looks completely fine")
        assert result.f1 == pytest.approx(0.0)

    def test_f1_never_exceeds_precision_or_recall(self) -> None:
        sample = _make_sample()
        result = score_sample(sample, PARTIAL_RESPONSE)
        assert result.f1 <= result.precision + 1e-9
        assert result.f1 <= result.recall + 1e-9

    def test_f1_symmetric_in_balanced_case(self) -> None:
        """When precision == recall, F1 should equal them."""
        sample = _make_sample(
            expected_issues=("null dereference",),
            bug_labels=("null_dereference",),
        )
        response = "null dereference here"
        result = score_sample(sample, response)
        if math.isclose(result.precision, result.recall, rel_tol=1e-6):
            assert result.f1 == pytest.approx(result.precision, rel=1e-4)


# ---------------------------------------------------------------------------
# Regression / integration tests against bundled dataset
# ---------------------------------------------------------------------------


class TestScorerWithBundledData:
    """Integration tests that exercise the scorer against the bundled samples.jsonl."""

    def test_score_all_bundled_samples_no_crash(self) -> None:
        """Scoring all 50 bundled samples with empty responses should not raise."""
        from review_bench.samples import load_samples

        samples = load_samples()
        for sample in samples:
            result = score_sample(sample, "", model_id="test")
            assert isinstance(result, SampleResult)

    def test_score_all_bundled_samples_perfect_response(self) -> None:
        """Bundled samples scored with their own expected_issues yield recall=1.0."""
        from review_bench.samples import load_samples

        samples = load_samples()
        for sample in samples:
            response = " ".join(sample.expected_issues)
            result = score_sample(sample, response, model_id="oracle")
            assert result.recall == pytest.approx(1.0), (
                f"sample {sample.id}: expected recall=1.0, got {result.recall}"
            )

    def test_full_model_run_bundled_samples(self) -> None:
        """score_model_run against the bundled dataset returns sane aggregate scores."""
        from review_bench.samples import load_samples

        samples = load_samples()
        responses = {s.id: " ".join(s.expected_issues) for s in samples}
        sample_results, ms = score_model_run(
            samples, responses, model_id="oracle", model_name="Oracle"
        )
        assert ms.sample_count == 50
        assert ms.recall == pytest.approx(1.0)
        assert ms.f1 > 0.0
        assert len(ms.category_scores) == 5
