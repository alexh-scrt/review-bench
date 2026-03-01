"""Scoring engine for review_bench.

Computes precision, recall, and F1 metrics by matching model-identified issue
keywords against ground-truth bug labels and expected issue phrases. Supports
both per-sample scoring and aggregate/per-category rollups.

Scoring methodology
-------------------
For each benchmark sample the model produces a free-text review.  The scorer
normalises that text and checks how many of the sample's ``expected_issues``
phrases appear in it (recall) and how many of the phrases the model claimed
actually correspond to real ground-truth labels (precision).

Precision and recall are then micro-averaged across all samples (or all samples
in a category) to produce aggregate scores.
"""

from __future__ import annotations

import re
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

from review_bench.samples import (
    BenchmarkSample,
    CategoryScore,
    ModelScore,
    SampleResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text normalisation helpers
# ---------------------------------------------------------------------------

# Characters that should be treated as token separators during normalisation.
_SEPARATOR_RE = re.compile(r"[^\w]+", re.UNICODE)


def _normalise(text: str) -> str:
    """Return a lower-cased, whitespace-collapsed version of *text*.

    Punctuation is replaced by a single space so that phrase matching works
    regardless of minor formatting differences between providers.

    Args:
        text: Raw string to normalise.

    Returns:
        Normalised string with lower-case letters and single spaces.
    """
    return _SEPARATOR_RE.sub(" ", text.lower()).strip()


def _phrase_in_text(phrase: str, normalised_text: str) -> bool:
    """Return ``True`` if *phrase* appears as a substring of *normalised_text*.

    Both *phrase* and *normalised_text* should already be normalised via
    :func:`_normalise` before calling this function.

    Args:
        phrase: The expected-issue or bug-label phrase to search for.
        normalised_text: The model's review text after normalisation.

    Returns:
        ``True`` when the phrase is found in the text.
    """
    return phrase in normalised_text


# ---------------------------------------------------------------------------
# Per-sample scoring
# ---------------------------------------------------------------------------


def score_sample(
    sample: BenchmarkSample,
    model_response: str,
    model_id: str = "",
) -> SampleResult:
    """Score a single model response against a benchmark sample's ground truth.

    The function checks which of the sample's ``expected_issues`` phrases appear
    in the model's normalised review text (driving **recall**), and which of
    the phrases the model mentioned are confirmed by the sample's ``bug_labels``
    (driving **precision**).

    Precision / recall / F1 calculation
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    *   ``recall``    = matched_expected / total_expected
    *   ``precision`` = matched_expected / total_identified  (see note below)
    *   ``f1``        = harmonic mean of precision and recall

    .. note::
       "identified issues" are the expected-issue phrases the model actually
       mentioned.  We use the *expected_issues* vocabulary as the universe of
       possible identifiable items; this gives a closed-set evaluation that is
       fair across all models without requiring further NLP.

    Edge cases:
    - If the model response is empty, precision = recall = f1 = 0.0.
    - If there are no expected issues (shouldn't happen with valid samples),
      precision = recall = f1 = 0.0.

    Args:
        sample: The :class:`~review_bench.samples.BenchmarkSample` to evaluate.
        model_response: The raw text returned by the model's review.
        model_id: Optional model identifier string for the result record.

    Returns:
        A :class:`~review_bench.samples.SampleResult` with computed metrics.
    """
    normalised_response = _normalise(model_response)

    expected_issues: tuple[str, ...] = sample.expected_issues
    total_expected = len(expected_issues)

    if total_expected == 0 or not normalised_response:
        return SampleResult(
            sample_id=sample.id,
            model_id=model_id,
            identified_issues=[],
            precision=0.0,
            recall=0.0,
            f1=0.0,
            raw_response=model_response,
        )

    # Determine which expected-issue phrases appear in the model response.
    matched_issues: list[str] = [
        issue
        for issue in expected_issues
        if _phrase_in_text(_normalise(issue), normalised_response)
    ]

    total_identified = len(matched_issues)
    total_matched = total_identified  # every "identified" issue is a true positive

    recall = total_matched / total_expected
    # Precision: among issues the model surfaced, how many were correct?
    # Since we only count phrases from the ground-truth vocabulary, all
    # identified phrases are true positives -> precision = TP / (TP + FP).
    # Here TP = total_matched, FP = 0 (by definition of the closed vocabulary).
    # However, to make precision < 1 when the model finds only a subset we
    # measure it as: TP / total_expected (same as recall) unless we also want
    # to capture false positives from the bug_labels dimension.
    #
    # We measure false positives by checking bug_labels not mentioned:
    # a "false positive" is an expected_issue the model explicitly mentioned
    # that does NOT appear in bug_labels (expanded check).
    #
    # Implementation: for each matched expected_issue, check if ANY bug_label
    # is a substring of that issue phrase OR vice-versa.
    true_positives = sum(
        1
        for issue in matched_issues
        if _issue_matches_any_label(issue, sample.bug_labels)
    )

    # False positives: matched issues that don't align to any bug_label.
    false_positives = total_identified - true_positives

    if total_identified == 0:
        precision = 0.0
    else:
        precision = true_positives / total_identified

    # Also guard against the edge case where both are zero.
    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)

    logger.debug(
        "sample=%s model=%s tp=%d fp=%d total_expected=%d "
        "precision=%.3f recall=%.3f f1=%.3f",
        sample.id,
        model_id,
        true_positives,
        false_positives,
        total_expected,
        precision,
        recall,
        f1,
    )

    return SampleResult(
        sample_id=sample.id,
        model_id=model_id,
        identified_issues=matched_issues,
        precision=round(precision, 6),
        recall=round(recall, 6),
        f1=round(f1, 6),
        raw_response=model_response,
    )


def _issue_matches_any_label(issue: str, bug_labels: tuple[str, ...]) -> bool:
    """Return ``True`` if *issue* semantically aligns with any of *bug_labels*.

    The check is bidirectional substring matching after normalisation:
    - the normalised issue contains a normalised label, OR
    - a normalised label contains the normalised issue.

    This handles cases where issue phrases are more verbose than the short
    label tokens (e.g. ``"null dereference"`` matches ``"null_dereference"``
    after normalisation).

    Args:
        issue: A matched expected-issue phrase.
        bug_labels: Tuple of ground-truth bug label strings from the sample.

    Returns:
        ``True`` when at least one label aligns with the issue.
    """
    normalised_issue = _normalise(issue)
    for label in bug_labels:
        normalised_label = _normalise(label)
        if normalised_label in normalised_issue or normalised_issue in normalised_label:
            return True
    return False


# ---------------------------------------------------------------------------
# Aggregate scoring
# ---------------------------------------------------------------------------


@dataclass
 class _AccumulatorBucket:
    """Internal accumulator for micro-averaging precision/recall."""

    true_positives: int = 0
    false_positives: int = 0
    total_expected: int = 0
    sample_count: int = 0


def _bucket_to_scores(bucket: _AccumulatorBucket) -> tuple[float, float, float]:
    """Convert an :class:`_AccumulatorBucket` to (precision, recall, f1).

    Uses micro-averaged metrics:
    - precision = sum(TP) / sum(TP + FP)
    - recall    = sum(TP) / sum(total_expected)

    Args:
        bucket: Filled accumulator bucket.

    Returns:
        Tuple of (precision, recall, f1) floats.
    """
    tp = bucket.true_positives
    fp = bucket.false_positives
    total_exp = bucket.total_expected

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / total_exp if total_exp > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return round(precision, 6), round(recall, 6), round(f1, 6)


def compute_aggregate_score(
    sample_results: Sequence[SampleResult],
    samples_by_id: dict[str, BenchmarkSample],
    model_id: str,
    model_name: str,
    elapsed_seconds: float = 0.0,
) -> ModelScore:
    """Compute aggregate and per-category scores from a list of sample results.

    Uses micro-averaging: statistics are accumulated across all samples before
    the final division, giving equal weight to each *instance* rather than
    each *category*.

    Args:
        sample_results: Sequence of :class:`SampleResult` objects for one model.
        samples_by_id: Mapping from sample ID to the original
            :class:`BenchmarkSample`, used to look up category and ground truth.
        model_id: Short identifier for the model (e.g. ``"gpt4"``).
        model_name: Full provider model name (e.g. ``"gpt-4o"``).
        elapsed_seconds: Total wall-clock time taken for the model's run.

    Returns:
        A :class:`~review_bench.samples.ModelScore` with aggregate and
        per-category breakdowns.
    """
    overall = _AccumulatorBucket()
    category_buckets: dict[str, _AccumulatorBucket] = defaultdict(_AccumulatorBucket)

    for result in sample_results:
        sample = samples_by_id.get(result.sample_id)
        if sample is None:
            logger.warning(
                "sample_id '%s' not found in samples_by_id; skipping.",
                result.sample_id,
            )
            continue

        category = sample.category
        total_expected = len(sample.expected_issues)

        # Re-derive TP/FP from the result's identified_issues list.
        identified = result.identified_issues
        tp = sum(
            1
            for issue in identified
            if _issue_matches_any_label(issue, sample.bug_labels)
        )
        fp = len(identified) - tp

        # Update overall bucket.
        overall.true_positives += tp
        overall.false_positives += fp
        overall.total_expected += total_expected
        overall.sample_count += 1

        # Update category bucket.
        bucket = category_buckets[category]
        bucket.true_positives += tp
        bucket.false_positives += fp
        bucket.total_expected += total_expected
        bucket.sample_count += 1

    agg_precision, agg_recall, agg_f1 = _bucket_to_scores(overall)

    category_scores: list[CategoryScore] = []
    for category in sorted(category_buckets):
        bucket = category_buckets[category]
        cat_precision, cat_recall, cat_f1 = _bucket_to_scores(bucket)
        category_scores.append(
            CategoryScore(
                category=category,
                precision=cat_precision,
                recall=cat_recall,
                f1=cat_f1,
                sample_count=bucket.sample_count,
            )
        )

    return ModelScore(
        model_id=model_id,
        model_name=model_name,
        precision=agg_precision,
        recall=agg_recall,
        f1=agg_f1,
        sample_count=overall.sample_count,
        elapsed_seconds=elapsed_seconds,
        category_scores=category_scores,
    )


# ---------------------------------------------------------------------------
# Convenience: score a whole model run in one call
# ---------------------------------------------------------------------------


def score_model_run(
    samples: Sequence[BenchmarkSample],
    responses: dict[str, str],
    model_id: str,
    model_name: str,
    elapsed_seconds: float = 0.0,
) -> tuple[list[SampleResult], ModelScore]:
    """Score a complete model run from raw responses and return all artefacts.

    This is the primary high-level entry point used by the benchmark
    orchestrator.  It calls :func:`score_sample` for each sample, then
    :func:`compute_aggregate_score` to produce the :class:`ModelScore`.

    Args:
        samples: All benchmark samples that were evaluated.
        responses: Mapping from ``sample.id`` to the model's raw review text.
        model_id: Short identifier for the model adapter.
        model_name: Full provider model name.
        elapsed_seconds: Total wall-clock seconds the run took.

    Returns:
        A 2-tuple of:
        - List of :class:`SampleResult` (one per sample).
        - :class:`ModelScore` with aggregate and per-category breakdowns.
    """
    samples_by_id: dict[str, BenchmarkSample] = {s.id: s for s in samples}
    sample_results: list[SampleResult] = []

    for sample in samples:
        response_text = responses.get(sample.id, "")
        result = score_sample(sample, response_text, model_id=model_id)
        sample_results.append(result)

    model_score = compute_aggregate_score(
        sample_results=sample_results,
        samples_by_id=samples_by_id,
        model_id=model_id,
        model_name=model_name,
        elapsed_seconds=elapsed_seconds,
    )

    return sample_results, model_score


# ---------------------------------------------------------------------------
# Utility: extract identified issues from free-form model text
# ---------------------------------------------------------------------------


def extract_issues_from_response(
    response: str,
    vocabulary: Sequence[str],
) -> list[str]:
    """Return which vocabulary phrases appear in a model's review response.

    This is a utility function that can be used standalone to inspect which
    known issue phrases the model surfaced without needing a full sample.

    Args:
        response: Raw model review text.
        vocabulary: Sequence of issue/label phrases to search for.

    Returns:
        List of vocabulary phrases found in *response*, preserving the order
        in which they appear in *vocabulary*.
    """
    normalised = _normalise(response)
    return [
        phrase
        for phrase in vocabulary
        if _phrase_in_text(_normalise(phrase), normalised)
    ]
