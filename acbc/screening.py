"""
Non-compensatory rule detection for the ACBC screening stage.

After the respondent evaluates screening scenarios, this module analyses
their accept/reject patterns to identify:
- **Unacceptable** levels: levels the respondent systematically avoided.
- **Must-have** levels: levels the respondent consistently required.
"""

# import modules
from __future__ import annotations

from acbc.models import (
    Attribute,
    Scenario,
    ScreeningResponse,
    SurveyConfig,
    SurveySettings,
)

# ------------------------------------------------------------------
# Non-compensatory rule detection
# ------------------------------------------------------------------

def _compute_level_stats(
    config: SurveyConfig,
    screening_pages: list[list[Scenario]],
    screening_responses: list[ScreeningResponse],
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    """
    Count how often each level was shown and how often it was accepted.

    Returns:
        (shown_counts, accepted_counts) – both nested dicts of
        attribute_name -> level -> count
    """
    shown: dict[str, dict[str, int]] = {
        attr.name: {lv: 0 for lv in attr.levels} for attr in config.attributes
    }
    accepted: dict[str, dict[str, int]] = {
        attr.name: {lv: 0 for lv in attr.levels} for attr in config.attributes
    }

    for page_scenarios, response in zip(screening_pages, screening_responses):
        for idx, scenario in enumerate(page_scenarios):
            was_accepted = response.responses.get(idx, False)
            for attr_name, level in scenario.levels.items():
                shown[attr_name][level] += 1
                if was_accepted:
                    accepted[attr_name][level] += 1

    return shown, accepted


def detect_unacceptable_candidates(
    config: SurveyConfig,
    screening_pages: list[list[Scenario]],
    screening_responses: list[ScreeningResponse],
) -> list[tuple[str, str]]:
    """
    Identify (attribute_name, level) pairs where the respondent consistently
    rejected scenarios containing that level.

    A level is flagged if its rejection rate exceeds the configured
    `unacceptable_threshold` AND it was shown at least twice (to avoid
    false positives from small samples).

    Returns a list of (attribute_name, level) tuples, ordered by rejection
    rate (most rejected first).
    """
    settings = config.settings
    shown, accepted = _compute_level_stats(config, screening_pages, screening_responses)

    candidates: list[tuple[str, str, float]] = []

    for attr in config.attributes:
        for lv in attr.levels:
            n_shown = shown[attr.name][lv]
            if n_shown < 2:
                continue
            n_accepted = accepted[attr.name][lv]
            rejection_rate = 1.0 - (n_accepted / n_shown)
            if rejection_rate >= settings.unacceptable_threshold:
                candidates.append((attr.name, lv, rejection_rate))

    # Sort by rejection rate descending
    candidates.sort(key=lambda x: x[2], reverse=True)

    # Limit to max questions configured
    max_q = settings.max_unacceptable_questions
    return [(attr, lv) for attr, lv, _ in candidates[:max_q]]


def detect_must_have_candidates(
    config: SurveyConfig,
    screening_pages: list[list[Scenario]],
    screening_responses: list[ScreeningResponse],
) -> list[tuple[str, str]]:
    """
    Identify (attribute_name, level) pairs where the respondent expressed
    interest in *only* that level of an attribute.

    A level is flagged if its acceptance rate exceeds the configured
    `must_have_threshold`, while all other levels of the same attribute
    have a much lower acceptance rate.

    Returns a list of (attribute_name, level) tuples.
    """
    settings = config.settings
    shown, accepted = _compute_level_stats(config, screening_pages, screening_responses)

    candidates: list[tuple[str, str, float]] = []

    for attr in config.attributes:
        if len(attr.levels) < 2:
            continue

        # Compute acceptance rates for all levels of this attribute
        rates: dict[str, float] = {}
        for lv in attr.levels:
            n_shown = shown[attr.name][lv]
            if n_shown == 0:
                rates[lv] = 0.0
            else:
                rates[lv] = accepted[attr.name][lv] / n_shown

        # Find the level with the highest acceptance rate
        best_level = max(rates, key=lambda k: rates[k])
        best_rate = rates[best_level]

        if best_rate < settings.must_have_threshold:
            continue

        # Check that all other levels have substantially lower acceptance
        other_rates = [r for lv, r in rates.items() if lv != best_level]
        if not other_rates:
            continue

        max_other = max(other_rates)
        # The gap should be meaningful (at least 40 percentage points)
        if best_rate - max_other >= 0.40:
            candidates.append((attr.name, best_level, best_rate))

    candidates.sort(key=lambda x: x[2], reverse=True)
    max_q = settings.max_must_have_questions
    return [(attr, lv) for attr, lv, _ in candidates[:max_q]]


def get_accepted_scenarios(
    screening_pages: list[list[Scenario]],
    screening_responses: list[ScreeningResponse],
) -> list[Scenario]:
    """Return all scenarios that the respondent marked as 'a possibility'."""
    accepted: list[Scenario] = []
    for page_scenarios, response in zip(screening_pages, screening_responses):
        for idx, scenario in enumerate(page_scenarios):
            if response.responses.get(idx, False):
                accepted.append(scenario)
    return accepted
