"""
Scenario generation for the ACBC survey.

Responsibilities:
- Generate screening scenarios that are "near-neighbours" of the BYO ideal
  (vary 1-3 attributes), while ensuring each level appears across the set.
- Generate choice-tournament concepts from the filtered consideration set.
"""

# Import modules
from __future__ import annotations
import random

from acbc.models import Attribute, Scenario, SurveyConfig, NonCompensatoryRule

# ------------------------------------------------------------------
# Scenario generation
# ------------------------------------------------------------------

def _random_swap(
    ideal: Scenario,
    attributes: list[Attribute],
    n_swaps: int,
    rng: random.Random,
) -> Scenario:
    """Create a new scenario by randomly changing *n_swaps* attributes from the ideal."""
    attrs_to_swap = rng.sample(attributes, min(n_swaps, len(attributes)))
    new_levels = dict(ideal.levels)
    for attr in attrs_to_swap:
        alternatives = [lv for lv in attr.levels if lv != ideal.levels[attr.name]]
        if alternatives:
            new_levels[attr.name] = rng.choice(alternatives)
    return Scenario(levels=new_levels)


def generate_screening_scenarios(
    config: SurveyConfig,
    ideal: Scenario,
    *,
    seed: int | None = None,
) -> list[list[Scenario]]:
    """
    Generate screening pages of scenarios near the BYO ideal.

    Strategy:
    - For each page, create *scenarios_per_page* scenarios.
    - Each scenario differs from the ideal in 1-3 attributes (chosen randomly).
    - We track level coverage and bias generation toward under-represented levels
      to approximate level balance.
    - Duplicates of the ideal or of previously generated scenarios are avoided.

    Returns a list of pages, each page being a list of Scenario objects.
    """
    rng = random.Random(seed)
    settings = config.settings
    total_scenarios = settings.screening_pages * settings.scenarios_per_page

    # Track how often each level has appeared (for balance)
    level_counts: dict[str, dict[str, int]] = {
        attr.name: {lv: 0 for lv in attr.levels} for attr in config.attributes
    }

    generated: list[Scenario] = []
    seen: set[int] = {hash(ideal)}

    attempts = 0
    max_attempts = total_scenarios * 20  # safety valve

    while len(generated) < total_scenarios and attempts < max_attempts:
        attempts += 1
        # Vary 1, 2, or 3 attributes (weighted toward fewer swaps to stay near ideal)
        n_swaps = rng.choices([1, 2, 3], weights=[0.45, 0.35, 0.20], k=1)[0]
        candidate = _random_swap(ideal, config.attributes, n_swaps, rng)

        h = hash(candidate)
        if h in seen:
            continue

        # Bias toward candidates that cover under-represented levels
        # Calculate a "coverage score" – lower is better (more needed)
        coverage_score = sum(
            level_counts[attr][candidate.levels[attr]]
            for attr in candidate.levels
        )
        # Accept with probability inversely proportional to coverage
        accept_prob = 1.0 / (1.0 + coverage_score * 0.3)
        if rng.random() > accept_prob and attempts < max_attempts - total_scenarios:
            continue

        seen.add(h)
        generated.append(candidate)
        for attr, lv in candidate.levels.items():
            level_counts[attr][lv] += 1

    # Chunk into pages
    pages: list[list[Scenario]] = []
    for i in range(0, len(generated), settings.scenarios_per_page):
        pages.append(generated[i : i + settings.scenarios_per_page])

    return pages


def generate_tournament_pool(
    config: SurveyConfig,
    ideal: Scenario,
    screening_accepted: list[Scenario],
    rules: list[NonCompensatoryRule],
    *,
    seed: int | None = None,
) -> list[Scenario]:
    """
    Build the pool of scenarios for the choice tournament.

    - Start from screening-accepted scenarios.
    - Filter out any that violate confirmed unacceptable / must-have rules.
    - If pool is too small, generate additional valid scenarios.
    - Always include the BYO ideal (if it passes rules).
    """
    rng = random.Random(seed)

    unacceptable: dict[str, set[str]] = {}
    must_have: dict[str, str] = {}
    for rule in rules:
        if rule.rule_type == "unacceptable":
            unacceptable.setdefault(rule.attribute_name, set()).add(rule.level)
        elif rule.rule_type == "must_have":
            must_have[rule.attribute_name] = rule.level

    def is_valid(scenario: Scenario) -> bool:
        for attr, lv in scenario.levels.items():
            if attr in unacceptable and lv in unacceptable[attr]:
                return False
            if attr in must_have and lv != must_have[attr]:
                return False
        return True

    pool: list[Scenario] = []
    seen: set[int] = set()

    # Add ideal if valid
    if is_valid(ideal):
        pool.append(ideal)
        seen.add(hash(ideal))

    # Add accepted screening scenarios that pass rules
    for sc in screening_accepted:
        h = hash(sc)
        if h not in seen and is_valid(sc):
            pool.append(sc)
            seen.add(h)

    # If pool is small, generate more valid scenarios
    min_pool = max(config.settings.choice_tournament_size * 3, 6)
    attempts = 0
    while len(pool) < min_pool and attempts < 500:
        attempts += 1
        n_swaps = rng.choices([1, 2, 3], weights=[0.4, 0.4, 0.2], k=1)[0]
        candidate = _random_swap(ideal, config.attributes, n_swaps, rng)
        h = hash(candidate)
        if h not in seen and is_valid(candidate):
            pool.append(candidate)
            seen.add(h)

    rng.shuffle(pool)
    return pool


def chunk_tournament_pool(
    pool: list[Scenario],
    group_size: int,
) -> list[list[Scenario]]:
    """Split the tournament pool into choice sets of *group_size*."""
    chunks: list[list[Scenario]] = []
    for i in range(0, len(pool), group_size):
        chunk = pool[i : i + group_size]
        if len(chunk) >= 2:  # need at least 2 to make a choice
            chunks.append(chunk)
    return chunks
