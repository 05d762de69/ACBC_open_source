"""
Utility estimation and analysis for ACBC survey results.

Three tiers of analysis, all producing the same output format:
1. **Counting-based** -> frequency of acceptance/choice per level.
2. **Monotone regression** -> individual-level ordinal utilities via
   isotonic regression (as in Al-Omari et al., 2017).
3. **Hierarchical Bayes** -> MNL lower level + multivariate normal upper
   level, using MCMC Metropolis-Hastings for individual part-worths
   (as in Sanchez, 2019 / Sawtooth Software).

All methods return an ``AnalysisResult`` that the frontend can render.
"""

# Import modules
from __future__ import annotations
import json
import csv
import io
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from numpy.typing import NDArray

# scipy is imported lazily inside analyze_hb() to avoid slow macOS
# Gatekeeper scans of C extensions at module-import time.
# Do NOT add a top-level `from scipy ...` import here.

# =====================================================================
# Result container
# =====================================================================

@dataclass
class LevelUtility:
    """Utility value for a single level of an attribute."""

    attribute: str
    level: str
    utility: float
    # For HB: posterior standard deviation
    std: float | None = None


@dataclass
class AttributeImportance:
    """Relative importance of an attribute (sums to 100% across all attributes)."""

    attribute: str
    importance: float  # 0-100


@dataclass
class AnalysisResult:
    """Full analysis output."""

    method: str  # "counts", "monotone", "hb"
    level_utilities: list[LevelUtility]
    attribute_importances: list[AttributeImportance]
    predicted_winner: dict[str, str] | None = None  # attribute -> best level

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "level_utilities": [
                {"attribute": lu.attribute, "level": lu.level, "utility": lu.utility}
                for lu in self.level_utilities
            ],
            "attribute_importances": [
                {"attribute": ai.attribute, "importance": ai.importance}
                for ai in self.attribute_importances
            ],
            "predicted_winner": self.predicted_winner,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_csv(self) -> str:
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["type", "attribute", "level", "value"])
        for lu in self.level_utilities:
            writer.writerow(["utility", lu.attribute, lu.level, f"{lu.utility:.4f}"])
        for ai in self.attribute_importances:
            writer.writerow(["importance", ai.attribute, "", f"{ai.importance:.2f}"])
        return buf.getvalue()


# =====================================================================
# Helper: build design matrix from survey results
# =====================================================================


def _build_attribute_level_index(
    results: dict[str, Any],
) -> tuple[list[str], dict[str, list[str]], dict[str, int]]:
    """
    Build an ordered index of attribute-level pairs for matrix construction.

    Returns:
        attr_names: ordered list of attribute names
        attr_levels: attr_name -> list of level names
        level_to_col: "attr_name::level" -> column index in design matrix
    """
    config = results["config"]
    attr_names: list[str] = [a.name for a in config.attributes]
    attr_levels: dict[str, list[str]] = {
        a.name: list(a.levels) for a in config.attributes
    }
    level_to_col: dict[str, int] = {}
    col = 0
    for attr_name in attr_names:
        for lv in attr_levels[attr_name]:
            level_to_col[f"{attr_name}::{lv}"] = col
            col += 1
    return attr_names, attr_levels, level_to_col


def _compute_importances(
    attr_names: list[str],
    attr_levels: dict[str, list[str]],
    utilities: dict[str, float],
) -> list[AttributeImportance]:
    """
    Compute relative importance from part-worth utilities.

    Importance of an attribute = range of its level utilities
    (max - min), then normalized so all importances sum to 100.
    """
    ranges: dict[str, float] = {}
    for attr_name in attr_names:
        utils = [utilities.get(f"{attr_name}::{lv}", 0.0) for lv in attr_levels[attr_name]]
        ranges[attr_name] = max(utils) - min(utils) if utils else 0.0

    total = sum(ranges.values())
    if total == 0:
        total = 1.0  # avoid division by zero

    return [
        AttributeImportance(attribute=attr, importance=100.0 * ranges[attr] / total)
        for attr in attr_names
    ]


def _predict_winner(
    attr_names: list[str],
    attr_levels: dict[str, list[str]],
    utilities: dict[str, float],
) -> dict[str, str]:
    """Pick the level with highest utility for each attribute."""
    winner: dict[str, str] = {}
    for attr_name in attr_names:
        best_lv = max(
            attr_levels[attr_name],
            key=lambda lv: utilities.get(f"{attr_name}::{lv}", 0.0),
        )
        winner[attr_name] = best_lv
    return winner


# =====================================================================
# Tier 1: Counting-based estimation
# =====================================================================

def analyze_counts(results: dict[str, Any]) -> AnalysisResult:
    """
    Simple counting-based utility estimation.

    Utility for each level is based on:
    - Screening acceptance rate (how often it was accepted vs shown)
    - Choice frequency (how often it was in the chosen concept)

    These are combined into a normalized score, zero-centered per attribute.
    """
    attr_names, attr_levels, _ = _build_attribute_level_index(results)
    shown = results.get("level_shown_count", {})
    accepted = results.get("level_accepted_count", {})
    chosen = results.get("level_chosen_count", {})

    utilities: dict[str, float] = {}

    for attr_name in attr_names:
        raw_scores: dict[str, float] = {}
        for lv in attr_levels[attr_name]:
            n_shown = shown.get(attr_name, {}).get(lv, 0)
            n_accepted = accepted.get(attr_name, {}).get(lv, 0)
            n_chosen = chosen.get(attr_name, {}).get(lv, 0)

            # Acceptance rate component (0-1)
            acc_rate = n_accepted / n_shown if n_shown > 0 else 0.5

            # Choice component (bonus weight for tournament choices)
            choice_bonus = n_chosen * 0.5

            raw_scores[lv] = acc_rate + choice_bonus

        # Zero-center within attribute
        mean_score = np.mean(list(raw_scores.values())) if raw_scores else 0.0
        for lv in attr_levels[attr_name]:
            utilities[f"{attr_name}::{lv}"] = raw_scores.get(lv, 0.0) - float(mean_score)

    level_utils = [
        LevelUtility(
            attribute=attr_name,
            level=lv,
            utility=utilities[f"{attr_name}::{lv}"],
        )
        for attr_name in attr_names
        for lv in attr_levels[attr_name]
    ]
    importances = _compute_importances(attr_names, attr_levels, utilities)
    predicted = _predict_winner(attr_names, attr_levels, utilities)

    return AnalysisResult(
        method="counts",
        level_utilities=level_utils,
        attribute_importances=importances,
        predicted_winner=predicted,
    )


# =====================================================================
# Tier 2: Monotone regression (individual-level ordinal utilities)
# =====================================================================


def _isotonic_regression_1d(y: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Pool-adjacent-violators algorithm for isotonic (monotone) regression.

    Given values y[0..n-1], find the monotone non-decreasing sequence
    that minimises sum of squared deviations.
    """
    n = len(y)
    if n <= 1:
        return y.copy()

    # Pool Adjacent Violators
    result = y.astype(float).copy()
    block_start = list(range(n))
    block_size = [1] * n

    i = 0
    while i < n - 1:
        # Find the start of this block
        while block_start[i] != i:
            i = block_start[i]
        j = i + block_size[i]
        if j >= n:
            break
        if result[i] <= result[j]:
            i = j
            continue
        # Pool blocks i and j
        total = result[i] * block_size[i] + result[j] * block_size[j]
        new_size = block_size[i] + block_size[j]
        result[i] = total / new_size
        block_size[i] = new_size
        # Update block_start for merged block
        for k in range(i + 1, i + new_size):
            if k < n:
                block_start[k] = i

        # Back up to check for new violations
        # Find previous block start
        if i > 0:
            prev = i - 1
            while prev > 0 and block_start[prev] != prev:
                prev = block_start[prev]
            i = prev
        # else stay at i=0

    # Expand block values
    expanded = np.empty(n, dtype=float)
    i = 0
    while i < n:
        bs = block_size[i]
        expanded[i : i + bs] = result[i]
        i += bs

    return expanded


def analyze_monotone(results: dict[str, Any]) -> AnalysisResult:
    """
    Monotone regression for individual-level utility estimation.

    For each attribute, the levels are ordered by their raw acceptance/choice
    score and then isotonic regression is applied to produce ordinal utilities.
    Utilities are zero-centered per attribute.

    Reference: Al-Omari et al. (2017), Sawtooth Software monotone regression.
    """
    attr_names, attr_levels, _ = _build_attribute_level_index(results)
    shown = results.get("level_shown_count", {})
    accepted = results.get("level_accepted_count", {})
    chosen = results.get("level_chosen_count", {})

    utilities: dict[str, float] = {}

    for attr_name in attr_names:
        # Compute raw scores (same as counts method)
        levels = attr_levels[attr_name]
        raw: list[tuple[str, float]] = []
        for lv in levels:
            n_shown = shown.get(attr_name, {}).get(lv, 0)
            n_accepted = accepted.get(attr_name, {}).get(lv, 0)
            n_chosen = chosen.get(attr_name, {}).get(lv, 0)
            acc_rate = n_accepted / n_shown if n_shown > 0 else 0.5
            score = acc_rate + n_chosen * 0.5
            raw.append((lv, score))

        # Sort by raw score
        raw.sort(key=lambda x: x[1])
        sorted_levels = [r[0] for r in raw]
        sorted_scores = np.array([r[1] for r in raw])

        # Apply isotonic regression (enforce monotone non-decreasing)
        isotonic_scores = _isotonic_regression_1d(sorted_scores)

        # Zero-center
        mean_val = float(np.mean(isotonic_scores))
        for i, lv in enumerate(sorted_levels):
            utilities[f"{attr_name}::{lv}"] = float(isotonic_scores[i]) - mean_val

    level_utils = [
        LevelUtility(attribute=attr_name, level=lv, utility=utilities[f"{attr_name}::{lv}"])
        for attr_name in attr_names
        for lv in attr_levels[attr_name]
    ]
    importances = _compute_importances(attr_names, attr_levels, utilities)
    predicted = _predict_winner(attr_names, attr_levels, utilities)

    return AnalysisResult(
        method="monotone",
        level_utilities=level_utils,
        attribute_importances=importances,
        predicted_winner=predicted,
    )


# =====================================================================
# Tier 3: Hierarchical Bayes (MCMC / Metropolis-Hastings)
# =====================================================================


def _encode_scenario(
    scenario_levels: dict[str, str],
    attr_names: list[str],
    attr_levels: dict[str, list[str]],
    level_to_col: dict[str, int],
) -> NDArray[np.float64]:
    """Encode a scenario as a dummy-coded row vector."""
    n_cols = sum(len(attr_levels[a]) for a in attr_names)
    row = np.zeros(n_cols)
    for attr_name in attr_names:
        lv = scenario_levels.get(attr_name)
        if lv is not None:
            key = f"{attr_name}::{lv}"
            if key in level_to_col:
                row[level_to_col[key]] = 1.0
    return row


def _mnl_log_likelihood(
    beta: NDArray[np.float64],
    X_choices: list[NDArray[np.float64]],
    y_choices: list[int],
) -> float:
    """
    Negative log-likelihood for MNL (for minimization).

    X_choices[t] is the design matrix for choice task t (rows = alternatives).
    y_choices[t] is the index of the chosen alternative.
    """
    ll = 0.0
    for X_task, chosen_idx in zip(X_choices, y_choices):
        utilities = X_task @ beta
        # For numerical stability, subtract max
        utilities -= np.max(utilities)
        exp_u = np.exp(utilities)
        ll += utilities[chosen_idx] - np.log(np.sum(exp_u))
    return -ll  # negative because we minimize


def analyze_hb(
    results: dict[str, Any],
    *,
    n_iterations: int = 2000,
    burn_in: int = 500,
    seed: int | None = None,
) -> AnalysisResult:
    """
    Hierarchical Bayes estimation with MCMC (Metropolis-Hastings).

    This is a simplified but functional HB implementation suitable for
    individual-level estimation from a single respondent's ACBC data.

    For a single respondent, this reduces to a Bayesian logit with
    a normal prior on beta, estimated via random-walk Metropolis-Hastings.

    Reference: Orme (2009), Sawtooth Software HB technical paper;
               Sanchez (2019) thesis.
    """
    from scipy.optimize import minimize  # lazy import — see module docstring

    rng = np.random.default_rng(seed)

    attr_names, attr_levels, level_to_col = _build_attribute_level_index(results)
    n_params = sum(len(attr_levels[a]) for a in attr_names)

    # Build choice data from the tournament rounds
    tournament_rounds = results.get("tournament_rounds", [])
    choice_responses = results.get("choice_responses", [])

    # Also use screening data: accepted vs. rejected as pseudo-choices
    screening_scenarios = results.get("screening_scenarios", [])
    screening_responses = results.get("screening_responses", [])

    X_choices: list[NDArray[np.float64]] = []
    y_choices: list[int] = []

    # Tournament choices (true choices)
    for round_scenarios, response in zip(tournament_rounds, choice_responses):
        if not round_scenarios:
            continue
        X_task = np.array([
            _encode_scenario(sc.levels, attr_names, attr_levels, level_to_col)
            for sc in round_scenarios
        ])
        X_choices.append(X_task)
        y_choices.append(response.chosen_index)

    # Screening data as pseudo-choices: for each page, treat accepted
    # scenarios as "chosen" in pairwise comparisons with rejected ones
    for page_scenarios, response in zip(screening_scenarios, screening_responses):
        accepted_indices = [i for i, acc in response.responses.items() if acc]
        rejected_indices = [i for i, acc in response.responses.items() if not acc]
        for acc_idx in accepted_indices:
            for rej_idx in rejected_indices:
                X_pair = np.array([
                    _encode_scenario(page_scenarios[acc_idx].levels, attr_names, attr_levels, level_to_col),
                    _encode_scenario(page_scenarios[rej_idx].levels, attr_names, attr_levels, level_to_col),
                ])
                X_choices.append(X_pair)
                y_choices.append(0)  # accepted is always index 0

    if not X_choices:
        # Fall back to counts if no choice data
        return analyze_counts(results)

    # Prior: N(0, I)
    prior_mean = np.zeros(n_params)
    prior_cov_inv = np.eye(n_params)  # precision matrix

    # Initialize beta from MLE (or zeros if MLE fails)
    try:
        result_opt = minimize(
            _mnl_log_likelihood,
            x0=np.zeros(n_params),
            args=(X_choices, y_choices),
            method="L-BFGS-B",
        )
        beta_current = result_opt.x
    except Exception:
        beta_current = np.zeros(n_params)

    # Proposal scale (adapt during burn-in)
    proposal_scale = 0.1 * np.ones(n_params)

    # MCMC chain storage
    chain: list[NDArray[np.float64]] = []
    n_accepted = 0

    for iteration in range(n_iterations):
        # Propose new beta
        beta_proposal = beta_current + rng.normal(0, proposal_scale)

        # Log-posterior current
        ll_current = -_mnl_log_likelihood(beta_current, X_choices, y_choices)
        lp_current = -0.5 * float(
            (beta_current - prior_mean) @ prior_cov_inv @ (beta_current - prior_mean)
        )

        # Log-posterior proposal
        ll_proposal = -_mnl_log_likelihood(beta_proposal, X_choices, y_choices)
        lp_proposal = -0.5 * float(
            (beta_proposal - prior_mean) @ prior_cov_inv @ (beta_proposal - prior_mean)
        )

        # Acceptance ratio
        log_ratio = (ll_proposal + lp_proposal) - (ll_current + lp_current)
        if np.log(rng.random()) < log_ratio:
            beta_current = beta_proposal
            n_accepted += 1

        if iteration >= burn_in:
            chain.append(beta_current.copy())

        # Adapt proposal scale during burn-in
        if iteration < burn_in and iteration > 0 and iteration % 100 == 0:
            accept_rate = n_accepted / (iteration + 1)
            if accept_rate < 0.2:
                proposal_scale *= 0.8
            elif accept_rate > 0.5:
                proposal_scale *= 1.2

    # Posterior mean and std
    chain_array = np.array(chain)
    posterior_mean = np.mean(chain_array, axis=0)
    posterior_std = np.std(chain_array, axis=0)

    # Zero-center within each attribute
    utilities: dict[str, float] = {}
    stds: dict[str, float] = {}
    col = 0
    for attr_name in attr_names:
        n_levels = len(attr_levels[attr_name])
        attr_utils = posterior_mean[col : col + n_levels]
        attr_stds = posterior_std[col : col + n_levels]
        mean_u = float(np.mean(attr_utils))
        for i, lv in enumerate(attr_levels[attr_name]):
            key = f"{attr_name}::{lv}"
            utilities[key] = float(attr_utils[i]) - mean_u
            stds[key] = float(attr_stds[i])
        col += n_levels

    level_utils = [
        LevelUtility(
            attribute=attr_name,
            level=lv,
            utility=utilities[f"{attr_name}::{lv}"],
            std=stds[f"{attr_name}::{lv}"],
        )
        for attr_name in attr_names
        for lv in attr_levels[attr_name]
    ]
    importances = _compute_importances(attr_names, attr_levels, utilities)
    predicted = _predict_winner(attr_names, attr_levels, utilities)

    return AnalysisResult(
        method="hb",
        level_utilities=level_utils,
        attribute_importances=importances,
        predicted_winner=predicted,
    )
