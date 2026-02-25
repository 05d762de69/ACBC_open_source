"""
Multi-respondent aggregation for ACBC survey data.

Loads all raw result files from ``data/raw/``, re-runs analysis for
each participant, and computes group-level summary statistics.

For counting, monotone, and bayesian_logit: analyses are run independently
per participant and then averaged.

For **hb** (true Hierarchical Bayes): all participants are modelled
jointly via Gibbs sampling, producing both group-level and
shrinkage-adjusted individual-level estimates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from acbc.analysis import (
    AnalysisResult,
    analyze_bayesian_logit,
    analyze_counts,
    analyze_hb,
    analyze_monotone,
)
from acbc.io import load_all_raw_results, reconstruct_results_for_analysis

console = Console()

PER_PARTICIPANT_METHODS = {
    "counts": ("Counting-based", analyze_counts),
    "monotone": ("Monotone regression", analyze_monotone),
    "bayesian_logit": ("Bayesian logit", analyze_bayesian_logit),
}


def _aggregate_analysis_results(
    per_participant: dict[str, AnalysisResult],
) -> dict[str, Any]:
    """
    Compute group-level means and standard deviations from individual results.

    Returns a dict with:
        - ``level_utilities``: {attr::level: (mean, std, n)}
        - ``attribute_importances``: {attr: (mean, std, n)}
        - ``predicted_winner``: {attr: most-frequently-predicted level}
    """
    from collections import Counter, defaultdict

    level_utils: dict[str, list[float]] = defaultdict(list)
    attr_importances: dict[str, list[float]] = defaultdict(list)
    winner_votes: dict[str, Counter[str]] = defaultdict(Counter)

    for pid, result in per_participant.items():
        for lu in result.level_utilities:
            key = f"{lu.attribute}::{lu.level}"
            level_utils[key].append(lu.utility)
        for ai in result.attribute_importances:
            attr_importances[ai.attribute].append(ai.importance)
        if result.predicted_winner:
            for attr, lv in result.predicted_winner.items():
                winner_votes[attr][lv] += 1

    return {
        "level_utilities": {
            k: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, len(v))
            for k, v in level_utils.items()
        },
        "attribute_importances": {
            k: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, len(v))
            for k, v in attr_importances.items()
        },
        "predicted_winner": {
            attr: counter.most_common(1)[0][0]
            for attr, counter in winner_votes.items()
        },
    }


def _display_aggregate(
    agg: dict[str, Any],
    method_name: str,
    n_participants: int,
) -> None:
    """Render group-level results to the terminal."""
    console.print()
    console.print(
        Panel(
            f"[bold]Group-Level Results[/bold] — {method_name}\n"
            f"N = {n_participants} participants",
            border_style="bright_blue",
        )
    )

    # Attribute importances
    console.print()
    console.print("[bold underline]Attribute Importance (group mean ± SD)[/bold underline]")
    console.print()

    imp = agg["attribute_importances"]
    max_bar = 40
    max_imp = max((v[0] for v in imp.values()), default=1.0)
    max_name = max((len(k) for k in imp), default=10)

    for attr in sorted(imp, key=lambda a: -imp[a][0]):
        mean, std, n = imp[attr]
        bar_len = int((mean / max_imp) * max_bar) if max_imp > 0 else 0
        bar = "█" * bar_len
        name_padded = attr.ljust(max_name)
        console.print(f"  {name_padded}  [cyan]{bar}[/cyan] {mean:5.1f}% ± {std:4.1f}")

    # Level utilities
    console.print()
    console.print("[bold underline]Level Utilities (group mean ± SD)[/bold underline]")
    console.print()

    attrs_seen: list[str] = []
    entries_by_attr: dict[str, list[tuple[str, float, float]]] = {}
    for key, (mean, std, n) in agg["level_utilities"].items():
        attr, level = key.split("::", 1)
        if attr not in entries_by_attr:
            attrs_seen.append(attr)
            entries_by_attr[attr] = []
        entries_by_attr[attr].append((level, mean, std))

    for attr in attrs_seen:
        table = Table(
            title=attr, box=box.SIMPLE, show_header=True,
            header_style="bold", padding=(0, 1),
        )
        table.add_column("Level", min_width=16)
        table.add_column("Mean Utility", justify="right", min_width=12)
        table.add_column("SD", justify="right", min_width=8)

        entries = sorted(entries_by_attr[attr], key=lambda x: -x[1])
        for lv, mean, std in entries:
            style = "green" if mean > 0 else ("red" if mean < 0 else "")
            table.add_row(lv, f"{mean:+.4f}", f"{std:.4f}", style=style)
        console.print(table)

    # Predicted winner
    winner = agg.get("predicted_winner", {})
    if winner:
        console.print()
        console.print("[bold underline]Group Predicted Ideal (mode across participants)[/bold underline]")
        console.print()
        table = Table(box=box.ROUNDED, show_header=True, header_style="bold green")
        table.add_column("Attribute", min_width=16)
        table.add_column("Most-Chosen Level", min_width=16)
        for attr in attrs_seen:
            table.add_row(attr, winner.get(attr, "—"))
        console.print(table)


def _run_per_participant_method(
    raw_files: list[dict[str, Any]],
    method_key: str,
    *,
    seed: int | None = None,
) -> None:
    """Run a per-participant analysis method and display aggregated results."""
    method_name, analyze_fn = PER_PARTICIPANT_METHODS[method_key]
    console.print(f"\n[bold]Running {method_name} for each participant...[/bold]")

    per_participant: dict[str, AnalysisResult] = {}
    for raw in raw_files:
        pid = raw.get("participant_id", "unknown")
        try:
            engine_results = reconstruct_results_for_analysis(raw)
            if method_key == "bayesian_logit":
                ar = analyze_fn(engine_results, seed=seed)
            else:
                ar = analyze_fn(engine_results)
            per_participant[pid] = ar
            console.print(f"  [dim]{pid}[/dim]")
        except Exception as exc:
            console.print(f"  [red]{pid}: {exc}[/red]")

    if not per_participant:
        console.print("[yellow]No participants could be analysed.[/yellow]")
        return

    agg = _aggregate_analysis_results(per_participant)
    _display_aggregate(agg, method_name, len(per_participant))


def _run_hb(
    raw_files: list[dict[str, Any]],
    *,
    seed: int | None = None,
) -> None:
    """Run true Hierarchical Bayes jointly on all participants."""
    n = len(raw_files)
    if n < 2:
        console.print(
            f"[yellow]Hierarchical Bayes requires >= 2 participants "
            f"(found {n}).  Falling back to Bayesian logit.[/yellow]"
        )
        _run_per_participant_method(raw_files, "bayesian_logit", seed=seed)
        return

    console.print(
        f"\n[bold]Running Hierarchical Bayes (Gibbs sampler) "
        f"on {n} participants jointly...[/bold]"
    )

    participant_results: dict[str, dict[str, Any]] = {}
    for raw in raw_files:
        pid = raw.get("participant_id", "unknown")
        try:
            participant_results[pid] = reconstruct_results_for_analysis(raw)
        except Exception as exc:
            console.print(f"  [red]{pid}: {exc}[/red]")

    if len(participant_results) < 2:
        console.print("[yellow]Not enough valid participants for HB.[/yellow]")
        return

    with console.status("[bold cyan]Running Gibbs sampler (this may take a moment)...[/bold cyan]"):
        group_result, individual_results = analyze_hb(
            participant_results, seed=seed,
        )

    # Display group-level via the standard aggregate display
    agg = _aggregate_analysis_results(individual_results)
    _display_aggregate(agg, "Hierarchical Bayes (joint)", len(individual_results))

    # Also show the group-level alpha estimates
    console.print()
    console.print(
        Panel(
            "[bold]HB Group-Level Alpha (upper-level mean)[/bold]\n"
            "These are the posterior mean of the group distribution,\n"
            "not an average of individuals — they represent the\n"
            "population-level preference structure.",
            border_style="magenta",
        )
    )

    console.print()
    console.print("[bold underline]Group Alpha Importances[/bold underline]")
    console.print()
    max_bar = 40
    max_imp = max(
        (ai.importance for ai in group_result.attribute_importances), default=1.0,
    )
    max_name = max(
        (len(ai.attribute) for ai in group_result.attribute_importances), default=10,
    )
    for ai in sorted(group_result.attribute_importances, key=lambda x: -x.importance):
        bar_len = int((ai.importance / max_imp) * max_bar) if max_imp > 0 else 0
        bar = "█" * bar_len
        console.print(f"  {ai.attribute.ljust(max_name)}  [magenta]{bar}[/magenta] {ai.importance:5.1f}%")


def run_aggregate(
    data_dir: Path,
    *,
    method: str = "all",
    seed: int | None = None,
) -> None:
    """
    Load all raw results, run analysis, and display group-level aggregates.
    """
    raw_files = load_all_raw_results(data_dir)

    if not raw_files:
        console.print(
            f"[red]No raw data files found in {data_dir / 'raw'}.[/red]\n"
            f"Run the survey first with:  python main.py"
        )
        return

    console.print(
        Panel(
            f"[bold]ACBC Multi-Respondent Aggregation[/bold]\n\n"
            f"Found [cyan]{len(raw_files)}[/cyan] participant(s) in {data_dir / 'raw'}",
            border_style="bright_blue",
        )
    )

    if method == "all":
        for mk in PER_PARTICIPANT_METHODS:
            _run_per_participant_method(raw_files, mk, seed=seed)
        _run_hb(raw_files, seed=seed)
    elif method == "hb":
        _run_hb(raw_files, seed=seed)
    else:
        _run_per_participant_method(raw_files, method, seed=seed)

    console.print("\n[bold green]Aggregation complete.[/bold green]\n")
