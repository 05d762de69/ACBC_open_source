"""
CLI frontend for the ACBC survey engine.

Uses *questionary* for keyboard-driven prompts and *rich* for formatted
output (tables, panels, progress indicators).

This module is the **only** place that depends on terminal I/O — the engine
itself is purely functional and can be driven by any frontend.
"""

# Import modules
from __future__ import annotations
import sys
from pathlib import Path
from typing import Any

import questionary
from questionary import Style
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from acbc.engine import ACBCEngine
from acbc.models import (
    BYOQuestion,
    ChoiceQuestion,
    MustHaveQuestion,
    Question,
    Scenario,
    ScreeningQuestion,
    SurveyConfig,
    UnacceptableQuestion,
)
from acbc.analysis import (
    AnalysisResult,
    analyze_counts,
    analyze_hb,
    analyze_monotone,
)

# Rich console for pretty output
console = Console()

# Questionary style
SURVEY_STYLE = Style(
    [
        ("qmark", "fg:cyan bold"),
        ("question", "bold"),
        ("answer", "fg:green bold"),
        ("pointer", "fg:cyan bold"),
        ("highlighted", "fg:cyan bold"),
        ("selected", "fg:green"),
    ]
)

# =====================================================================
# Scenario rendering
# =====================================================================


def _render_scenario_table(
    scenarios: list[Scenario],
    attribute_names: list[str],
    *,
    title: str = "",
    numbered: bool = True,
) -> Table:
    """Build a Rich Table showing scenarios side-by-side."""
    table = Table(
        title=title,
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        padding=(0, 1),
    )

    table.add_column("Attribute", style="bold", min_width=16)
    for i, _sc in enumerate(scenarios):
        label = f"Option {i + 1}" if numbered else f"Scenario"
        table.add_column(label, min_width=14, justify="center")

    for attr_name in attribute_names:
        row = [attr_name]
        for sc in scenarios:
            row.append(sc.levels.get(attr_name, "—"))
        table.add_row(*row)

    return table


# =====================================================================
# Question handlers
# =====================================================================


def _ask_byo(question: BYOQuestion) -> str:
    """Handle a Build-Your-Own question."""
    console.print()
    console.print(
        Panel(
            f"[bold]Build Your Own[/bold]\n\n"
            f"Pick your preferred [cyan]{question.attribute.name}[/cyan].",
            border_style="cyan",
        )
    )

    answer = questionary.select(
        question.prompt,
        choices=question.attribute.levels,
        style=SURVEY_STYLE,
    ).ask()

    if answer is None:
        console.print("[red]Survey cancelled.[/red]")
        sys.exit(0)

    return answer


def _ask_screening(
    question: ScreeningQuestion,
    attribute_names: list[str],
) -> dict[int, bool]:
    """Handle a screening page: show scenarios, ask accept/reject for each."""
    console.print()
    console.print(
        Panel(
            f"[bold]Screening[/bold] — Page {question.page_number}/{question.total_pages}\n\n"
            f"For each option, indicate whether it is a possibility for you.",
            border_style="yellow",
        )
    )

    table = _render_scenario_table(
        question.scenarios,
        attribute_names,
        title=f"Screening Page {question.page_number}",
    )
    console.print(table)
    console.print()

    responses: dict[int, bool] = {}
    for i, _sc in enumerate(question.scenarios):
        answer = questionary.select(
            f"Option {i + 1}:",
            choices=[
                questionary.Choice("A possibility", value=True),
                questionary.Choice("Won't work for me", value=False),
            ],
            style=SURVEY_STYLE,
        ).ask()

        if answer is None:
            console.print("[red]Survey cancelled.[/red]")
            sys.exit(0)

        responses[i] = answer

    return responses


def _ask_unacceptable(question: UnacceptableQuestion) -> bool:
    """Confirm whether a level is truly unacceptable."""
    console.print()
    console.print(
        Panel(
            f"[bold]Unacceptable Check[/bold]\n\n"
            f"We noticed you avoided [red bold]{question.level}[/red bold] "
            f"for [cyan]{question.attribute_name}[/cyan].",
            border_style="red",
        )
    )

    answer = questionary.select(
        question.prompt,
        choices=[
            questionary.Choice("Yes — this is totally unacceptable", value=True),
            questionary.Choice("No — I could still consider it", value=False),
        ],
        style=SURVEY_STYLE,
    ).ask()

    if answer is None:
        console.print("[red]Survey cancelled.[/red]")
        sys.exit(0)

    return answer


def _ask_must_have(question: MustHaveQuestion) -> bool:
    """Confirm whether a level is a must-have."""
    console.print()
    console.print(
        Panel(
            f"[bold]Must-Have Check[/bold]\n\n"
            f"You consistently chose [green bold]{question.level}[/green bold] "
            f"for [cyan]{question.attribute_name}[/cyan].",
            border_style="green",
        )
    )

    answer = questionary.select(
        question.prompt,
        choices=[
            questionary.Choice("Yes — this is a must-have for me", value=True),
            questionary.Choice("No — I'm flexible on this", value=False),
        ],
        style=SURVEY_STYLE,
    ).ask()

    if answer is None:
        console.print("[red]Survey cancelled.[/red]")
        sys.exit(0)

    return answer

def _ask_choice(question: ChoiceQuestion, attribute_names: list[str]) -> int:
    """Handle a choice tournament question."""
    console.print()
    console.print(
        Panel(
            f"[bold]Choice Task[/bold] — Round {question.round_number}\n\n"
            f"Which of the following options do you prefer?",
            border_style="magenta",
        )
    )

    table = _render_scenario_table(
        question.scenarios,
        attribute_names,
        title=f"Choice Round {question.round_number}",
    )
    console.print(table)
    console.print()

    choices = [
        questionary.Choice(f"Option {i + 1}", value=i)
        for i in range(len(question.scenarios))
    ]
    answer = questionary.select(
        "Your choice:",
        choices=choices,
        style=SURVEY_STYLE,
    ).ask()

    if answer is None:
        console.print("[red]Survey cancelled.[/red]")
        sys.exit(0)

    return answer

# =====================================================================
# Results display
# =====================================================================


def _display_results(result: AnalysisResult, config: SurveyConfig) -> None:
    """Display analysis results with tables and ASCII bar charts."""
    console.print()
    console.print(
        Panel(
            f"[bold]Analysis Results[/bold] — Method: [cyan]{result.method}[/cyan]",
            border_style="bright_blue",
        )
    )

    # Attribute importance bar chart
    console.print()
    console.print("[bold underline]Relative Attribute Importance[/bold underline]")
    console.print()

    max_bar_width = 40
    max_importance = max(ai.importance for ai in result.attribute_importances) if result.attribute_importances else 1
    max_name_len = max(len(ai.attribute) for ai in result.attribute_importances) if result.attribute_importances else 10

    for ai in sorted(result.attribute_importances, key=lambda x: -x.importance):
        bar_len = int((ai.importance / max_importance) * max_bar_width) if max_importance > 0 else 0
        bar = "█" * bar_len
        name_padded = ai.attribute.ljust(max_name_len)
        console.print(f"  {name_padded}  [cyan]{bar}[/cyan] {ai.importance:5.1f}%")

    # Level utilities table
    console.print()
    console.print("[bold underline]Level Utilities (zero-centered per attribute)[/bold underline]")
    console.print()

    # Group by attribute
    attr_names = [a.name for a in config.attributes]
    utils_by_attr: dict[str, list[tuple[str, float, float | None]]] = {}
    for lu in result.level_utilities:
        utils_by_attr.setdefault(lu.attribute, []).append((lu.level, lu.utility, lu.std))

    for attr_name in attr_names:
        entries = utils_by_attr.get(attr_name, [])
        if not entries:
            continue

        table = Table(
            title=attr_name,
            box=box.SIMPLE,
            show_header=True,
            header_style="bold",
            padding=(0, 1),
        )
        table.add_column("Level", min_width=16)
        table.add_column("Utility", justify="right", min_width=10)
        if result.method == "hb":
            table.add_column("Std Dev", justify="right", min_width=10)

        # Sort by utility descending
        entries.sort(key=lambda x: -x[1])
        for lv, util, std in entries:
            style = "green" if util > 0 else ("red" if util < 0 else "")
            row = [lv, f"{util:+.4f}"]
            if result.method == "hb" and std is not None:
                row.append(f"{std:.4f}")
            table.add_row(*row, style=style)

        console.print(table)

    # Predicted winner
    if result.predicted_winner:
        console.print()
        console.print("[bold underline]Predicted Ideal Product[/bold underline]")
        console.print()
        table = Table(box=box.ROUNDED, show_header=True, header_style="bold green")
        table.add_column("Attribute", min_width=16)
        table.add_column("Best Level", min_width=16)
        for attr_name in attr_names:
            lv = result.predicted_winner.get(attr_name, "—")
            table.add_row(attr_name, lv)
        console.print(table)


def _display_winner(winner: Scenario | None, config: SurveyConfig) -> None:
    """Show the tournament winner."""
    if winner is None:
        console.print("\n[yellow]No clear winner from the tournament.[/yellow]")
        return

    console.print()
    console.print(
        Panel(
            "[bold green]Tournament Winner[/bold green]\n\n"
            "Based on your choices, this is your most preferred option:",
            border_style="green",
        )
    )

    attr_names = [a.name for a in config.attributes]
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold green")
    table.add_column("Attribute", min_width=16)
    table.add_column("Your Preference", min_width=16)
    for attr_name in attr_names:
        table.add_row(attr_name, winner.levels.get(attr_name, "—"))
    console.print(table)


# =====================================================================
# Main survey runner
# =====================================================================


def run_survey(config_path: str | Path, *, seed: int | None = None) -> None:
    """
    Run a complete ACBC survey in the terminal.

    Parameters
    ----------
    config_path : path to YAML config file
    seed : random seed for reproducibility
    """
    config = SurveyConfig.from_yaml(config_path)
    engine = ACBCEngine(config, seed=seed)
    attr_names = [a.name for a in config.attributes]

    # Welcome
    console.print()
    console.print(
        Panel(
            f"[bold]Welcome to the ACBC Survey[/bold]\n\n"
            f"[cyan]{config.name}[/cyan]\n"
            f"{config.description}\n\n"
            f"This survey has {len(config.attributes)} attributes and will guide you through\n"
            f"three stages: Build Your Own, Screening, and Choice Tasks.\n\n"
            f"Use [bold]arrow keys[/bold] to navigate and [bold]Enter[/bold] to select.",
            border_style="bright_blue",
            padding=(1, 2),
        )
    )

    # Main loop
    while not engine.is_complete:
        question = engine.get_current_question()

        if isinstance(question, BYOQuestion):
            answer = _ask_byo(question)
            engine.submit_answer(answer)

        elif isinstance(question, ScreeningQuestion):
            answer = _ask_screening(question, attr_names)
            engine.submit_answer(answer)

        elif isinstance(question, UnacceptableQuestion):
            answer = _ask_unacceptable(question)
            engine.submit_answer(answer)

        elif isinstance(question, MustHaveQuestion):
            answer = _ask_must_have(question)
            engine.submit_answer(answer)

        elif isinstance(question, ChoiceQuestion):
            if not question.scenarios:
                break  # tournament done signal
            answer = _ask_choice(question, attr_names)
            engine.submit_answer(answer)

    # Results
    results = engine.get_results()

    _display_winner(results["winner"], config)

    # Run all three analysis methods
    console.print()
    console.print(
        Panel(
            "[bold]Running Analysis[/bold]\n\n"
            "Computing utilities with three methods:\n"
            "1. Counting-based\n"
            "2. Monotone regression\n"
            "3. Hierarchical Bayes (MCMC)",
            border_style="bright_blue",
        )
    )

    # Ask which analysis to show
    analysis_choice = questionary.select(
        "Which analysis method would you like to see?",
        choices=[
            questionary.Choice("Counting-based (fastest)", value="counts"),
            questionary.Choice("Monotone regression (individual-level)", value="monotone"),
            questionary.Choice("Hierarchical Bayes (most advanced)", value="hb"),
            questionary.Choice("All three", value="all"),
        ],
        style=SURVEY_STYLE,
    ).ask()

    if analysis_choice is None:
        analysis_choice = "counts"

    methods = {
        "counts": ("Counting-based", analyze_counts),
        "monotone": ("Monotone regression", analyze_monotone),
        "hb": ("Hierarchical Bayes", analyze_hb),
    }

    if analysis_choice == "all":
        for key, (name, func) in methods.items():
            console.print(f"\n[bold]--- {name} ---[/bold]")
            if key == "hb":
                with console.status("[bold cyan]Running MCMC...[/bold cyan]"):
                    result = func(results, seed=seed)
            else:
                result = func(results)
            _display_results(result, config)
    else:
        name, func = methods[analysis_choice]
        if analysis_choice == "hb":
            with console.status("[bold cyan]Running MCMC...[/bold cyan]"):
                result = func(results, seed=seed)
        else:
            result = func(results)
        _display_results(result, config)

    # Offer export
    console.print()
    export = questionary.confirm(
        "Would you like to export the results?",
        default=False,
        style=SURVEY_STYLE,
    ).ask()

    if export:
        # Use the last computed result for export
        export_format = questionary.select(
            "Export format:",
            choices=[
                questionary.Choice("JSON", value="json"),
                questionary.Choice("CSV", value="csv"),
            ],
            style=SURVEY_STYLE,
        ).ask()

        content = result.to_json() if export_format == "json" else result.to_csv()
        suffix = ".json" if export_format == "json" else ".csv"
        out_path = Path(f"acbc_results{suffix}")

        try:
            out_path.write_text(content)
            console.print(f"\n[green]Results exported to {out_path.resolve()}[/green]")
        except (OSError, TimeoutError):
            # iCloud-synced or network folders can time out on write;
            # fall back to a temp location, then print to stdout.
            import tempfile
            fallback = Path(tempfile.gettempdir()) / f"acbc_results{suffix}"
            try:
                fallback.write_text(content)
                console.print(
                    f"\n[yellow]Could not write to project directory. "
                    f"Results saved to {fallback}[/yellow]"
                )
            except OSError:
                console.print(f"\n[yellow]Could not write file. Printing results:[/yellow]")
                console.print(content)

    console.print(
        "\n[bold green]Thank you for completing the survey![/bold green]\n"
    )
