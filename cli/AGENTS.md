# AGENTS.md — CLI Frontend (`cli/`)

## Purpose

This package contains the **terminal-based user interface** for the ACBC survey. It is the only part of the codebase that does I/O. The engine (`acbc/`) handles all survey logic.

## Module: `survey.py`

### Dependencies
- **`questionary`** — keyboard-driven prompts (arrow keys + enter). Used for all selection UIs.
- **`rich`** — formatted output (tables, panels, colored text, progress). Used for scenario display and results.

### Key Functions

- `run_survey(config_path, seed)` — Main entry point. Loads config, creates engine, runs the interactive loop, displays results.
- `_ask_byo(question)` → `str` — Render BYO attribute selection.
- `_ask_screening(question, attr_names)` → `dict[int, bool]` — Render scenario table, collect accept/reject per scenario.
- `_ask_unacceptable(question)` → `bool` — Confirm unacceptable level.
- `_ask_must_have(question)` → `bool` — Confirm must-have level.
- `_ask_choice(question, attr_names)` → `int` — Render choice task, collect selection.
- `_display_results(result, config)` — Render analysis results (bar charts, utility tables).
- `_display_winner(winner, config)` — Show tournament winner.

### Rendering Pattern

Each question type has a corresponding `_ask_*` function that:
1. Prints a `rich.Panel` header indicating the current stage.
2. If applicable, renders a `rich.Table` showing scenarios side-by-side.
3. Uses `questionary.select()` for the actual choice.
4. Returns the answer in the format expected by `engine.submit_answer()`.

### Adding New Question Types

1. Import the new question model from `acbc.models`.
2. Create an `_ask_new_type(question)` function following the pattern above.
3. Add an `isinstance` check in the main loop inside `run_survey()`.

### User Cancellation

All `questionary` calls check for `None` (Ctrl+C) and exit gracefully.
