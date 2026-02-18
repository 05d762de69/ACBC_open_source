# AGENTS.md — ACBC Survey Engine (Project Root)

## Project Overview

This is an open-source implementation of **Adaptive Choice-Based Conjoint (ACBC)** analysis — an advanced survey methodology that adaptively learns respondent preferences through a three-stage process. It is modeled after Sawtooth Software's ACBC, based on the methodology described in:

- **Al-Omari et al. (2017)** — "Generating Individual Patient Preferences for the Treatment of Osteoarthritis Using ACBC Analysis"
- **Sanchez (2019)** — Master Thesis on Adaptive Kano Choice-Based Conjoint Analysis

## Architecture

The codebase has a strict **engine/frontend separation**:

- **`acbc/`** — The core engine. Pure Python, no I/O, no terminal dependencies. Can be driven by any frontend (CLI, web API, etc.). Uses a state-machine pattern.
- **`cli/`** — The CLI frontend. Uses `questionary` for keyboard navigation and `rich` for formatted output. This is the only module that does terminal I/O.
- **`configs/`** — YAML survey configuration files. Attributes and levels are loaded from here.
- **`main.py`** — CLI entry point.

## ACBC Survey Flow

The survey has three main stages:

1. **BYO (Build Your Own)** — Respondent picks preferred level for each attribute.
2. **Screening** — Near-neighbor scenarios are shown; respondent marks each as "possibility" or "won't work". The engine detects non-compensatory rules:
   - **Unacceptable**: levels systematically avoided (confirmed with respondent).
   - **Must-have**: levels consistently required (confirmed with respondent).
3. **Choice Tournament** — Filtered concepts compete head-to-head until a winner emerges.

After the survey, three analysis methods are available:
- **Counting-based** — Simple frequency-based utilities.
- **Monotone regression** — Isotonic regression for ordinal utilities.
- **Hierarchical Bayes** — MCMC Metropolis-Hastings for Bayesian part-worth estimation.

## Key Design Patterns

- **State machine**: `ACBCEngine` exposes `get_current_question()` / `submit_answer()`. The frontend never manages survey logic.
- **Typed questions**: The engine returns union-typed question objects (`BYOQuestion`, `ScreeningQuestion`, etc.) that the frontend pattern-matches on.
- **YAML config**: All survey attributes/levels/settings are in YAML. No survey content is hardcoded.
- **Pydantic models**: All data structures use Pydantic for validation.

## Running

```bash
# Default demo (laptop survey)
uv run python main.py

# Custom config
uv run python main.py --config configs/my_survey.yaml

# With fixed seed for reproducibility
uv run python main.py --seed 42
```

## Key Conventions

- Python 3.11+, type hints everywhere.
- Use `uv` for dependency management (not pip directly).
- When adding new analysis methods, follow the pattern in `acbc/analysis.py`: return an `AnalysisResult` with level utilities and attribute importances.
- When adding new question types, add the model to `acbc/models.py`, handle it in `acbc/engine.py`, and render it in `cli/survey.py`.
