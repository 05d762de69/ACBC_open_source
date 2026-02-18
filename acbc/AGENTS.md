# AGENTS.md — ACBC Engine (`acbc/`)

## Purpose

This package contains the **core ACBC engine** — all survey logic, scenario generation, screening detection, and statistical analysis. It is **frontend-agnostic**: no terminal I/O, no UI dependencies.

## Module Responsibilities

### `models.py` — Data Models
- **Survey config**: `Attribute`, `SurveySettings`, `SurveyConfig` (with YAML loader).
- **Scenarios**: `Scenario` — a dict mapping attribute names to chosen levels.
- **Questions**: `BYOQuestion`, `ScreeningQuestion`, `UnacceptableQuestion`, `MustHaveQuestion`, `ChoiceQuestion` — typed objects the engine returns to the frontend.
- **Responses**: `ScreeningResponse`, `ChoiceResponse`, `NonCompensatoryRule`.
- **State**: `SurveyState` — mutable state tracking all respondent data.

### `engine.py` — Survey State Machine
- `ACBCEngine` is the main class. Frontend interacts via:
  - `get_current_question()` → returns a `Question` union type
  - `submit_answer(answer)` → advances state
  - `is_complete` → check if survey is done
  - `get_results()` → raw data dict for analysis
- Manages transitions: INTRO → BYO → SCREENING → UNACCEPTABLE → MUST_HAVE → CHOICE_TOURNAMENT → COMPLETE.

### `design.py` — Scenario Generation
- `generate_screening_scenarios()` — creates near-neighbor scenarios from BYO ideal with level-balance bias.
- `generate_tournament_pool()` — builds filtered pool for choice tournament (applies unacceptable/must-have rules).
- `chunk_tournament_pool()` — splits pool into choice sets.

### `screening.py` — Non-Compensatory Detection
- `detect_unacceptable_candidates()` — finds levels with high rejection rates.
- `detect_must_have_candidates()` — finds levels with very high acceptance rates and large gap from alternatives.
- `get_accepted_scenarios()` — collects all scenarios marked as "a possibility".

### `analysis.py` — Utility Estimation
Three tiers, all returning `AnalysisResult`:
- `analyze_counts()` — frequency-based, fastest.
- `analyze_monotone()` — isotonic regression (Pool Adjacent Violators).
- `analyze_hb()` — Hierarchical Bayes with MCMC Metropolis-Hastings.

`AnalysisResult` includes:
- `level_utilities` — per-level utility values (zero-centered per attribute).
- `attribute_importances` — relative importance (sums to 100%).
- `predicted_winner` — best level per attribute.
- Export: `.to_json()`, `.to_csv()`.

## Adding New Features

- **New question type**: Add model to `models.py`, add to `Question` union, handle in `engine.py` state machine, render in `cli/survey.py`.
- **New analysis method**: Follow the `analyze_*()` pattern — accept `results: dict`, return `AnalysisResult`.
- **New screening heuristic**: Add detection function in `screening.py`, wire it into the engine's `_prepare_*` methods.

## Testing

Run a headless test by instantiating `ACBCEngine` and feeding it answers programmatically:

```python
engine = ACBCEngine(config, seed=42)
while not engine.is_complete:
    q = engine.get_current_question()
    engine.submit_answer(some_answer)
results = engine.get_results()
```
