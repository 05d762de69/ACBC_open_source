"""
Data persistence for ACBC survey results.

Handles serialization / deserialization of:
- **Raw results** — the full engine output for each participant, written
  to ``data/raw/<participant_id>_<timestamp>.json``.
- **Analysis results** — computed utilities and importances, written to
  ``data/analysis/<participant_id>_<method>_<timestamp>.json``.

All writes use the iCloud-safe fallback pattern (project dir → /tmp → stdout).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _safe_write(path: Path, content: str, *, console: Any = None) -> Path | None:
    """
    Write *content* to *path*, falling back to ``/tmp`` then stdout.

    Returns the path that was actually written, or ``None`` if we had to
    print to the console instead.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path
    except (OSError, TimeoutError):
        pass

    import tempfile

    fallback = Path(tempfile.gettempdir()) / path.name
    try:
        fallback.write_text(content)
        return fallback
    except OSError:
        if console is not None:
            console.print(f"[yellow]Could not write file. Printing content:[/yellow]")
            console.print(content)
        return None


# ------------------------------------------------------------------
# Timestamp helper
# ------------------------------------------------------------------

def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ------------------------------------------------------------------
# Raw results serialization
# ------------------------------------------------------------------

def _serialize_scenario(scenario: Any) -> dict[str, str]:
    """Scenario → plain dict of {attr: level}."""
    if scenario is None:
        return {}
    if hasattr(scenario, "levels"):
        return dict(scenario.levels)
    return dict(scenario)


def serialize_raw_results(
    results: dict[str, Any],
    participant_id: str,
    *,
    seed: int | None = None,
) -> dict[str, Any]:
    """Convert the engine's result dict into a JSON-safe dictionary."""
    config = results["config"]

    screening_scenarios = [
        [_serialize_scenario(sc) for sc in page]
        for page in results.get("screening_scenarios", [])
    ]

    screening_responses = [
        {"page_number": sr.page_number, "responses": {str(k): v for k, v in sr.responses.items()}}
        for sr in results.get("screening_responses", [])
    ]

    confirmed_rules = [
        {"attribute_name": r.attribute_name, "level": r.level, "rule_type": r.rule_type}
        for r in results.get("confirmed_rules", [])
    ]

    choice_responses = [
        {"round_number": cr.round_number, "chosen_index": cr.chosen_index}
        for cr in results.get("choice_responses", [])
    ]

    tournament_rounds = [
        [_serialize_scenario(sc) for sc in group]
        for group in results.get("tournament_rounds", [])
    ]

    return {
        "participant_id": participant_id,
        "timestamp": _timestamp(),
        "seed": seed,
        "config_name": config.name,
        "config": {
            "name": config.name,
            "description": config.description,
            "attributes": [
                {"name": a.name, "levels": list(a.levels)}
                for a in config.attributes
            ],
            "settings": config.settings.model_dump(),
        },
        "byo_ideal": _serialize_scenario(results.get("byo_ideal")),
        "screening_scenarios": screening_scenarios,
        "screening_responses": screening_responses,
        "confirmed_rules": confirmed_rules,
        "choice_responses": choice_responses,
        "tournament_rounds": tournament_rounds,
        "winner": _serialize_scenario(results.get("winner")),
        "level_shown_count": results.get("level_shown_count", {}),
        "level_accepted_count": results.get("level_accepted_count", {}),
        "level_chosen_count": results.get("level_chosen_count", {}),
    }


def save_raw_results(
    results: dict[str, Any],
    participant_id: str,
    output_dir: Path,
    *,
    seed: int | None = None,
    console: Any = None,
) -> Path | None:
    """
    Serialize and save raw survey results to ``<output_dir>/raw/<id>_<ts>.json``.
    """
    payload = serialize_raw_results(results, participant_id, seed=seed)
    ts = payload["timestamp"]
    filename = f"{participant_id}_{ts}.json"
    path = output_dir / "raw" / filename
    content = json.dumps(payload, indent=2, ensure_ascii=False)
    return _safe_write(path, content, console=console)


# ------------------------------------------------------------------
# Analysis results serialization
# ------------------------------------------------------------------

def save_analysis_results(
    analysis_result: Any,
    participant_id: str,
    output_dir: Path,
    *,
    console: Any = None,
) -> Path | None:
    """
    Save an ``AnalysisResult`` to ``<output_dir>/analysis/<id>_<method>_<ts>.json``.
    """
    payload = {
        "participant_id": participant_id,
        "timestamp": _timestamp(),
        **analysis_result.to_dict(),
    }
    method = analysis_result.method
    ts = payload["timestamp"]
    filename = f"{participant_id}_{method}_{ts}.json"
    path = output_dir / "analysis" / filename
    content = json.dumps(payload, indent=2, ensure_ascii=False)
    return _safe_write(path, content, console=console)


# ------------------------------------------------------------------
# Loading saved results
# ------------------------------------------------------------------

def load_raw_results(path: Path) -> dict[str, Any]:
    """Load a single raw-results JSON file."""
    with path.open() as f:
        return json.load(f)


def load_all_raw_results(data_dir: Path) -> list[dict[str, Any]]:
    """
    Load every ``*.json`` file in ``<data_dir>/raw/``, sorted by filename
    (i.e. chronological order since filenames contain timestamps).
    """
    raw_dir = data_dir / "raw"
    if not raw_dir.is_dir():
        return []
    files = sorted(raw_dir.glob("*.json"))
    return [load_raw_results(f) for f in files]


def load_all_analysis_results(data_dir: Path) -> list[dict[str, Any]]:
    """Load every ``*.json`` file in ``<data_dir>/analysis/``."""
    analysis_dir = data_dir / "analysis"
    if not analysis_dir.is_dir():
        return []
    files = sorted(analysis_dir.glob("*.json"))
    results = []
    for f in files:
        with f.open() as fh:
            results.append(json.load(fh))
    return results


def reconstruct_results_for_analysis(raw: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a loaded raw-results JSON back into the dict format that
    ``analyze_counts`` / ``analyze_monotone`` / ``analyze_hb`` expect.

    This bridges the gap between the JSON-serialized form and the in-memory
    Pydantic objects the analysis functions need.
    """
    from acbc.models import (
        ChoiceResponse,
        NonCompensatoryRule,
        Scenario,
        ScreeningResponse,
        SurveyConfig,
    )

    config = SurveyConfig.model_validate(raw["config"])
    byo_ideal = Scenario(levels=raw["byo_ideal"]) if raw.get("byo_ideal") else None

    screening_scenarios = [
        [Scenario(levels=sc) for sc in page]
        for page in raw.get("screening_scenarios", [])
    ]

    screening_responses = [
        ScreeningResponse(
            page_number=sr["page_number"],
            responses={int(k): v for k, v in sr["responses"].items()},
        )
        for sr in raw.get("screening_responses", [])
    ]

    confirmed_rules = [
        NonCompensatoryRule(**r) for r in raw.get("confirmed_rules", [])
    ]

    choice_responses = [
        ChoiceResponse(**cr) for cr in raw.get("choice_responses", [])
    ]

    tournament_rounds = [
        [Scenario(levels=sc) for sc in group]
        for group in raw.get("tournament_rounds", [])
    ]

    winner = Scenario(levels=raw["winner"]) if raw.get("winner") else None

    return {
        "config": config,
        "byo_ideal": byo_ideal,
        "screening_scenarios": screening_scenarios,
        "screening_responses": screening_responses,
        "confirmed_rules": confirmed_rules,
        "choice_responses": choice_responses,
        "tournament_rounds": tournament_rounds,
        "winner": winner,
        "level_shown_count": raw.get("level_shown_count", {}),
        "level_accepted_count": raw.get("level_accepted_count", {}),
        "level_chosen_count": raw.get("level_chosen_count", {}),
    }
