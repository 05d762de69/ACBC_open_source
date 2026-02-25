"""
ACBC - Adaptive Choice-Based Conjoint Analysis Engine

An open-source implementation of the ACBC methodology for conjoint analysis.
The engine is frontend-agnostic and can be driven by any UI (CLI, web, etc.).
"""

from acbc.models import (
    Attribute,
    SurveyConfig,
    Scenario,
    SurveyStage,
)
from acbc.engine import ACBCEngine
from acbc.io import (
    load_all_raw_results,
    reconstruct_results_for_analysis,
    save_analysis_results,
    save_raw_results,
)

__all__ = [
    "Attribute",
    "SurveyConfig",
    "Scenario",
    "SurveyStage",
    "ACBCEngine",
    "save_raw_results",
    "save_analysis_results",
    "load_all_raw_results",
    "reconstruct_results_for_analysis",
]
