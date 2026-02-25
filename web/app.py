"""
Web frontend for the ACBC survey engine.

A minimal FastAPI application that wraps the engine with HTML form-based
interactions.  No JavaScript framework — pure HTML + CSS with server-side
rendering via Jinja2 templates.

Sessions are stored in-memory (a dict keyed by UUID cookie).  This is
fine for local / single-process use.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from acbc.engine import ACBCEngine
from acbc.models import (
    BYOQuestion,
    ChoiceQuestion,
    MustHaveQuestion,
    ScreeningQuestion,
    SurveyConfig,
    UnacceptableQuestion,
)
from acbc.io import save_raw_results

WEB_DIR = Path(__file__).parent
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Module-level state — set by create_app()
_config: SurveyConfig | None = None
_seed: int | None = None
_output_dir: Path = Path("data")

SESSION_COOKIE = "acbc_session"

sessions: dict[str, dict[str, Any]] = {}


def _next_participant_id() -> str:
    """Generate the next sequential participant ID (P001, P002, ...)."""
    raw_dir = _output_dir / "raw"
    highest = 0
    if raw_dir.is_dir():
        for f in raw_dir.glob("*.json"):
            prefix = f.stem.split("_")[0]
            if prefix.startswith("P") and prefix[1:].isdigit():
                highest = max(highest, int(prefix[1:]))
    return f"P{highest + 1:03d}"


STAGE_ORDER = ["byo", "screening", "unacceptable", "must_have", "choice_tournament", "complete"]


def _stage_progress(stage: str) -> int:
    """Return a 0-100 progress percentage for the current stage."""
    if stage in STAGE_ORDER:
        idx = STAGE_ORDER.index(stage)
        return int(idx / (len(STAGE_ORDER) - 1) * 100)
    return 0


def create_app(
    config: SurveyConfig,
    *,
    seed: int | None = None,
    output_dir: Path = Path("data"),
) -> FastAPI:
    """Build and return the FastAPI application."""
    global _config, _seed, _output_dir
    _config = config
    _seed = seed
    _output_dir = output_dir

    app = FastAPI(title="ACBC Survey")
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # ── Routes ──────────────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def welcome(request: Request):
        return templates.TemplateResponse("welcome.html", {
            "request": request,
            "config": _config,
        })

    @app.post("/start")
    async def start(request: Request):
        engine = ACBCEngine(_config, seed=_seed)
        pid = _next_participant_id()
        sid = uuid.uuid4().hex
        sessions[sid] = {
            "engine": engine,
            "participant_id": pid,
        }
        response = RedirectResponse(url="/question", status_code=303)
        response.set_cookie(SESSION_COOKIE, sid)
        return response

    @app.get("/question", response_class=HTMLResponse)
    async def question_page(request: Request):
        sid = request.cookies.get(SESSION_COOKIE)
        if not sid or sid not in sessions:
            return RedirectResponse(url="/")

        session = sessions[sid]
        engine: ACBCEngine = session["engine"]

        if engine.is_complete:
            return RedirectResponse(url="/complete")

        q = engine.get_current_question()
        attr_names = [a.name for a in engine.config.attributes]
        progress = _stage_progress(q.stage)

        ctx = {
            "request": request,
            "question": q,
            "progress": progress,
            "participant_id": session["participant_id"],
        }

        if isinstance(q, BYOQuestion):
            return templates.TemplateResponse("byo.html", ctx)

        if isinstance(q, ScreeningQuestion):
            ctx["attr_names"] = attr_names
            return templates.TemplateResponse("screening.html", ctx)

        if isinstance(q, UnacceptableQuestion):
            ctx["rule_type"] = "unacceptable"
            return templates.TemplateResponse("rule_check.html", ctx)

        if isinstance(q, MustHaveQuestion):
            ctx["rule_type"] = "must_have"
            return templates.TemplateResponse("rule_check.html", ctx)

        if isinstance(q, ChoiceQuestion):
            if not q.scenarios:
                return RedirectResponse(url="/complete")
            ctx["attr_names"] = attr_names
            return templates.TemplateResponse("choice.html", ctx)

        return RedirectResponse(url="/")

    @app.post("/answer")
    async def submit_answer(request: Request):
        sid = request.cookies.get(SESSION_COOKIE)
        if not sid or sid not in sessions:
            return RedirectResponse(url="/")

        session = sessions[sid]
        engine: ACBCEngine = session["engine"]
        form = await request.form()

        q = engine.get_current_question()

        if isinstance(q, BYOQuestion):
            answer = form.get("level")
            if answer:
                engine.submit_answer(answer)

        elif isinstance(q, ScreeningQuestion):
            responses: dict[int, bool] = {}
            for i in range(len(q.scenarios)):
                val = form.get(f"scenario_{i}")
                responses[i] = val == "accept"
            engine.submit_answer(responses)

        elif isinstance(q, (UnacceptableQuestion, MustHaveQuestion)):
            answer = form.get("confirmed") == "yes"
            engine.submit_answer(answer)

        elif isinstance(q, ChoiceQuestion):
            chosen = form.get("chosen")
            if chosen is not None:
                engine.submit_answer(int(chosen))

        if engine.is_complete:
            results = engine.get_results()
            save_raw_results(
                results,
                session["participant_id"],
                _output_dir,
                seed=_seed,
            )
            return RedirectResponse(url="/complete", status_code=303)

        return RedirectResponse(url="/question", status_code=303)

    @app.get("/complete", response_class=HTMLResponse)
    async def complete(request: Request):
        sid = request.cookies.get(SESSION_COOKIE)
        if not sid or sid not in sessions:
            return RedirectResponse(url="/")

        session = sessions[sid]
        engine: ACBCEngine = session["engine"]
        winner = engine.state.winner
        attr_names = [a.name for a in engine.config.attributes]

        return templates.TemplateResponse("complete.html", {
            "request": request,
            "participant_id": session["participant_id"],
            "winner": winner,
            "attr_names": attr_names,
            "output_dir": str(_output_dir),
        })

    return app
