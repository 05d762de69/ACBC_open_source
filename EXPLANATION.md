# ACBC Survey Engine — Detailed Technical Explanation

## The Big Picture

This project is an **open-source implementation of Adaptive Choice-Based Conjoint (ACBC)** — the same methodology that [Sawtooth Software](https://sawtoothsoftware.com/conjoint-analysis/acbc) sells commercially. The goal is to have a fully transparent, customizable engine that can be used for academic research, particularly in decision-making contexts.

The architecture has a deliberate **two-layer separation**:

- **The engine** (`acbc/` package) — pure logic, no I/O. It can be driven by any frontend: a CLI today, a web app tomorrow, a lab experiment script, etc.
- **The frontend** (`cli/` package) — a terminal-based UI. This is the only code that reads from the keyboard or prints to the screen.

This means if you want to deploy this as a web survey, you only need to build a new frontend that calls the same engine API. The survey logic, scenario generation, screening detection, and statistical analysis all stay the same.

---

## Project Structure

```
pilot-acbc-ddm/
├── acbc/                  # Core engine (no I/O, frontend-agnostic)
│   ├── models.py          # Data models (attributes, scenarios, questions, state)
│   ├── engine.py          # Survey state machine
│   ├── design.py          # Scenario generation algorithms
│   ├── screening.py       # Non-compensatory rule detection
│   └── analysis.py        # Utility estimation (3 methods)
├── cli/                   # Terminal frontend (only I/O code)
│   └── survey.py          # Keyboard-driven survey UI
├── configs/               # YAML survey definitions
│   └── demo_laptop.yaml   # Example config
└── main.py                # Entry point
```

### Dependency Order (foundation to top-level)

```
1. configs/*.yaml           ← pure data, no code dependencies
2. acbc/models.py           ← foundation: all data types
3. acbc/screening.py        ← depends on models
4. acbc/design.py           ← depends on models
5. acbc/engine.py           ← depends on models, design, screening
6. acbc/analysis.py         ← depends on models (via results dict), numpy, scipy
7. cli/survey.py            ← depends on engine, models, analysis, questionary, rich
8. main.py                  ← depends on cli.survey
```

### External Dependencies

| Package        | Role                                                                                     | Used in          |
|----------------|------------------------------------------------------------------------------------------|------------------|
| **pydantic**   | Data validation and typed models. All data structures are Pydantic `BaseModel` subclasses with automatic validation, serialization, and type checking. | `acbc/models.py` |
| **pyyaml**     | Loads survey configurations from YAML files. `SurveyConfig.from_yaml()` reads a YAML file and validates it through Pydantic. | `acbc/models.py` |
| **questionary** | Keyboard-driven terminal prompts. Provides `select()` (arrow keys + Enter) for all user choices. Built on top of `prompt_toolkit`. | `cli/survey.py`  |
| **rich**       | Formatted terminal output — colored panels for stage headers, tables for side-by-side scenario comparison, bar charts for results. | `cli/survey.py`  |
| **numpy**      | Numerical arrays for the analysis module — stores utility vectors, computes means/standard deviations, matrix operations for the MNL design matrix. | `acbc/analysis.py` |
| **scipy**      | Specifically `scipy.optimize.minimize` for L-BFGS-B optimization — used to find the MLE starting point for the HB analysis. Imported lazily (only when HB is selected). | `acbc/analysis.py` |

---

## Step-by-Step: How the Code Flows

### 1. Configuration Loading

Everything begins with a YAML config file. Example (`configs/demo_laptop.yaml`):

```yaml
name: "Decision-Making Under Risk Preferences"
description: >
  A simple demo survey exploring preferences when making decisions under risk.

attributes:
  - name: Reward Size
    levels: ["30", "60", "120"]

  - name: Reward Variability
    levels:
      - Low (outcomes stay close to the expected value)
      - Moderate (outcomes can swing moderately)
      - High (outcomes can swing widely)

  - name: Worst Case Outcome
    levels: ["0", "-25", "-75"]
    
  - name: Timing
    levels: [immediately, in one month, in three months]

settings:
  screening_pages: 5
  scenarios_per_page: 4
  max_unacceptable_questions: 4
  max_must_have_questions: 3
  choice_tournament_size: 3
  unacceptable_threshold: 0.75
  must_have_threshold: 0.90
```

This defines **what** is being studied (attributes and their levels) and **how** the survey behaves (number of screening pages, thresholds for detecting non-compensatory rules, etc.). The YAML is parsed by `SurveyConfig.from_yaml()` in `acbc/models.py`, which uses Pydantic to validate that all attribute names are unique, each attribute has at least 2 levels, and all settings are within valid ranges.

The key data structures created here:

- **`Attribute`** — a name (e.g., "Reward Size") with a list of levels (e.g., ["30", "60", "120"])
- **`Scenario`** — a dictionary mapping each attribute name to one specific level. Represents one hypothetical "product" or "option" the respondent evaluates.
- **`SurveySettings`** — all the knobs that control the adaptive behaviour

---

### 2. The Engine State Machine

When the survey starts, `main.py` calls `run_survey()` in `cli/survey.py`, which creates an `ACBCEngine`:

```python
config = SurveyConfig.from_yaml(config_path)
engine = ACBCEngine(config, seed=seed)
```

The engine is a **state machine** that progresses through stages. The frontend drives it with a simple loop:

```python
while not engine.is_complete:
    question = engine.get_current_question()   # Engine says "ask this"
    answer = collect_from_respondent(question)  # Frontend gets the answer
    engine.submit_answer(answer)                # Engine processes and advances
```

The stages are:

```
INTRO → BYO → SCREENING → UNACCEPTABLE → MUST_HAVE → CHOICE_TOURNAMENT → COMPLETE
```

Each call to `get_current_question()` returns a **typed question object** (e.g., `BYOQuestion`, `ScreeningQuestion`). The frontend pattern-matches on the type to know how to render it. Each call to `submit_answer()` takes the respondent's answer, updates the internal state, and potentially advances to the next stage.

| Stage              | Question Type          | Answer Format                               |
|--------------------|------------------------|---------------------------------------------|
| BYO                | `BYOQuestion`          | `str` — the selected level                  |
| Screening          | `ScreeningQuestion`    | `dict[int, bool]` — scenario index → accept/reject |
| Unacceptable       | `UnacceptableQuestion` | `bool` — confirmed?                         |
| Must-have          | `MustHaveQuestion`     | `bool` — confirmed?                         |
| Choice tournament  | `ChoiceQuestion`       | `int` — index of chosen scenario            |

---

### 3. Stage 1: Build Your Own (BYO)

**What the respondent sees:** One question per attribute. "Which Reward Size do you prefer?" with options [30, 60, 120]. They pick their ideal for each.

**What the engine does** (in `engine.py`):

The engine iterates through the attributes list one by one. For each, it returns a `BYOQuestion`. When the respondent picks a level, it is stored in `byo_selections`. Once all attributes have been answered, the engine constructs the **BYO ideal** — a `Scenario` representing the respondent's dream option:

```python
self._state.byo_ideal = Scenario(levels=dict(self._state.byo_selections))
# e.g., {"Reward Size": "120", "Reward Variability": "Low ...", 
#         "Worst Case Outcome": "0", "Timing": "immediately"}
```

This ideal is the anchor for everything that follows. It transitions to screening.

**Why this matters methodologically:** The BYO stage is a key ACBC innovation over standard CBC. By having the respondent explicitly state their ideal, the subsequent screening scenarios can be generated *around* that ideal, making the survey more engaging and personally relevant. In standard CBC, scenarios are random and may feel irrelevant.

---

### 4. Scenario Generation (the "adaptive" part)

When BYO completes, the engine calls `generate_screening_scenarios()` in `acbc/design.py`. This is where the "adaptive" in ACBC happens.

**The algorithm:**

1. Start from the BYO ideal.
2. Generate candidate scenarios by **randomly swapping 1–3 attributes** to different levels. The `_random_swap()` function picks which attributes to change and what to change them to.
3. Swaps are weighted: 45% chance of changing 1 attribute, 35% for 2, 20% for 3. This keeps scenarios **near** the ideal (most differ in just 1–2 attributes), which is the "near-neighbour" approach described in the ACBC literature.
4. A **level-balance mechanism** tracks how often each level has appeared across all generated scenarios. Candidates that would cover under-represented levels are preferred. This approximates the experimental design principles of level balance and minimal overlap from conjoint theory.
5. Duplicates and exact copies of the ideal are excluded.
6. The scenarios are chunked into pages of 4 (configurable).

For a config with 5 pages × 4 scenarios, this generates 20 screening scenarios, each close to but different from the ideal.

---

### 5. Stage 2: Screening

**What the respondent sees:** A table showing 4 options side-by-side. For each, they say "A possibility" or "Won't work for me." This repeats for all pages (e.g., 5 pages = 20 total evaluations).

**What the engine does** (in `engine.py`):

For each page, it records the accept/reject decisions and updates **per-level tracking counters**:

- `level_shown_count` — how many times each level appeared in a scenario
- `level_accepted_count` — how many times a scenario containing that level was accepted

These counters are the raw data that feeds both the non-compensatory detection and the analysis.

**Why this matters:** Unlike standard CBC where the respondent directly chooses between options, screening is a simpler cognitive task (accept/reject). This lets the system show more scenarios without fatiguing the respondent, and the binary responses reveal patterns about which levels are deal-breakers.

---

### 6. Non-Compensatory Rule Detection

After screening completes, the engine analyses the response patterns. This is the code in `acbc/screening.py`.

#### Unacceptable Detection

For every level of every attribute, compute the **rejection rate**:

```
rejection_rate = 1 - (times_accepted / times_shown)
```

If a level was rejected in 75%+ of scenarios it appeared in (and it appeared at least twice), it is flagged as a candidate. For example, in the demo run, "High (outcomes can swing widely)" was flagged because the respondent rejected almost every scenario that contained it.

The respondent is then asked to confirm: *"You seemed to avoid 'High (outcomes can swing widely)' for Reward Variability. Is this totally unacceptable?"* If confirmed, it becomes a hard rule — this level will be excluded from all subsequent scenarios.

#### Must-Have Detection

The inverse: find levels with 90%+ acceptance rate where all other levels of the same attribute are at least 40 percentage points lower. This means the respondent only accepts scenarios with that specific level. They are asked to confirm.

**Why this matters methodologically:** This is the non-compensatory decision-making component from behavioural decision theory. Standard CBC assumes compensatory behaviour (respondents weigh all attributes against each other). But real humans often use simplifying heuristics — *"I won't take any medication with high risk of addiction, period."* ACBC captures these rules explicitly and uses them to filter the scenario space, making subsequent choices more meaningful.

---

### 7. Stage 3: Choice Tournament

#### Pool Generation (in `design.py`)

The engine builds a tournament pool from:

1. All scenarios the respondent marked as "a possibility" during screening
2. Filtered by confirmed unacceptable/must-have rules (any scenario with an unacceptable level is removed; any scenario not matching a must-have level is removed)
3. If the pool is too small (fewer than 6–9 scenarios), additional valid scenarios are generated near the ideal

The pool is split into groups of 3 (configurable).

#### Tournament Flow (in `engine.py`)

This works like a sports tournament bracket:

1. Present groups of 3 scenarios. Respondent picks the best one from each group.
2. The winners advance to the next round.
3. New groups are formed from the winners. Respondent picks again.
4. This repeats until only 1 scenario remains — the **tournament winner**.

In a typical run with ~9 valid scenarios in the pool, this takes about 4 rounds: 3 groups of 3 in round 1, then 3 winners compete, then the final winner emerges.

**Why this matters:** The tournament format is much more efficient than asking the respondent to rank all options. It also mimics real decision-making: you narrow down to a shortlist (screening), then compare finalists (tournament).

---

### 8. Analysis

After the survey, all collected data flows to `acbc/analysis.py`. Three methods are available, all producing the same output format (`AnalysisResult` containing level utilities, attribute importances, and a predicted ideal product).

#### Tier 1: Counting-Based

The simplest method. For each level:

- Compute `acceptance_rate = times_accepted / times_shown` from screening data
- Add a bonus for each time the level appeared in a tournament-winning scenario
- Zero-center within each attribute (so utilities sum to 0 per attribute)

**Attribute importance** is computed the same way across all three methods: take the range (max utility − min utility) within each attribute, then normalize so all importances sum to 100%. A wider range means more discriminating power = more important to the respondent.

#### Tier 2: Monotone Regression

Same raw scores as counting, but applies **isotonic regression** (Pool Adjacent Violators algorithm) to enforce monotone ordering. This smooths out noise: if the raw data suggests level A > B > C but B and C are nearly tied, isotonic regression can merge them.

This is the method used in Al-Omari et al. (2017) for individual-level estimation with Sawtooth Software's built-in monotone regression.

#### Tier 3: Hierarchical Bayes (MCMC)

The most statistically sophisticated. This is a Bayesian logit model estimated with MCMC:

1. **Encode the data**: Each scenario becomes a dummy-coded vector (a 1 for each level present, 0 otherwise). Tournament choices become direct choice observations. Screening accept/reject pairs are converted to pseudo-choices (each accepted scenario "beats" each rejected scenario on the same page).

2. **Find a starting point**: Use Maximum Likelihood Estimation (L-BFGS-B optimizer from scipy) on the Multinomial Logit model to get initial beta values.

3. **Run Metropolis-Hastings MCMC** for 2000 iterations:
   - Propose a new beta by adding random noise to the current one
   - Compute the log-posterior (log-likelihood from MNL + log-prior from N(0, I))
   - Accept the proposal if it improves the posterior, otherwise accept with probability proportional to the improvement ratio
   - During the first 500 iterations (burn-in), adapt the proposal scale to target a 20–50% acceptance rate

4. **Extract results**: The posterior mean of the chain (after discarding burn-in) gives the part-worth utilities. The posterior standard deviation gives uncertainty estimates.

This is a simplified version of the Sawtooth Software HB algorithm described in the Sanchez (2019) thesis. For a single respondent it is effectively a Bayesian logit with a normal prior.

---

### 9. Results Display

The CLI renders:

- **Attribute importance bar chart** — shows which attributes mattered most to this respondent
- **Level utility tables** — within each attribute, which levels are preferred (positive utility) vs. avoided (negative utility)
- **Predicted ideal product** — the combination with highest utility per attribute
- **Export to JSON/CSV**

---

## Data Flow Summary

```
YAML config ──→ SurveyConfig ──→ ACBCEngine
                                      │
                            BYO answers ──→ ideal Scenario
                                      │
                  generate_screening_scenarios(ideal) ──→ screening pages
                                      │
                  screening responses ──→ level_shown_count / level_accepted_count
                                      │
                  detect_unacceptable / detect_must_have ──→ confirmed rules
                                      │
                  generate_tournament_pool(ideal, accepted, rules) ──→ pool
                                      │
                  tournament choices ──→ winner + level_chosen_count
                                      │
                  analyze_*(results) ──→ AnalysisResult
                                              │
                                  level_utilities + attribute_importances
```

---

## What Makes This a Strong Pilot

1. **Transparent and reproducible**: Unlike Sawtooth's black box, every algorithm is visible and auditable. You can explain exactly how scenarios were generated, how rules were detected, and how utilities were estimated.

2. **Configurable for any domain**: Switch from laptop preferences to decision-making under risk by editing a YAML file. No code changes needed.

3. **Multiple analysis tiers**: You can show that the basic counting method and the full Bayesian method produce consistent results, or explore where they diverge — which is itself interesting for decision-making research.

4. **Frontend-agnostic**: The engine API (`get_current_question` / `submit_answer`) can be wired to a web interface for online studies, or embedded in a lab experiment script (e.g., PsychoPy), with zero changes to the core logic.

5. **Captures both compensatory and non-compensatory preferences**: The unacceptable/must-have detection explicitly models non-compensatory decision heuristics, which is directly relevant to DDM research.

---

## References

- Al-Omari, B., Sim, J., Croft, P., & Frisher, M. (2017). Generating Individual Patient Preferences for the Treatment of Osteoarthritis Using Adaptive Choice-Based Conjoint (ACBC) Analysis. *Rheumatology and Therapy*, 4, 167–182.
- Sanchez, O. F. (2019). Adaptive Kano Choice-Based Conjoint Analysis (AK-CBC). *Master Thesis, Erasmus University Rotterdam*.
- Johnson, R., & Orme, B. (2007). A New Approach to Adaptive CBC. *Sawtooth Software Technical Paper*.
- Orme, B. K. (2009). Hierarchical Bayes: Why All the Attention? *Sawtooth Software Technical Paper*.
