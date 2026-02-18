# ACBC Survey Engine

An open-source implementation of **Adaptive Choice-Based Conjoint (ACBC)** analysis — the advanced survey methodology used by [Sawtooth Software](https://sawtoothsoftware.com/conjoint-analysis/acbc) for preference elicitation and conjoint studies.

This is a proof-of-concept / pilot implementation for research purposes.

## What is ACBC?

ACBC is a survey methodology that adaptively learns respondent preferences through three stages:

1. **Build Your Own (BYO)** — The respondent defines their ideal product by picking preferred levels for each attribute.
2. **Screening** — Near-neighbor scenarios are generated around the ideal; the respondent evaluates each as "a possibility" or "won't work for me". The system detects non-compensatory rules:
   - **Unacceptable** levels (systematically avoided)
   - **Must-have** levels (consistently required)
3. **Choice Tournament** — Filtered concepts compete in head-to-head choice tasks until a winner emerges.

## Features

- **YAML-configurable surveys** — Define attributes, levels, and settings in simple YAML files.
- **Three analysis methods**:
  - Counting-based (simple, fast)
  - Monotone regression (individual-level ordinal utilities)
  - Hierarchical Bayes (MCMC Metropolis-Hastings for Bayesian estimation)
- **Engine/frontend separation** — The core engine is UI-agnostic and can be driven by a CLI, web API, or any other frontend.
- **Interactive CLI** — Keyboard-driven terminal survey using arrow keys and Enter.

## Quick Start

```bash
# Install dependencies
uv sync

# Run the demo survey (laptop preferences)
uv run python main.py

# Run with a custom config
uv run python main.py --config configs/my_survey.yaml

# With fixed random seed for reproducibility
uv run python main.py --seed 42
```

## Creating Your Own Survey

Create a YAML file in `configs/`:

```yaml
name: "My Product Survey"
description: "Exploring preferences for my product"

attributes:
  - name: Brand
    levels: [Apple, Samsung, Google]
  - name: Price
    levels: ["$299", "$499", "$699"]
  - name: Battery
    levels: [Small, Medium, Large]

settings:
  screening_pages: 5
  scenarios_per_page: 4
  choice_tournament_size: 3
```

Then run: `uv run python main.py --config configs/my_survey.yaml`

## Using the Engine Programmatically

The engine can be used without the CLI for integration into other systems:

```python
from acbc.models import SurveyConfig
from acbc.engine import ACBCEngine
from acbc.analysis import analyze_counts, analyze_monotone, analyze_hb

config = SurveyConfig.from_yaml("configs/demo_laptop.yaml")
engine = ACBCEngine(config, seed=42)

while not engine.is_complete:
    question = engine.get_current_question()
    # Your frontend renders the question and collects the answer
    answer = your_frontend.ask(question)
    engine.submit_answer(answer)

results = engine.get_results()
analysis = analyze_hb(results, seed=42)
print(analysis.to_json())
```

## References

- Al-Omari, B., Sim, J., Croft, P., & Frisher, M. (2017). Generating Individual Patient Preferences for the Treatment of Osteoarthritis Using Adaptive Choice-Based Conjoint (ACBC) Analysis. *Rheumatology and Therapy*, 4, 167–182.
- Sanchez, O. F. (2019). Adaptive Kano Choice-Based Conjoint Analysis (AK-CBC). *Master Thesis, Erasmus University Rotterdam*.
- Johnson, R., & Orme, B. (2007). A New Approach to Adaptive CBC. *Sawtooth Software Technical Paper*.

## License

This project is for research/academic use.
