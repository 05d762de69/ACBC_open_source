"""
Entry point for the ACBC survey CLI.

Usage:
    python main.py                          # Run with default demo config
    python main.py --config path/to.yaml    # Run with custom config
    python main.py --seed 42                # Run with fixed random seed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cli.survey import run_survey


DEFAULT_CONFIG = Path(__file__).parent / "configs" / "demo_laptop.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ACBC — Adaptive Choice-Based Conjoint Analysis Survey",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Path to YAML survey config (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible scenario generation",
    )
    args = parser.parse_args()

    if not args.config.exists():
        print(f"Error: config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    run_survey(args.config, seed=args.seed)


if __name__ == "__main__":
    main()
