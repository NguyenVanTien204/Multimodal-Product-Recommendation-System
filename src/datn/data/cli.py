from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from .pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the resource-aware phase 1 data pipeline")
    parser.add_argument("--config", type=Path, default=Path("configs/phase1.yaml"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    plan, stats = run(config, overwrite=args.overwrite or None)
    print(f"Completed with a {plan.budget_gb:.2f} GiB memory ceiling: {stats}")


if __name__ == "__main__":
    main()
