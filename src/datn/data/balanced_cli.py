from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

from .balanced import prepare_balanced_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the balanced recommendation benchmark")
    parser.add_argument("--config", type=Path, default=Path("configs/balanced_dataset.yaml"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override paths.output_dir; must point to a new versioned directory",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.output_dir is not None:
        config["paths"]["output_dir"] = str(args.output_dir)
    result = prepare_balanced_dataset(config)
    print(json.dumps(result.manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
