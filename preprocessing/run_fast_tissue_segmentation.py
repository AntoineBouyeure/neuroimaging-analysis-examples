#!/usr/bin/env python3
"""Run FSL FAST tissue segmentation on anatomical images."""
from __future__ import annotations

import argparse
from pathlib import Path
import logging
import subprocess

LOGGER = logging.getLogger("run_fast_tissue_segmentation")


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def run_fast(t1_brain: Path, output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "fast",
        "-n", "3",
        "-t", "1",
        "-o", str(output_prefix),
        "-B",
        str(t1_brain),
    ]
    subprocess.run(cmd, check=True)
    LOGGER.info("Completed FAST for %s", t1_brain)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t1-brain", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)
    for t1 in args.t1_brain:
        run_fast(t1, args.output_dir / f"{t1.stem}_fast")


if __name__ == "__main__":
    main()
