"""
Entry point for running zonal detection inference using a YAML config.
"""

import sys
import argparse
from pathlib import Path

from src.flair_zonal_detection.inference import run_inference


# Ensure src is in the Python path
def ensure_src_in_sys_path() -> None:
    current = Path(__file__).resolve()
    while current.name != "src" and current != current.parent:
        current = current.parent
    if str(current) not in sys.path:
        sys.path.insert(0, str(current))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run zonal detection inference.")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the detection config file"
    )
    args = parser.parse_args()
    run_inference(args.config)


if __name__ == '__main__':
    ensure_src_in_sys_path()
    main()
