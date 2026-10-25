import argparse
import sys
from pathlib import Path

src_root = str(Path(__file__).resolve().parents[1])
if src_root not in sys.path:
    sys.path.insert(0, src_root)

from flair_zonal_detection.inference import run_inference

def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Run zonal detection inference.")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the detection config file"
    )
    args = parser.parse_args()
    run_inference(args.config)

if __name__ == '__main__':
    main()
