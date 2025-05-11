import sys, os
import argparse
from pathlib import Path
from src.flair_zonal_detection.inference import run_inference


current = Path(__file__).resolve()
while current.name != "src" and current != current.parent:
    current = current.parent
src_root = current
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the detection config file")
    args = parser.parse_args()
    
    run_inference(args.config)

