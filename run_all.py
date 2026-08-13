#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from shumo_ccc.pipeline import run_all


def main():
    parser = argparse.ArgumentParser(description="Run all four Huashu Cup C analyses")
    parser.add_argument("--data-dir", default="data", help="Directory containing the six XLSX inputs")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--config", default="config.json", help="JSON configuration")
    parser.add_argument("--quick", action="store_true", help="Use five Q4 scenarios instead of the full 3x3 grid")
    args = parser.parse_args()
    result = run_all(args.data_dir, args.output_dir, args.config, args.quick)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    required = ["q1_valid", "q2_valid", "q3_valid", "q4_all_schedules_valid", "q4_all_energy_valid"]
    if not all(bool(result.get(k)) for k in required):
        raise SystemExit("Validation gate failed: " + ", ".join(k for k in required if not result.get(k)))
    if not all(result[k] for k in ["q1_valid", "q2_valid", "q3_valid", "q4_all_schedules_valid"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
