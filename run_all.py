"""Reproduce everything from the raw source package.

    python run_all.py                 # all stages
    python run_all.py --from task1    # resume from a stage

Stages: inspect -> dataset -> ssl -> task1 -> task2 -> task3 -> figures
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGES = [
    ("inspect", ["scripts/inspect_data.py"]),
    ("dataset", ["scripts/build_dataset.py"]),
    ("ssl", ["scripts/train_ssl.py"]),
    ("task1", ["scripts/task1_room.py"]),
    ("task2", ["scripts/task2_ssl.py"]),
    ("task3", ["scripts/task3_fewshot.py"]),
    ("figures", ["scripts/make_figures.py"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default="inspect", choices=[s for s, _ in STAGES])
    ap.add_argument("--only", choices=[s for s, _ in STAGES])
    args = ap.parse_args()
    names = [s for s, _ in STAGES]
    for name, cmd in STAGES:
        if args.only and name != args.only:
            continue
        if not args.only and names.index(name) < names.index(args.start):
            continue
        print(f"\n=== {name} ===", flush=True)
        subprocess.run([sys.executable, *cmd], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
