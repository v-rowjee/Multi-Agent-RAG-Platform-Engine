"""Calculate agreement after manually labelling exported judge decisions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("review_csv", type=Path, help="CSV containing human_label and judge_label columns (0 or 1).")
    arguments = parser.parse_args()
    frame = pd.read_csv(arguments.review_csv)
    required = {"human_label", "judge_label"}
    if not required <= set(frame):
        raise SystemExit("Review CSV must contain human_label and judge_label columns.")
    labelled = frame.dropna(subset=list(required))
    if labelled.empty:
        raise SystemExit("No manually labelled judge decisions were found.")
    agreement = (labelled["human_label"].astype(int) == labelled["judge_label"].astype(int)).mean()
    print(f"Reviewed cases: {len(labelled)}\nRaw agreement: {agreement:.3f}")


if __name__ == "__main__":
    main()
