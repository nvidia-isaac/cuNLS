#!/usr/bin/env python3
"""Plot benchmark_schur_in_docker.sh's CSV; requires matplotlib and numpy."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("output", type=Path, help="Output filename stem (PNG and SVG)")
    parser.add_argument("--hardware", help="Override the GPU name recorded in the CSV")
    args = parser.parse_args()
    with args.csv.open() as source:
        rows = list(csv.DictReader(source))

    hardware = args.hardware or rows[0].get("device", "GPU not recorded").replace("_", " ")
    plt.style.use("default")
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), layout="constrained")
    for ax, stage, title in zip(
        axes,
        ("SBA_WARM", "SBA_LINEAR"),
        ("Complete warm SBA optimization", "Frozen initial damped linear solve"),
    ):
        data = [row for row in rows if row["stage"] == stage]
        if not data:
            raise ValueError(f"Missing measurements for {stage}")
        positions = np.arange(len(data))
        for offset, key, label, color in (
            (-0.2, "pcg", "BlockSparsePCG", "tab:blue"),
            (0.2, "schur", "GeneralizedSchur", "tab:orange"),
        ):
            values = [float(row[f"{key}_median_ms"]) for row in data]
            bars = ax.bar(positions + offset, values, 0.38, label=label, color=color)
            ax.bar_label(bars, labels=[f"{time_ms:.2f}" for time_ms in values], padding=3, fontsize=9)
        labels = []
        for row in data:
            poses, landmarks, _ = row["scale"].split("_")
            labels.append(f"{poses[1:]} / {landmarks[1:]}\n{float(row['speedup']):.2f}× speedup")
        ax.set_xticks(positions, labels)
        ax.set_xlabel("Poses / nominal landmarks (speedup = PCG time ÷ Schur time)")
        ax.set_ylabel("Median wall time (ms, logarithmic scale)")
        ax.set_yscale("log")
        ax.set_ylim(top=max(float(row[f"{solver_key}_median_ms"]) for row in data for solver_key in ("pcg", "schur")) * 2)
        ax.set_title(title)
        ax.set_axisbelow(True)
        ax.grid(axis="y", which="major", linestyle="--", alpha=0.45)
        ax.legend(loc="upper left")
    repeats = sorted({row["samples"] for row in rows})
    fig.suptitle(
        f"SBA performance: GeneralizedSchur vs BlockSparsePCG\n"
        f"{hardware} · {'/'.join(repeats)} repeats · structure reused",
        fontsize=15,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=180)
    fig.savefig(args.output.with_suffix(".svg"))
    plt.close(fig)


if __name__ == "__main__":
    main()
