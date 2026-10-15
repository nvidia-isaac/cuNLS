#!/bin/bash
# Reproduce the Schur/PCG SBA measurements using an existing Release build.
set -euo pipefail
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "Usage: $0 <build_output_dir> [repetitions=20]"
  exit 1
fi
OUTPUT_DIR=$(realpath "$1")
REPEATS=${2:-20}
if ! [[ "$REPEATS" =~ ^[1-9][0-9]*$ ]]; then
  echo "Repetitions must be a positive integer."
  exit 1
fi
if [ ! -x "$OUTPUT_DIR/build_shared/bin/nls_tests" ]; then
  echo "Missing test binary. Run EXTRA_CMAKE_ARGS=-DBUILD_TESTING=ON ./scripts/build_cunls_in_docker.sh Release $OUTPUT_DIR"
  exit 1
fi
TTY_FLAG=""
[ -t 0 ] && TTY_FLAG="-it"
docker run --runtime=nvidia --gpus all --rm $TTY_FLAG \
  -v "$(pwd):/cunls:ro" -v "$OUTPUT_DIR:/cunls_install" \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  -e CUNLS_BENCHMARK=1 -e CUNLS_BENCHMARK_REPEATS="$REPEATS" \
  -e CUNLS_PCG_TOL=0.001 -e CUNLS_PCG_MAX_ITER=200 \
  cunls:local bash -c '
    set -euo pipefail
    nvidia-smi
    mkdir -p /cunls_install/schur-benchmarks
    for solver in BlockSparsePCG GeneralizedSchur; do
      CUNLS_SOLVER=$solver /cunls_install/build_shared/bin/nls_tests \
        --gtest_filter="*SyntheticSba*" \
        --gtest_output="xml:/cunls_install/schur-benchmarks/$solver.xml" \
        | tee "/cunls_install/schur-benchmarks/$solver.log"
    done
  '

# Standard-library-only summary; preserve raw measurements beside the CSV.
docker run --rm -i --entrypoint python3 \
  -v "$OUTPUT_DIR:/cunls_install" cunls:local - /cunls_install/schur-benchmarks <<'PY'
import csv
import math
from pathlib import Path
import statistics
import sys

root = Path(sys.argv[1])
records = {}
for solver in ("BlockSparsePCG", "GeneralizedSchur"):
    rows = []
    for line in (root / f"{solver}.log").read_text().splitlines():
        if line.startswith("SBA_"):
            parts = line.split()
            rows.append((parts[0], dict(part.split("=", 1) for part in parts[1:])))
    records[solver] = rows
scales = [record["scale"] for kind, record in records["BlockSparsePCG"] if kind == "SBA_SHAPE"]
assert scales, "No benchmark records found"
devices = []
for solver, rows in records.items():
    found = [record for kind, record in rows if kind == "SBA_DEVICE"]
    assert found and all(record == found[0] for record in found), (solver, "Missing or inconsistent GPU identity")
    devices.append(found[0])
assert devices[0] == devices[1], "Backends ran on different GPUs"
device = devices[0]
print(f"CUDA device: {device['name']} (SM {device['sm']}, PCI bus {device['pci_bus']})")
summary = []
for scale in scales:
    for stage in ("SBA_LINEAR", "SBA_WARM"):
        stats = {}
        for solver, rows in records.items():
            samples = sorted(float(record["ms"]) for kind, record in rows
                             if kind == stage and record["scale"] == scale)
            assert samples, (solver, scale, stage)
            stats[solver] = (statistics.median(samples), samples[math.ceil(.95 * len(samples))-1], len(samples))
        pcg, schur = stats["BlockSparsePCG"], stats["GeneralizedSchur"]
        summary.append(dict(scale=scale, stage=stage, samples=pcg[2],
                            device=device["name"], compute_capability=device["sm"],
                            pcg_median_ms=pcg[0], schur_median_ms=schur[0],
                            pcg_p95_ms=pcg[1], schur_p95_ms=schur[1], speedup=pcg[0]/schur[0]))
        assert pcg[2] == schur[2]
    final_costs = {}
    shapes = []
    for solver, rows in records.items():
        shape = next(record for kind, record in rows if kind == "SBA_SHAPE" and record["scale"] == scale)
        shapes.append(shape)
        residual = next(float(record["relative"]) for kind, record in rows if kind == "SBA_RESIDUAL" and record["scale"] == scale)
        assert math.isfinite(residual) and residual <= 1.05e-3, (solver, scale, residual)
        final_costs[solver] = statistics.median(float(record["final"]) for kind, record in rows
                                               if kind == "SBA_WARM" and record["scale"] == scale)
    assert shapes[0] == shapes[1], "Solvers must use identical input sizes"
    pcg_cost, sc_cost = final_costs["BlockSparsePCG"], final_costs["GeneralizedSchur"]
    assert abs(pcg_cost-sc_cost) <= max(1e-7, .001*abs(pcg_cost)), (scale, final_costs)
with (root / "summary.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=summary[0].keys())
    writer.writeheader()
    writer.writerows(summary)
for row in summary:
    print(f"{row['scale']:20} {row['stage']:11} PCG {row['pcg_median_ms']:9.3f} ms  "
          f"Schur {row['schur_median_ms']:9.3f} ms  {row['speedup']:5.2f}x")
print(f"Raw logs, JUnit results, and summary: {root}")
PY
