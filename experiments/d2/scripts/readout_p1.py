#!/usr/bin/env python
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Apply the pre-registered D2 P1 decision rule to a finished run matrix.

The decision rule was submitted in `docs/design.md` 2.2 before any mAP existed, and it is deliberately not a
"which number is bigger" comparison: the statistical unit is the per-seed paired difference `on - off`, and a
cell is called no-go only when the effect is both small and indistinguishable from zero. Doing this by eye
invites moving the line after seeing the data, so the rule lives in code and reads the runs directly.

    |mean paired delta| < 0.003  AND  95% CI contains 0   ->   no-go

Anything else is *not* a go: with three seeds the interval is wide, and a cell whose mean clears 0.003 while
its interval still spans zero is inconclusive. Per the protocol the response to an inconclusive cell is more
seeds, never a wider threshold.

Usage
    python experiments/d2/scripts/readout_p1.py --runs runs/detect/d2/p1voc
    python experiments/d2/scripts/readout_p1.py --runs runs/detect/d2/p1voc --cells a,b,c --metric tail5
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path

# Two-sided 97.5th percentile of Student's t, indexed by degrees of freedom. Small-sample only: this readout is
# meaningless at the sample sizes where a normal approximation would do.
T_CRITICAL_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}

# design.md 2.2. The metric is Ultralytics' 0-1 scale, so this is 0.3 percentage points, not 0.3.
NULL_THRESHOLD = 0.003

MAP_COLUMN = "metrics/mAP50-95(B)"


def read_map_series(run_dir: Path) -> list[float]:
    """Return the per-epoch mAP50-95 series of one finished run.

    Args:
        run_dir (Path): Directory holding the run's `results.csv`.

    Returns:
        (list[float]): mAP50-95 at each logged epoch, in order.
    """
    rows = list(csv.DictReader((run_dir / "results.csv").open(encoding="utf-8")))
    if not rows:
        raise ValueError(f"{run_dir}/results.csv has no rows")
    return [float(row[MAP_COLUMN]) for row in rows]


def summarize(series: list[float], metric: str) -> float:
    """Reduce a per-epoch mAP series to the single number the readout compares.

    `tail5` averages the last five epochs, which is preferred over a single final epoch because it is less
    sensitive to per-epoch validation noise without peeking at a maximum the way `best` does.

    Args:
        series (list[float]): Per-epoch mAP50-95 values.
        metric (str): One of `last`, `best`, `tail5`.

    Returns:
        (float): The reduced value.

    Examples:
        >>> summarize([0.1, 0.2, 0.3], "last")
        0.3
        >>> summarize([0.1, 0.4, 0.3], "best")
        0.4
        >>> round(summarize([0.1, 0.2, 0.3], "tail5"), 4)
        0.2
    """
    if metric == "last":
        return series[-1]
    if metric == "best":
        return max(series)
    if metric == "tail5":
        tail = series[-5:]
        return sum(tail) / len(tail)
    raise ValueError(f"unknown metric {metric!r}")


def paired_stats(diffs: list[float]) -> dict[str, float]:
    """Compute the mean, spread and 95% t interval of a set of paired differences.

    Args:
        diffs (list[float]): Per-seed `on - off` differences. At least two are required for an interval.

    Returns:
        (dict[str, float]): Keys `mean`, `sd`, `lo`, `hi`, `t`, `n`.

    Examples:
        >>> stats = paired_stats([0.01, 0.02, 0.03])
        >>> round(stats["mean"], 4)
        0.02
        >>> stats["lo"] < 0.02 < stats["hi"]
        True
    """
    n = len(diffs)
    if n < 2:
        raise ValueError("a paired interval needs at least two seeds")
    if n - 1 not in T_CRITICAL_95:
        raise ValueError(f"no tabulated t critical value for df={n - 1}")
    mean = statistics.mean(diffs)
    sd = statistics.stdev(diffs)
    stderr = sd / math.sqrt(n)
    tcrit = T_CRITICAL_95[n - 1]
    return {
        "mean": mean,
        "sd": sd,
        "lo": mean - tcrit * stderr,
        "hi": mean + tcrit * stderr,
        "t": mean / stderr if stderr else math.nan,
        "n": float(n),
    }


def verdict(stats: dict[str, float], threshold: float = NULL_THRESHOLD) -> str:
    """Apply the pre-registered decision rule to one cell's paired statistics.

    Args:
        stats (dict[str, float]): Output of [paired_stats][].
        threshold (float): The pre-registered null band half-width, on the 0-1 mAP scale.

    Returns:
        (str): `no-go`, `inconclusive` or `effect`.

    Examples:
        >>> verdict({"mean": 0.0005, "lo": -0.002, "hi": 0.003})
        'no-go'
        >>> verdict({"mean": 0.005, "lo": -0.001, "hi": 0.011})
        'inconclusive'
        >>> verdict({"mean": 0.005, "lo": 0.002, "hi": 0.008})
        'effect'
    """
    spans_zero = stats["lo"] < 0.0 < stats["hi"]
    if not spans_zero:
        return "effect"
    return "no-go" if abs(stats["mean"]) < threshold else "inconclusive"


def seeds_needed(stats: dict[str, float], threshold: float = NULL_THRESHOLD) -> int | None:
    """Estimate the seed count at which an inconclusive cell's interval would exclude zero.

    Assumes the observed mean and standard deviation hold as more seeds are added, which is exactly the
    assumption three seeds cannot support -- read the result as a planning number, not a promise.

    Args:
        stats (dict[str, float]): Output of [paired_stats][].
        threshold (float): The pre-registered null band half-width, unused except to skip settled cells.

    Returns:
        (int | None): Smallest tabulated seed count whose interval would exclude zero, or None if unreachable.

    Examples:
        >>> seeds_needed({"mean": 0.00468, "sd": 0.00238, "lo": -0.00124, "hi": 0.0106, "n": 3.0})
        4
    """
    if stats["sd"] == 0:
        return int(stats["n"])
    for n in range(int(stats["n"]), max(T_CRITICAL_95) + 2):
        if n - 1 not in T_CRITICAL_95:
            break
        if abs(stats["mean"]) - T_CRITICAL_95[n - 1] * stats["sd"] / math.sqrt(n) > 0:
            return n
    return None


def main(argv: list[str] | None = None) -> int:
    """Print the paired readout for every cell against the shared off control.

    Args:
        argv (list[str] | None): Command line arguments, defaulting to `sys.argv[1:]`.

    Returns:
        (int): Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, required=True, help="directory holding <cell>-s<seed>/ run dirs")
    parser.add_argument("--cells", default="a,b,c", help="comma separated treatment cells")
    parser.add_argument("--control", default="off", help="shared control cell")
    parser.add_argument("--seeds", default="17,29,43", help="comma separated seeds")
    parser.add_argument("--metric", default="tail5", choices=("last", "best", "tail5"))
    args = parser.parse_args(argv)

    seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]

    def value(cell: str, seed: str) -> float:
        return summarize(read_map_series(args.runs / f"{cell}-s{seed}"), args.metric)

    control = {seed: value(args.control, seed) for seed in seeds}
    print(f"metric={args.metric}  threshold=±{NULL_THRESHOLD}  control={args.control}  n={len(seeds)} seeds\n")
    print(f"{'cell':6} {'mean':>9} {'sd':>9} {'95% CI':>21} {'t':>7}  verdict")
    for cell in cells:
        diffs = [value(cell, seed) - control[seed] for seed in seeds]
        stats = paired_stats(diffs)
        call = verdict(stats)
        interval = f"[{stats['lo']:+.5f},{stats['hi']:+.5f}]"
        line = f"{cell:6} {stats['mean']:+9.5f} {stats['sd']:9.5f} {interval:>21} {stats['t']:+7.2f}  {call}"
        if call == "inconclusive":
            need = seeds_needed(stats)
            line += f" (needs ~{need} seeds)" if need else " (unreachable at tabulated n)"
        print(line)
        print(f"       per-seed: {[round(d, 5) for d in diffs]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
