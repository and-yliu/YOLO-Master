#!/usr/bin/env python
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Enforce the D2 no-confound constraint across the P1 2x2 matrix.

A same-budget comparison is worth nothing if some unrelated field drifted between two runs, and eyeballing five
YAML files is not a check. This asserts mechanically that every P1 config and every matrix row agrees on the
shared budget, so "no confounding variables" is verified rather than claimed.

Scope note: this checks the *shared* invariant -- everything outside the Foundation axis is identical everywhere.
It does not yet check the per-comparison axis (when comparing teachers, multiscale must match; when comparing
scales, the teacher must match). That axis-aware check is deferred; see design.md 7.2.

Usage
    python experiments/d2/scripts/validate_pair.py
    python experiments/d2/scripts/validate_pair.py --configs experiments/d2/configs --matrix experiments/d2/experiment_matrix.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent  # experiments/d2, one level up from scripts/

# Config fields a cell is allowed to own. Everything else -- budget, optimizer, data, seed, determinism --
# must be identical across all five configs or the 2x2 is void.
ALLOWED_CONFIG_PREFIXES = ("foundation_",)
ALLOWED_CONFIG_KEYS = {"name"}

# Matrix columns a run is allowed to own: its identity, its seed, and its position on the Foundation axes.
ALLOWED_MATRIX_KEYS = {
    "run_id",
    "cell",
    "config",
    "seed",
    "foundation_enabled",
    "teacher",
    "teacher_model",
    "teacher_revision",
    "levels",
    "multiscale",
    "loss_weight",
    "native_grid_match",
    "axis_note",
    # Bookkeeping, not budget: records whether a row has been run, and is expected to diverge as the matrix is
    # executed or as a cell is dropped. Holding it constant would force every finished matrix to fail.
    "status",
}


def shared_fields(config_path: Path) -> dict[str, str]:
    """Return a config's non-Foundation scalar fields, keyed by name.

    Parsed line-wise rather than with a YAML loader so the check runs in any environment, including one without
    PyYAML installed.

    Args:
        config_path (Path): Path to a P1 arm config.

    Returns:
        (dict[str, str]): Field name to raw value, excluding comments, list items and allowed-to-differ keys.

    Examples:
        >>> sorted(ALLOWED_CONFIG_KEYS)
        ['name']
    """
    fields: dict[str, str] = {}
    for line in config_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "- ")):
            continue
        key, separator, value = stripped.partition(":")
        if not separator:
            continue
        key = key.strip()
        if key in ALLOWED_CONFIG_KEYS or key.startswith(ALLOWED_CONFIG_PREFIXES):
            continue
        fields[key] = value.split("#")[0].strip()
    return fields


def check_configs(config_dir: Path) -> list[str]:
    """Report P1 configs whose shared budget differs from the first config's."""
    configs = sorted(config_dir.glob("p1_*.yaml"))
    if len(configs) < 2:
        return [f"expected at least 2 P1 configs in {config_dir}, found {len(configs)}"]
    reference, *others = configs
    baseline = shared_fields(reference)
    problems = []
    for config in others:
        current = shared_fields(config)
        offending = sorted(key for key in set(baseline) | set(current) if baseline.get(key) != current.get(key))
        if offending:
            problems.append(f"{config.name} differs from {reference.name} outside the Foundation axis in {offending}")
    return problems


def check_matrix(matrix_path: Path) -> list[str]:
    """Report matrix columns that vary across runs but are not on the allowlist."""
    rows = list(csv.DictReader(matrix_path.open(encoding="utf-8")))
    if not rows:
        return [f"{matrix_path.name} has no runs"]
    problems = []
    for column in rows[0]:
        if column in ALLOWED_MATRIX_KEYS:
            continue
        values = {row[column] for row in rows}
        if len(values) > 1:
            problems.append(f"budget column {column!r} is not constant across runs: {sorted(values)}")
    run_ids = [row["run_id"] for row in rows]
    duplicates = sorted({run_id for run_id in run_ids if run_ids.count(run_id) > 1})
    if duplicates:
        problems.append(f"duplicate run_id: {duplicates}")
    return problems


def main() -> int:
    """Validate the P1 configs and matrix, and report a single verdict."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", default=str(HERE / "configs"))
    parser.add_argument("--matrix", default=str(HERE / "experiment_matrix.csv"))
    args = parser.parse_args()

    config_problems = check_configs(Path(args.configs))
    matrix_problems = check_matrix(Path(args.matrix))

    for problem in config_problems:
        print(f"FAIL  config {problem}")
    if not config_problems:
        print("PASS  every P1 config shares an identical budget")

    for problem in matrix_problems:
        print(f"FAIL  matrix {problem}")
    if not matrix_problems:
        print("PASS  every matrix run shares an identical budget")

    ok = not config_problems and not matrix_problems
    print(f"\nno-confound check: {'PASS' if ok else 'FAIL'}")
    print("note: per-comparison axis check not implemented yet (design.md 7.2)\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
