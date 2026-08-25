#!/usr/bin/env python
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Run the repository's own config-driven Foundation distillation path and check five things about it.

``p0_smoke.py`` wires the same components (tap, projector, KD loss) by hand on a synthetic batch. This runs the
path the trainer actually builds from config, so what gets checked is the wiring in
:mod:`ultralytics.engine.trainer`, not a hand-made replica of it.

Usage
    python experiments/d2/p0_path_check.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.foundation_distill_model import FoundationDistillationModel
from ultralytics.utils.torch_utils import unwrap_model

RESULTS = Path(__file__).resolve().parent / "results" / "p0_path_check.json"


def main() -> int:
    """Train briefly with Foundation KD on, then report whether the KD term really reached the optimizer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="coco8.yaml")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    trainer = DetectionTrainer(
        overrides={
            "model": "ultralytics/cfg/models/26/yolo26-master-n.yaml",
            "data": args.data,
            "epochs": args.epochs,
            "imgsz": 256,
            "batch": 4,
            "workers": 0,
            "device": args.device,
            "seed": 17,
            "amp": False,
            "plots": False,
            "project": "runs/d2/p0",
            "name": "path_check",
            "exist_ok": True,
            "foundation_enabled": True,
            "foundation_teacher": "dinov3",
            "foundation_model": "facebook/dinov3-vits16-pretrain-lvd1689m",
            "foundation_loss_weight": 0.05,
        }
    )

    seen: dict = {"setup": {}, "kd": []}

    def on_start(t) -> None:
        """Record what the trainer built, once it has finished building it."""
        model = unwrap_model(t.model)
        wrapped = isinstance(model, FoundationDistillationModel)
        teacher_params = list(model.teacher_manager.parameters()) if wrapped else []
        optimized = {id(p) for group in t.optimizer.param_groups for p in group["params"]}
        seen["setup"] = {
            "model_type": type(model).__name__,
            "wrapped": wrapped,
            "loss_names": list(t.loss_names),
            "teacher_in_optimizer": any(id(p) in optimized for p in teacher_params),
            "resolved_loss": str(getattr(t.args, "foundation_loss", None)),
            "align_dim": int(getattr(t.args, "foundation_align_dim", 0)),
        }

    def on_batch(t) -> None:
        """Record the KD scalar the wrapper reported for this step."""
        metrics = getattr(t, "foundation_metric_latest", None)
        if metrics:
            seen["kd"].append(float(metrics["foundation_loss"]))

    trainer.add_callback("on_train_start", on_start)
    trainer.add_callback("on_train_batch_end", on_batch)
    trainer.train()

    setup, kd = seen["setup"], seen["kd"]
    csv_text = Path(trainer.csv).read_text(encoding="utf-8") if Path(trainer.csv).exists() else ""
    csv_header = csv_text.splitlines()[0] if csv_text else ""

    checks = {
        "wrapper_installed": setup.get("wrapped", False),
        "teacher_absent_from_optimizer": setup.get("teacher_in_optimizer") is False,
        "foundation_in_loss_names": "foundation" in setup.get("loss_names", []),
        "kd_is_nonzero_and_finite": bool(kd) and all(v == v and v != 0 for v in kd),
        "kd_reaches_results_csv": "foundation" in csv_header.lower(),
    }

    print()
    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    verdict = all(checks.values())
    print(f"\nverdict: {'PASS' if verdict else 'FAIL'}")
    print(f"loss={setup.get('resolved_loss')}  align_dim={setup.get('align_dim')}")
    if kd:
        print(f"kd  {kd[0]:.6f} -> {kd[-1]:.6f}  over {len(kd)} steps")

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(
        json.dumps(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "verdict": "pass" if verdict else "fail",
                "checks": checks,
                "setup": setup,
                "kd_history": kd,
                "results_csv": str(trainer.csv),
                "claim": "path_integrity_only_no_accuracy_claim",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nevidence -> {RESULTS}")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
