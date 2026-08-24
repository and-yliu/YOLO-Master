#!/usr/bin/env python
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""D2 P0 admission gate: DINOv3 stage alignment and distillation-loss descent on a fixed batch.

Usage
    python experiments/d2/p0_smoke.py
    python experiments/d2/p0_smoke.py --steps 100 --imgsz 640
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ultralytics.nn.foundation import (
    DINOv3Teacher,
    P4AlignmentProjector,
    StudentFeatureTap,
    cosine_kd_loss,
)
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG

DEFAULT_RESULTS = Path(__file__).resolve().parent / "results" / "p0_smoke_dinov3.json"


def synthetic_batch(batch_size: int, imgsz: int, device: torch.device, boxes_per_image: int = 3) -> dict:
    """Build a fixed, labelled detection batch so the real task loss can be computed alongside the KD term."""
    total = batch_size * boxes_per_image
    # Normalized xywh, inset so clipping cannot degenerate a box.
    centers = torch.rand(total, 2, device=device) * 0.5 + 0.25
    sizes = torch.rand(total, 2, device=device) * 0.2 + 0.1
    return {
        "img": torch.rand(batch_size, 3, imgsz, imgsz, device=device),
        "cls": torch.randint(0, 80, (total, 1), device=device).float(),
        "bboxes": torch.cat([centers, sizes], dim=1),
        "batch_idx": torch.arange(batch_size, device=device).repeat_interleave(boxes_per_image).float(),
    }


def main() -> int:
    """Run the P0 admission gate and write a JSON verdict."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", default="ultralytics/cfg/models/26/yolo26-master-n.yaml")
    parser.add_argument("--teacher", default="facebook/dinov3-vits16-pretrain-lvd1689m")
    parser.add_argument("--level", default="p4", choices=["p3", "p4", "p5"])
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--align-dim", type=int, default=64)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--boxes", type=int, default=3, help="ground-truth boxes per synthetic image")
    parser.add_argument("--kd-weight", type=float, default=0.05, help="weight of the KD term in the total loss")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--results", default=str(DEFAULT_RESULTS))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # ------------------------------------------------------------------ setup

    # The tap hooks the layer feeding the Detect head at `level`, keeping gradients so KD can flow back.
    student = DetectionModel(args.student, ch=3, nc=80, verbose=False).to(device)
    student.args = DEFAULT_CFG
    tap = StudentFeatureTap(student, target=args.level)
    teacher = DINOv3Teacher(args.teacher, device=args.device, dtype="fp32")

    # Teacher is frozen, so the target is constant: encode once and reuse for every step.
    batch = synthetic_batch(args.batch, args.imgsz, device, boxes_per_image=args.boxes)
    target = teacher.encode(batch["img"]).dense["p4"]

    # One throwaway forward, only to learn the student's channel count before sizing the projector.
    student.train()
    student(batch["img"])
    student_feat = tap.feature

    # Student side trainable, teacher side frozen. Both land in a shared `align_dim` space.
    projector = P4AlignmentProjector(
        student_channels=int(student_feat.shape[1]),
        teacher_channels=int(target.shape[1]),
        align_dim=args.align_dim,
    ).to(device)

    print(f"\nteacher   : {args.teacher}  {tuple(target.shape)}")
    print(f"student   : {args.student}  {args.level} @ layer {tap.source_index}  {tuple(student_feat.shape)}")
    print(f"projector : {student_feat.shape[1]} / {target.shape[1]} -> align_dim {args.align_dim}\n")

    # Teacher params are deliberately absent here; `teacher_ids` lets us assert that afterwards.
    params = list(student.parameters()) + [p for p in projector.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    teacher_ids = {id(p) for p in teacher.parameters()}

    kd_history: list[float] = []
    task_history: list[float] = []
    total_history: list[float] = []
    student_grad = projector_grad = kd_reaches_student = False

    # --------------------------------------------------- optimize a fixed batch

    for step in range(args.steps):
        tap.clear()

        # Forward once; `student.loss` runs the model, and the tap catches P4 on the way through.
        task_loss = student.loss(batch)[0].sum()
        s_aligned, t_aligned = projector(tap.feature, target)
        kd_loss = cosine_kd_loss(s_aligned, t_aligned)
        total_loss = task_loss + args.kd_weight * kd_loss

        optimizer.zero_grad()

        if step == 0:
            # Backward KD by itself first, while the graph is alive. All-zero student grads here would mean
            # the task loss is the only thing touching the student, i.e. KD is decorative.
            (args.kd_weight * kd_loss).backward(retain_graph=True)
            kd_reaches_student = any(p.grad is not None and p.grad.abs().sum() > 0 for p in student.parameters())
            optimizer.zero_grad()

        total_loss.backward()

        if step == 0:
            student_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in student.parameters())
            projector_grad = any(
                p.grad is not None and p.grad.abs().sum() > 0 for p in projector.student_proj.parameters()
            )

        optimizer.step()

        kd_history.append(float(kd_loss.item()))
        task_history.append(float(task_loss.item()))
        total_history.append(float(total_loss.item()))
        if step % max(1, args.steps // 10) == 0 or step == args.steps - 1:
            print(
                f"  step {step:4d}  total {total_loss.item():10.4f}  "
                f"task {task_loss.item():10.4f}  kd {kd_loss.item():.6f}"
            )

    tap.close()

    # ---------------------------------------------------------------- verdict

    # patch-16 teacher on a stride-16 level: grids coincide, no interpolation. False means misconfigured.
    native_grid_match = tuple(target.shape[-2:]) == tuple(student_feat.shape[-2:])
    every_loss = kd_history + task_history + total_history

    # Every entry must be True to pass. Grouped by what kind of failure it catches.
    checks = {
        # Alignment -- the student's grid is the one that must survive.
        "projector_preserves_student_grid": tuple(s_aligned.shape[-2:]) == tuple(student_feat.shape[-2:]),
        "teacher_grid_matches_student_natively": native_grid_match,

        # Numerics -- catches NaN/Inf and a silently disabled KD branch.
        "all_losses_finite": all(torch.isfinite(torch.tensor(v)) for v in every_loss),
        "kd_term_is_nonzero": kd_history[0] > 0,

        # Composition -- KD is optimized, not just logged beside the task loss.
        "kd_term_enters_total_loss": abs(total_history[0] - (task_history[0] + args.kd_weight * kd_history[0])) < 1e-4,
        "kd_gradient_reaches_student": kd_reaches_student,
        "projector_receives_gradient": projector_grad,
        "student_receives_gradient": student_grad,

        # Teacher stays inert -- frozen, unoptimized, and not learning through its own projection.
        "teacher_params_frozen": all(not p.requires_grad for p in teacher.parameters()),
        "teacher_absent_from_optimizer": not any(
            id(p) in teacher_ids for group in optimizer.param_groups for p in group["params"]
        ),
        "teacher_projection_frozen": projector.teacher_projection_frozen,
        
        # Optimizable -- only KD must descend; total is noisy under dynamic assignment and BatchNorm.
        "kd_loss_descends_on_fixed_batch": kd_history[-1] < kd_history[0],
        "total_loss_improves_at_least_once": min(total_history) < total_history[0],
    }
    passed = all(checks.values())

    print(f"\nkd   {kd_history[0]:.6f} -> {kd_history[-1]:.6f}  ({kd_history[0] / max(kd_history[-1], 1e-12):.1f}x)")
    print(f"task {task_history[0]:.4f} -> {task_history[-1]:.4f}   (min {min(task_history):.4f})")
    for name, value in checks.items():
        print(f"  {'PASS' if value else 'FAIL'}  {name}")
    print(f"\nP0 admission gate: {'PASS' if passed else 'FAIL'}\n")

    # --------------------------------------------------------------- evidence

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "verdict": "pass" if passed else "fail",
        "checks": checks,
        "teacher": {"model_id": args.teacher, "metadata": teacher.encode(batch["img"][:1]).metadata},
        "student": {"cfg": args.student, "level": args.level, "source_layer": tap.source_index},
        "shapes": {"teacher": list(target.shape), "student": list(student_feat.shape)},
        "alignment": {"native_grid_match": native_grid_match, **projector.alignment},
        "config": vars(args),
        "loss_history": {"kd": kd_history, "task": task_history, "total": total_history},
        "env": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        "claim": "plumbing_only_no_accuracy_claim",
    }
    out = Path(args.results)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    print(f"evidence written to {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}\n")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
