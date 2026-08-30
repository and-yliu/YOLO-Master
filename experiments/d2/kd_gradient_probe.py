"""Measure the gradient a Foundation KD loss actually delivers to the student.

Motivation: ``foundation_task_ratio`` compares loss *values*, and §5.5's weight sweep showed
a 16x weight range moving neither the detection loss nor the KD loss itself. This probe
measures the quantity that governs optimization instead -- gradient magnitude -- and reports
the weight that would put KD at a target share of the detection gradient.

Two modes:

``--mode kd``
    KD-side only. Pure torch, no teacher weights, no GPU, no ultralytics import. Reports how
    each ``foundation_loss`` form and each ``foundation_relation_samples`` value scales the
    gradient reaching the student feature. Runs anywhere.

``--mode ratio``
    Probe A from ``kd_gradient_analysis.md`` §5.2. Needs a real training setup (teacher weights
    + dataset). Measures ||d L_kd / d theta|| against ||d L_task / d theta|| over real batches
    and solves for ``w* = target_ratio * ||g_task|| / ||g_kd||``.

Examples:
    $ python experiments/d2/kd_gradient_probe.py --mode kd
    $ python experiments/d2/kd_gradient_probe.py --mode ratio --batches 50 --target-ratio 0.1
"""

from __future__ import annotations

import argparse
import json

import torch

from ultralytics.nn.foundation.losses import cosine_kd_loss, relational_kd_loss

# wsweep actuals: batch=4, foundation_align_dim=256, imgsz=256 -> P4 grid 16x16.
DEFAULT_SHAPE = (4, 256, 16, 16)


def _l2_kd_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """Mirror the ``l2`` branch of ``FoundationDistillationModel._kd_components_with_weights``."""
    return (student.float() - teacher.detach().float()).square().mean(dim=1).mean()


def _grad_norm(loss_fn, shape: tuple[int, int, int, int], seed: int) -> tuple[float, float]:
    """Return ``(loss_value, ||d loss / d student||_2)`` for one synthetic feature pair.

    Args:
        loss_fn (Callable): Maps ``(student, teacher)`` to a scalar loss.
        shape (tuple[int, int, int, int]): Student/teacher feature shape in ``(B, C, H, W)``.
        seed (int): Manual seed, so repeated calls are comparable.

    Returns:
        (tuple[float, float]): The loss value and the gradient norm at the student feature.

    Examples:
        >>> value, norm = _grad_norm(lambda s, t: (s - t).square().mean(), (2, 4, 3, 3), seed=0)
        >>> norm > 0
        True
    """
    torch.manual_seed(seed)
    student = torch.randn(*shape, requires_grad=True)
    teacher = torch.randn(*shape)
    loss = loss_fn(student, teacher)
    loss.backward()
    return float(loss.item()), float(student.grad.norm().item())


def probe_kd_side(shape: tuple[int, int, int, int] = DEFAULT_SHAPE, seed: int = 0) -> dict:
    """Compare the gradient scale of every config-selectable ``foundation_loss`` form.

    The numbers characterize each loss form's *reduction structure* (how many terms it averages
    over, whether its gradient is sign-only) under matched inputs. They are magnitudes for
    ranking, not calibrated values for a real feature distribution.

    Args:
        shape (tuple[int, int, int, int]): Feature shape in ``(B, C, H, W)``.
        seed (int): Manual seed shared by every measurement.

    Returns:
        (dict): Nested results for loss forms, relation-sample counts, and mismatch sensitivity.

    Examples:
        >>> out = probe_kd_side(shape=(2, 8, 4, 4))
        >>> sorted(out)
        ['loss_forms', 'mismatch_sensitivity', 'relation_samples']
    """
    forms = {
        "relational": lambda s, t: relational_kd_loss(s, t),
        "cosine": cosine_kd_loss,
        "l2": _l2_kd_loss,
        "hybrid": lambda s, t: cosine_kd_loss(s, t) + relational_kd_loss(s, t),
    }
    loss_forms = {name: dict(zip(("value", "grad_norm"), _grad_norm(fn, shape, seed))) for name, fn in forms.items()}

    samples = {}
    for count in (16, 32, 64, 128, 256):
        value, norm = _grad_norm(lambda s, t, n=count: relational_kd_loss(s, t, num_samples=n), shape, seed)
        samples[count] = {"value": value, "grad_norm": norm}

    # L1 on the Gram makes the gradient sign-only: being far from the teacher barely changes it.
    mismatch = {}
    for scale in (0.1, 0.5, 1.0, 4.0):
        torch.manual_seed(seed)
        student = torch.randn(*shape, requires_grad=True)
        teacher = student.detach() + scale * torch.randn(*shape)
        loss = relational_kd_loss(student, teacher)
        loss.backward()
        mismatch[scale] = {"value": float(loss.item()), "grad_norm": float(student.grad.norm().item())}

    return {"loss_forms": loss_forms, "relation_samples": samples, "mismatch_sensitivity": mismatch}


def probe_gradient_ratio(args: argparse.Namespace) -> dict:
    """Measure ``||g_kd|| / ||g_task||`` on real batches and solve for the balancing weight.

    Builds the same Foundation distillation wrapper the trainer builds, then for each batch takes
    two separate backward passes over the shared trunk -- one from the KD term, one from the
    detection term -- and accumulates their gradient norms over the parameters feeding the tap.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.

    Returns:
        (dict): Mean gradient norms and the solved ``w*`` at each target ratio.
    """
    from ultralytics.cfg import get_cfg
    from ultralytics.models.yolo.detect import DetectionTrainer

    overrides = {
        "model": args.model,
        "data": args.data,
        "epochs": 1,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": 0,
        "device": args.device,
        "pretrained": False,
        "amp": False,
        "plots": False,
        "optimizer": "SGD",
        "lr0": 0.01,
        "foundation_enabled": True,
        "foundation_teacher": args.teacher,
        "foundation_model": args.teacher_model,
        # Measure the unweighted KD gradient; w is what we are solving for.
        "foundation_loss_weight": 1.0,
    }
    trainer = DetectionTrainer(overrides=overrides)
    trainer._setup_train(world_size=1)
    model = trainer.model
    model.train()

    # Parameters upstream of the tap are what both losses compete over.
    trunk = [p for p in model.parameters() if p.requires_grad]

    def _snapshot() -> list[torch.Tensor | None]:
        """Copy the current gradient of every trunk parameter."""
        return [None if p.grad is None else p.grad.detach().clone() for p in trunk]

    def _norm(grads: list[torch.Tensor | None]) -> float:
        return float(torch.sqrt(sum((g**2).sum() for g in grads if g is not None)).item())

    def _dot(a: list[torch.Tensor | None], b: list[torch.Tensor | None]) -> float:
        return float(sum((x * y).sum() for x, y in zip(a, b) if x is not None and y is not None).item())

    kd_norms: list[float] = []
    task_norms: list[float] = []
    cosines: list[float] = []
    for index, batch in enumerate(trainer.train_loader):
        if index >= args.batches:
            break
        batch = trainer.preprocess_batch(batch)
        captured = {}
        for which in ("kd", "task"):
            model.zero_grad(set_to_none=True)
            loss, _ = model(batch)
            # loss is cat([task_items..., foundation]); the trainer backwards loss.sum().
            component = loss[-1] if which == "kd" else loss[:-1].sum()
            component.backward()
            captured[which] = _snapshot()
        kd_norm, task_norm = _norm(captured["kd"]), _norm(captured["task"])
        kd_norms.append(kd_norm)
        task_norms.append(task_norm)
        # Negative cosine means the KD term is pulling against detection, not merely weakly:
        # a different problem from "too small", with the opposite remedy.
        cosines.append(_dot(captured["kd"], captured["task"]) / max(kd_norm * task_norm, 1e-12))

    mean_kd = sum(kd_norms) / max(len(kd_norms), 1)
    mean_task = sum(task_norms) / max(len(task_norms), 1)
    mean_cos = sum(cosines) / max(len(cosines), 1)
    ratios = sorted({args.target_ratio, 0.05, 0.1, 0.2})
    return {
        "batches": len(kd_norms),
        "grad_norm_kd_at_w1": mean_kd,
        "grad_norm_task": mean_task,
        "observed_ratio_at_w1": mean_kd / max(mean_task, 1e-12),
        "w_star": {r: r * mean_task / max(mean_kd, 1e-12) for r in ratios},
        "target_ratio": args.target_ratio,
        # Gradient-conflict diagnostic: see kd_literature.md section 5.
        "grad_cosine_mean": mean_cos,
        "grad_conflict_fraction": sum(1 for c in cosines if c < 0) / max(len(cosines), 1),
    }


def main() -> None:
    """Parse arguments and print the requested probe as JSON."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("kd", "ratio"), default="kd", help="kd: offline loss-form scales")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batches", type=int, default=50, help="ratio mode: batches to average over")
    parser.add_argument("--target-ratio", type=float, default=0.1, help="ratio mode: desired ||g_kd||/||g_task||")
    parser.add_argument("--model", default="ultralytics/cfg/models/26/yolo26-master-n.yaml")
    parser.add_argument("--data", default="coco128.yaml")
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--teacher", default="dinov3")
    parser.add_argument("--teacher-model", default="facebook/dinov3-vits16-pretrain-lvd1689m")
    args = parser.parse_args()

    result = probe_kd_side(seed=args.seed) if args.mode == "kd" else probe_gradient_ratio(args)
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
