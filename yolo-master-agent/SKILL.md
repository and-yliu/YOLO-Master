---
name: yolo-master-agent
description: Use when the user wants to train, validate, predict, track, export, benchmark, tune, inspect, or orchestrate YOLO-Master / Ultralytics experiments in this repository, including LoRA, MoE, multimodal inference/evaluation, and solutions workflows.
---

# YOLO-Master Agent Skill

## Use This Skill

Use this skill for any repository task that should drive the YOLO-Master stack end-to-end:

- `train`, `val`, `predict`, `track`, `export`, `benchmark`, `tune`
- model inspection and task detection
- LoRA save/load/merge
- MoE diagnose/prune
- multimodal visual inference with OpenAI VLM/LLM cooperation
- multimodal batch evaluation over a dataset or image folder
- `solutions` workflows
- launchers for Gradio / Streamlit

## Execution Rule

First make sure the local Ultralytics framework is installed and the `yolo` CLI is available. Prefer the CLI over raw Python API for supported commands, and use the bundled dispatcher when you want a deterministic, structured run:

```bash
python -m pip install -e .
yolo version
python yolo-master-agent/scripts/run_yolo_master_skill.py --json '{"skill":"yolo.train","inputs":{"model":"yolo11n.pt","data":"coco8.yaml"},"params":{"epochs":1,"imgsz":32}}'
```

On Apple Silicon hosts with PyTorch MPS support, the dispatcher now defaults heavy compute modes such as `train`, `val`, `benchmark`, `predict`, and `track` to `device=mps` when no explicit device is provided. Override with `runtime.device` or `params.device` if needed.

If the CLI run is auto-selected onto MPS/CUDA and fails for a device-level runtime reason, the dispatcher will retry once on CPU and return a structured `recovery` record with the full attempt trail.

When you need fast coverage across many skills or requests, use the AutoTrain-style validator first:

```bash
python yolo-master-agent/scripts/validate_yolo_master_skill.py --suite quick --pretty --summary-only
```

`quick` is the default agent loop. It combines `fast-smoke`, `dry-run`, and `contract` so agents can iterate without waiting on real model inspection or CLI cold-start probes. Use `all` only when you explicitly want the slower full non-manual regression pass.
The case pack now includes multimodal dry-run and contract probes for `yolo.multimodal.infer` and `yolo.multimodal.evaluate`.

For quick regression checks, prefer the tiered suites:

```bash
python yolo-master-agent/scripts/validate_yolo_master_skill.py --suite quick --pretty --summary-only
python yolo-master-agent/scripts/validate_yolo_master_skill.py --suite fast-smoke --pretty --summary-only
python yolo-master-agent/scripts/validate_yolo_master_skill.py --suite cli-smoke --pretty --summary-only
python yolo-master-agent/scripts/validate_yolo_master_skill.py --suite deep-smoke --pretty --summary-only
python yolo-master-agent/scripts/validate_yolo_master_skill.py --suite extended --pretty --summary-only
python yolo-master-agent/scripts/validate_yolo_master_skill.py --suite dry-run --pretty --summary-only
python yolo-master-agent/scripts/validate_yolo_master_skill.py --suite contract --pretty --summary-only
python yolo-master-agent/scripts/validate_yolo_master_skill.py --suite all --pretty --summary-only
```

## Workflow

1. Inspect the request and normalize paths.
2. Install/refresh the local Ultralytics package first when the `yolo` CLI is missing.
3. Prefer `policy.dry_run=true` while validating or evolving the skill surface.
4. On Apple Silicon, let the dispatcher pick `mps` by default for train/val/eval unless the request already sets `device`.
5. Let the dispatcher auto-complete safe runtime defaults such as `workers=0` on macOS train/val paths when the request leaves them unset.
6. Use `yolo` CLI for supported tasks; fall back to Python API only when the CLI does not cover the action.
7. For `predict` and `track`, accept `source` in either `inputs.source` or `params.source`; the dispatcher will normalize it before CLI emission.
8. Pass all task-specific options through `params` unchanged.
9. Return structured artifacts, metrics, evaluation summaries, environment reports, and next actions.
10. For long jobs, use `async`/launcher behavior and write a manifest.

## Multimodal Inference

`yolo.multimodal.infer` is an optional enhancement layer for visual reasoning. It does not replace `yolo.predict`: it runs YOLO first, condenses detections into reasoning evidence, then calls the OpenAI Responses API with `input_text` plus `input_image`, and optionally runs a second LLM refinement pass.

Environment variables:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL` optional
- `OPENAI_API_MODE` optional, `auto`, `responses`, or `chat.completions`
- `OPENAI_VLM_MODEL` optional
- `OPENAI_LLM_MODEL` optional
- `structured_output=true` asks the VLM/LLM to return a strict JSON verdict that can be parsed into `verdict`

Behavior:

- `thinking_with_image=true` attaches the image to the VLM request
- `enable_llm_refine=true` or `OPENAI_LLM_MODEL` enables the refinement pass
- missing `OPENAI_API_KEY` returns a structured `blocked` result
- DashScope/OpenAI-compatible chat endpoints can use `params.openai_api_mode="chat.completions"` with `OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1`
- the manifest now preserves the `multimodal` block, including parsed verdicts when available

Example:

```bash
python yolo-master-agent/scripts/run_yolo_master_skill.py --json '{"skill":"yolo.multimodal.infer","inputs":{"model":"yolo11n.pt","source":"ultralytics/assets/bus.jpg","prompt":"What matters most in this image?"},"params":{"thinking_with_image":true,"vlm_model":"gpt-4.1-mini","llm_model":"gpt-4.1-mini","max_reasoning_items":3,"max_reasoning_boxes":20},"policy":{"dry_run":true}}' --pretty
```

## Multimodal Batch Evaluation

Use `yolo.multimodal.evaluate` when the agent needs to evaluate a real image sample or dataset split with YOLO first, then VLM/LLM cross-checks.

- `inputs.data` selects a dataset YAML such as `coco128.yaml`; `params.split` defaults to `val`
- `inputs.source` may point to a local image folder, image file, or image-list text file
- `params.limit`, `offset`, `stride`, `shuffle`, and `seed` control sampling; `limit=0` means all resolved images
- `params.run_yolo_val=true` also runs a YOLO-only validation baseline when `inputs.data` is available
- Ground-truth labels are read for reporting when available; they are not added to the VLM prompt unless `include_ground_truth_in_prompt=true`

Example:

```bash
python yolo-master-agent/scripts/run_yolo_master_skill.py --json '{"skill":"yolo.multimodal.evaluate","runtime":{"prefer_cli":true,"prefer_mps":true},"inputs":{"model":"yolo11n.pt","data":"coco128.yaml","prompt":"Cross-check detector outputs and summarize obvious false positives, misses, duplicates, and uncertainty."},"params":{"limit":5,"split":"val","imgsz":640,"batch":1,"thinking_with_image":true,"vlm_model":"qwen-vl-plus","llm_model":"qwen-plus","openai_base_url":"https://dashscope.aliyuncs.com/compatible-mode/v1","openai_api_mode":"chat.completions"},"policy":{"dry_run":false}}' --pretty
```

## AutoTrain Loop

Use the bundled validator and case pack to keep this skill honest:

- case pack: `assets/autotrain_cases.json`
- report: `logs/autotrain-report.json`
- bootstrap: `python -m pip install -e .`
- dispatcher supports `policy.dry_run=true` for cheap coverage before real runs
- `yolo` CLI is the preferred execution surface for supported actions
- `quick` is the default iteration suite: `fast-smoke` + `dry-run` + `contract`
- the validator enables a short-lived runtime cache for Torch/MPS detection so repeated subprocess cases do not re-import the stack
- `fast-smoke` protects bootstrap and planning paths with tight timing budgets
- `cli-smoke` validates real `yolo` CLI cold-start execution
- `deep-smoke` holds heavyweight real-model inspection and local `.pt` inference checks
- `all` runs every non-manual case, including slower `cli-smoke` and `deep-smoke`
- `extended-cli` carries slower real CLI validation probes such as mini-dataset `yolo train` and `yolo val` on `mps`, and is marked `manual_only`
- `contract` verifies failure-path behavior and manifest emission
- `contract` now also includes in-process recovery probes so auto device fallback semantics stay covered without adding test-only hooks to the dispatcher
- `contract` includes single-image and batch multimodal stub probes so OpenAI-compatible request shaping, structured verdict parsing, and aggregation stay covered
- CLI failures now carry categorized hints so the agent can recover instead of stopping at a raw traceback.
- Built-in dataset YAML names such as `coco128.yaml` are auto-resolved against the local repository before execution.
- `doctor` returns environment, device selection source, and agent-facing recommendations.
- CLI train/val/predict/benchmark/export responses now carry environment metadata, and auto-selected runs can include a recovery trail when a device fallback occurs.

## Manual Probes

Use these when you want stronger confidence than the default smoke suites without pulling slow jobs into routine validation.

Environment doctor and adaptive install probe:

```bash
python yolo-master-agent/scripts/run_yolo_master_skill.py --json '{"skill":"yolo.system","action":"doctor","params":{"ensure_cli":true}}' --pretty
```

Real CLI training and validation probes on the bundled mini dataset:

```bash
python yolo-master-agent/scripts/validate_yolo_master_skill.py --suite extended --pretty --summary-only
```

Equivalent direct CLI train command:

```bash
yolo train model=scripts/peft_validation/yolo11n.pt data=yolo-master-agent/assets/mini-detect/mini_detect.yaml imgsz=64 epochs=1 batch=1 device=mps workers=0 plots=False verbose=False patience=1 project=runs/agent name=train-mini-mps-manual
```

Equivalent direct CLI val command:

```bash
yolo val model=scripts/peft_validation/yolo11n.pt data=yolo-master-agent/assets/mini-detect/mini_detect.yaml imgsz=16 batch=1 device=mps workers=0 plots=False verbose=False project=runs/agent name=val-mini-mps-manual
```

Structured dispatcher example with automatic MPS selection:

```bash
python yolo-master-agent/scripts/run_yolo_master_skill.py --json '{"skill":"yolo.train","runtime":{"prefer_cli":true,"prefer_mps":true},"inputs":{"model":"scripts/peft_validation/yolo11n.pt","data":"yolo-master-agent/assets/mini-detect/mini_detect.yaml"},"params":{"epochs":1,"imgsz":64,"batch":1,"workers":0,"plots":false,"verbose":false,"patience":1},"artifacts":{},"policy":{"dry_run":false}}' --pretty
```

## References

- Read [`references/skill-architecture.md`](references/skill-architecture.md) for the full architecture map, skill registry, request/response contract, and execution logic.

## Guardrails

- Do not hardcode new CLI strings when a Python API exists.
- Keep `params` as the pass-through layer for new Ultralytics arguments.
- Prefer `yolo` CLI for supported commands; use Python API only as fallback.
- On Apple Silicon, prefer `mps` for training and validation unless the request explicitly overrides the device.
- Consume `evaluation` in addition to `metrics` when judging train/val runs.
- Use `yolo.system doctor` before long runs when the agent needs to confirm install state, selected device, and local repo activation.
- Prefer the `recovery` field over raw stderr when a run auto-falls back from MPS/CUDA to CPU.
- Keep slow real training out of default validator suites; use the manual probe path instead.
- Treat UI launchers and research scripts as launcher-style skills, not plain sync functions.
