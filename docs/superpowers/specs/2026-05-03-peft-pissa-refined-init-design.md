# PEFT PISSA Refined Init Design

## Goal

Validate whether `init_lora_weights=pissa` can break the current `PEFT refined` performance plateau on the already-converged 14-target curated target set.

## Why This Exists

Current `PEFT refined` experiments have already narrowed the useful target structure to:

- exclude shallow stem targets
- keep mid/late semantic blocks
- keep `17.conv` and `20.conv`

Recent repo-local experiments also show:

- changing `rank/alpha` within `r8/a16`, `r12/a24`, and `r16/a32` only moves results within a narrow band
- lowering `lora_lr_mult` from `2.0` to `1.5` does not improve the unified checkpoint result

This means the next most valuable optimization axis is initialization rather than more target pruning or more learning-rate tuning.

## Scope

### In Scope

- keep the existing 14-target refined PEFT target set unchanged
- run focused real experiments with `init_lora_weights=pissa`
- compare `pissa` runs against the current best `PEFT refined` baseline and the current `fallback exact` baseline
- preserve runtime metadata and unified report behavior

### Out of Scope

- changing fallback backend behavior
- changing target matching semantics again
- broad hyperparameter sweeps beyond the selected comparison points
- introducing `olora` in the same first validation step

## Baseline

Current reference runs on the 14-target refined set:

- `peft_curated_targets_refined_r12_a24_e5`
- `peft_curated_targets_refined_r16_a32_e5`

Current observed position:

- best `PEFT refined` is effectively on a small plateau around `mAP50 ~= 0.651`
- current `fallback exact` remains slightly ahead

## Recommended Approach

Use a two-run validation instead of a single-run probe.

### Run A

- backend: `peft`
- target set: current 14-target refined list
- `r=12`
- `alpha=24`
- `lora_lr_mult=2.0`
- `init_lora_weights=pissa`

### Run B

- backend: `peft`
- target set: current 14-target refined list
- `r=16`
- `alpha=32`
- `lora_lr_mult=2.0`
- `init_lora_weights=pissa`

## Why Two Runs

This avoids overfitting the initialization conclusion to a single rank point.

- `r12/a24` represents the stable mid-rank baseline
- `r16/a32` represents the slightly stronger `mAP50` point under the current refined setup

If both improve, `PISSA` is likely a real gain for this target regime.
If only one improves, we learn which rank band benefits from `PISSA`.
If neither improves, initialization is probably not the bottleneck for the current refined path.

## Implementation Notes

No new code path is required if the current PEFT config builder already passes `init_lora_weights` through for `LoraConfig`.

Implementation should therefore focus on:

- launching the two real training runs with explicit `lora_init_lora_weights=pissa`
- verifying saved runtime metadata reflects the requested init mode
- refreshing unified comparison output after both runs complete

If a PEFT/runtime incompatibility appears during the run:

- fail clearly
- do not silently downgrade the init mode
- inspect the emitted metadata before trusting the result

## Success Criteria

The experiment is successful if all of the following are true:

- both runs train successfully and save adapters
- runtime metadata records `requested_init_lora_weights` and `effective_init_lora_weights` consistently
- unified comparison report includes both new runs
- we can clearly classify the outcome as one of:
  - `PISSA improves refined PEFT`
  - `PISSA is neutral`
  - `PISSA regresses refined PEFT`

## Evaluation Criteria

Primary comparison:

- compare against current best `PEFT refined`

Secondary comparison:

- compare against current `fallback exact`

Decision rule:

- prefer the run with the best unified `mAP50-95`
- use `mAP50` as secondary tie-breaker
- treat improvements smaller than roughly `0.001` as practical ties unless the pattern repeats consistently

## Risks

### PEFT Compatibility Risk

`PISSA` support may exist in config plumbing but still fail at actual runtime for the current module types or installed PEFT behavior.

Mitigation:

- start with the existing proven PEFT target set
- inspect runtime metadata and adapter output after the first run starts

### False Positive Risk

Single-run fluctuation can look like a gain.

Mitigation:

- compare two rank points instead of one
- use unified checkpoint validation rather than in-training snapshots

### Hidden Fallback Risk

If backend selection silently changes, the result would be invalid for this question.

Mitigation:

- keep `lora_backend=peft`
- verify `effective_backend=peft` in runtime metadata

## Expected Outcome

Most likely outcomes, in order:

1. `PISSA` gives a small but real gain on one or both refined PEFT baselines
2. `PISSA` is effectively neutral and the current plateau remains
3. `PISSA` regresses these YOLO convolution-heavy refined targets

## Follow-up Rules

If `PISSA` improves:

- promote the better of the two runs as the new refined PEFT baseline
- then consider a targeted follow-up with `olora` only if the improvement is still insufficient

If `PISSA` is neutral or worse:

- stop spending time on init-mode tuning for this branch
- shift the next optimization step to a different axis such as regularization or a broader architecture-level target rethink
