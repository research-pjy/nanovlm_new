# Phase 5 — Tiny-data learning gate

The complete model passed the 175-test suite and real CPU/CUDA-BF16 acceptance
on rama. This phase implements actual optimizer updates, first on a tiny subset.
It does not offer or automatically launch full-dataset training.

## Fixed experiment

`configs/learning.overfit.json` selects 10 training images and 10 distinct validation
images, independently shuffled from sorted split IDs with seed 42. Each image
provides SHORT and LONG examples (20 training + 20 validation examples). All
selected IDs and image hashes are saved. The held-out test set is never read.
Existing production data, split files and tokenizer are unchanged.

Both branches must use the same config, tokenization, transforms, data ordering,
initialization seed and precision; only --strategy changes. Initial model weights
are constructed afresh from seed 42; do not initialize from another branch's
trained checkpoint. Tiny samples are cached in CPU RAM. DataLoader uses zero
workers, a separate deterministic generator seeded by 42+epoch for shuffle, and
no dropped final batch. Batch size 10, accumulation 1: two optimizer steps per
epoch, 600 steps over 300 epochs. The current model dropout stays 0.1.

Optimizer: AdamW, LR 1e-3, betas (0.9,0.999), epsilon 1e-8, weight decay 0 for
this memorization diagnostic; foreach/fused disabled for transparent replay.
Global gradient-norm clipping is 1.0. Learning rate is constant, no scheduler.
These are explicit debug choices, not a finalized full-training recipe.

Each optimizer step uses mean cross entropy over its supervised tokens.
Logged epoch training loss and evaluation losses are token-weighted sums divided
by token counts, not averages of batch means. There is no gradient accumulation
in this implementation. BF16 uses autocast without FP16 GradScaler. Non-finite
loss/gradients fail before optimizer updates. Validation uses eval mode, no-grad
and FP32; it never updates weights. Both tasks are evaluated separately each epoch.

## Predeclared pass criterion

For EACH task, eval-mode loss on the fixed TRAIN subset must:
1. decrease at least 80% from its initial value, AND
2. finish at or below 1.0 cross-entropy nats/token after 300 epochs.

The same gate must pass for each branch. There is no early-success stopping, so
paired runs use the same step budget. Validation is logged separately; the gate
is intentionally training-set memorization, not generalization. It does not prove
visual grounding or generated-language quality. If either task fails, exit status
is nonzero and status.json says failed. Stop and debug before large-scale training;
do not loosen the gate after seeing results without documenting a new experiment.

## Reproducibility and checkpointing

Python and torch use seed 42. Deterministic algorithms, no cuDNN benchmarking,
math attention backend and CUBLAS_WORKSPACE_CONFIG=:4096:8 are configured before
training. Numerical identity is expected only within the same recorded runtime
and device, not between CPU and CUDA or library versions.

The resume contract includes resolved model/learning settings, tokenizer and
pixel-transform identity, dataset hashes, selected IDs/image hashes, Python/torch/
CUDA/device/precision, optimizer details and hashes of every src Python file.
The run also records Git revision and dirty-tree status. Changed contracts fail
rather than silently starting over. This strict source check means unrelated src
edits also require reviewing and starting a new run; there is no override yet.

latest.pt is an atomically replaced training checkpoint, separate from the
model-only checkpoints of 3L. It stores model and optimizer state, completed epoch,
step count, initial losses, metrics history, Python/torch/CUDA RNG states and
scheduler=None (constant LR). Data order is regenerated from seed+epoch. Saves
occur at epoch zero, each 10 epochs, final epoch, and a controlled interruption.
Resume restarts at the next epoch after the last saved boundary. A crash may
replay up to nine unsaved epochs; it never claims to resume an arbitrary batch.

An exclusive lock prevents simultaneous writers in the same output directory.
Checkpoint history is authoritative: metrics.json is reconstructed from it on
resume, discarding any uncommitted history. Never share output directories across
strategies. Output must be new unless --resume is explicitly supplied.

Outputs:
- run.json: contract, parameter counts, code identity and initial losses.
- latest.pt: resumable training state, including full metrics history.
- metrics.json: per-epoch training/eval/validation losses, steps, LR, time, memory.
- status.json: interrupted/passed/failed with per-task gate results.
- model.pt: final model-only export with gate outcome in provenance, even if failed.

A failed model export is not permission to train at scale. Preserve failed-run
artifacts for diagnosis. No code path invokes a teacher or modifies source data.

## Run on rama after review/commit/push/pull

```bash
conda activate qwen-vl
python -m unittest discover -s tests -p 'test_learning.py' -v
python -m unittest discover -s tests -v

python -m src.learning.overfit \
  --dataset-dir data/processed/final \
  --data-root /home/jayanth/datasets/coco \
  --tokenizer-dir data/processed/student_tokenizer \
  --model-config data/processed/model.debug.tokenized.json \
  --preprocessing-config configs/image_preprocessing.json \
  --learning-config configs/learning.overfit.json \
  --strategy global --device cuda --bf16 \
  --output-dir runs/overfit_global
```

First review Global results. If its gate passes, run the identical command with
`--strategy patch --output-dir runs/overfit_patch`. Keep all other arguments and
configs the same. Share both status.json files and metrics if either gate fails.
GPU preflight records compatibility/availability, free VRAM/disk; per-epoch logs
show peak allocated memory. Do not infer a full-training batch size from this run.

To test interruption, add `--stop-after-epochs 10` to a fresh run. It saves an
interrupted status, not a pass. Continue the same run with `--resume` and omit
--stop-after-epochs. Do not change the configured epoch budget on resume. The
flag bounds additional epochs for that invocation, not total run epochs.

Five learning tests cover config/gate checks, actual parameter updates, exact
CPU epoch-boundary resume with dropout, token-weighted no-update evaluation and
CUDA BF16 optimizer execution. Local numerical tests skip without torch. The
real tiny-data gate must pass on rama before the full-training phase is designed.
