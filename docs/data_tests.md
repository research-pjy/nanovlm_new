# Phase 1H: DATA tests and small-data mode

The project seed is **42**. The earlier 404 specification is superseded.
Length checks are advisory: ShortDesc 20–27 words and LongDesc 60–70 words. A
length warning does not fail DATA checks, silently delete a sample, or certify
semantic quality.

## Local automated tests

```bash
python3 -m unittest discover -s tests -q
```

The suite covers all ten requirements: selection size and deterministic seed-42
selection (including an independent reference order), duplicate IDs, missing images,
missing captions, split leakage, availability of both descriptions, short and long
word counts, and reload of saved datasets. The integration test runs a 100-image
subset through both generation runners, description validation, the ID join/split
export, and final reload with temporary fixture images and **test-only teachers**.
It does not download or invoke a real teacher during unit tests.

## Full saved-dataset check on rama

After local review/commit/push:

```bash
cd /home/jayanth/projects/nanovlm_new
git pull --ff-only
conda activate qwen-vl
python -m unittest discover -s tests -q
python -m src.data.check_dataset \
  --dataset data/processed/final \
  --data-config configs/data.rama.json \
  --metadata data/processed/coco_metadata.json \
  --splits data/splits/coco_splits.json \
  --report data/processed/final_check.json
```

The check reproduces the source selection using seed 42 from the full COCO caption
annotations and compares the metadata to it. It reloads the saved dataset twice,
checks export checksums, manifest counts, expected IDs/order, disjoint assignments,
image paths/existence, original captions and caption IDs, and availability of both
descriptions. Word-count distributions and warnings are reported independently
of hard errors. Paths are checked under the configured data root. Full image
**decoding** is covered by Phase 1A, not repeated here. Semantic grounding is not
assessed. Expect `ok: true`, counts 25,200 / 2,800 / 100 and
`deterministic_source_selection_verified: true` before declaring DATA complete.
Reports are immutable: use a new report path if the input dataset changes.

## Prepare 100 images without inference

```bash
python -m src.data.prepare_small \
  --data-config configs/data.rama.json \
  --num-images 100 \
  --output-dir data/processed/smoke100/inputs
```

Here `--num-images` is the **total** across all three assignments: 89 train,
10 validation, 1 held-out. This differs from the original `number_of_images`
configuration, which counts only the training/validation pool. At least one image
is reserved per assignment; the remainder approximates the 90/10 ratio. Sampling
uses seed 42 within the existing split pools; no image moves between assignments.
The subset preserves the selected records' original order. Requests too large or
too small to represent all splits fail explicitly.

The existing full selection is verified and reused. There are no downloads or
image copies. `metadata.json` and `splits.json` contain independent subset
fingerprints, counts and IDs, and keep provenance back to the full source data.
Running preparation again produces identical inputs; differing inputs at the same
path are refused. The full 28K pipeline artifacts remain unchanged.

## Complete 100-image smoke pipeline on rama

```bash
python -m src.data.smoke_pipeline \
  --data-config configs/data.rama.json \
  --num-images 100 \
  --work-dir data/processed/smoke100
```

This is the real end-to-end path, including inference on the L40S:

1. Verify the reused source data and create the deterministic subset.
2. Generate ShortDesc in batches of 16, with checkpoint/resume.
3. Write its description-validation report without filtering.
4. Generate LongDesc in batches of 16 and write its validation report.
5. Export `final/train.jsonl`, `val.jsonl`, `test.jsonl`, and `manifest.json`.
6. Reload and check the subset, re-verifying its deterministic membership against
   the source selection, and write `dataset_check.json`.

The short and long generation steps each perform their established runtime, GPU,
VRAM and disk checks and use cached models only. Models are loaded in separate
processes so they are not kept on the GPU simultaneously. No scheduler is used.
If custom teacher configs are needed, pass `--short-config` and `--long-config`.
`--validation-config` optionally selects another reporting policy; this smoke
runner deliberately does not filter samples.

All outputs stay inside the supplied workspace: `inputs/`, `short/`, `long/`,
`short_validation.json`, `long_validation.json`, `final/`, and `dataset_check.json`.
Use a dedicated new workspace per configuration/dataset size. Do not point this
at an existing full-run folder. Rerunning the same command resumes saved batches
and reuses identical exports. A generation exit code 2 (saved validation flags)
continues to reporting; operational errors stop the pipeline. Inspect the reports
as well as the process exit status. The full 28K generation never runs in this mode.

The 100-image metadata preparation and test-teacher integration were verified
locally; actual GPU smoke inference and the full-data reload check must be run on
rama before final acceptance. No new TASK, MODEL, LOSS, LEARNING or EVALUATION
bucket implementation is included.
