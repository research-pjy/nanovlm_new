# Phase 1E, step 2: LongDesc

LongDesc is generated independently from all original COCO captions, not by
expanding ShortDesc. It uses the same Qwen3-VL-8B-Instruct adapter, captions-only
input, BF16/SDPA, batch size 16, greedy decoding, seed 42 and existing assignments.
The prompt targets approximately 60–70 words in 5–6 simple sentences and asks
for supported details without repetition or invented objects/events.

The new configuration raises the output token limit to 256. Length outside
60–70 is an advisory warning, not a retry trigger or rejection: this avoids
forcing awkward rewrites to meet an exact cutoff. Empty text, detected reasoning
or unwanted formatting, and missing EOS at the token limit are errors, with up
to three attempts. Semantic grounding and sentence quality still require review;
`valid` means passing the implemented structural checks, not human approval.
Very short/long descriptions remain visible in the length histogram and should
be reviewed before training. ShortDesc's original flags are not changed.

The completed short runner and teacher files are left unchanged to preserve their
resume fingerprints. LongDesc has its own phase identity, prompt, validator,
checkpoint, and output file. It reuses the teacher interface and input validation.
Pointing it at the ShortDesc output directory fails before writing descriptions.

## Read-only ShortDesc check on rama

After committing/pushing locally and pulling on rama, run:

```bash
cd /home/jayanth/projects/nanovlm_new
git pull --ff-only
conda activate qwen-vl
python -m src.data.report_descriptions --input data/generated/shortdesc_qwen/shortdesc.jsonl --field short_desc
```

This separates length-only flags from token-limit/formatting failures, reports
word counts, and counts 20–27-word outputs with no other errors. It does not
change old validation statuses, discard descriptions, or assess semantic quality.
Share the report to decide whether any ShortDesc repairs are actually needed.
The user-reported 28,100 completed rows establish generation completion, not
that all 4,585 flags were length-only.

## LongDesc preflight and smoke test

Run each command only after the previous one succeeds. Qwen cache resolution,
GPU/runtime checks, fingerprinting, and memory reporting work as in ShortDesc.
If ShortDesc used a custom model path, set the same path in the new configuration
before running. No model downloads or dependency changes are performed.

```bash
python -m src.data.generate_longdesc \
  --config configs/longdesc.qwen.json \
  --metadata data/processed/coco_metadata.json \
  --splits data/splits/coco_splits.json \
  --output-dir data/generated/longdesc_qwen \
  --preflight-only

python -m src.data.generate_longdesc \
  --config configs/longdesc.qwen.json \
  --metadata data/processed/coco_metadata.json \
  --splits data/splits/coco_splits.json \
  --output-dir data/generated/longdesc_qwen \
  --max-batches 2
```

Inspect the three preview descriptions, error counts, length-warning count, and
GPU memory before scaling up. Longer responses can use more time and memory;
ShortDesc success alone does not verify LongDesc batch behavior.

## Full run / resume after reviewing the smoke test

```bash
python -m src.data.generate_longdesc \
  --config configs/longdesc.qwen.json \
  --metadata data/processed/coco_metadata.json \
  --splits data/splits/coco_splits.json \
  --output-dir data/generated/longdesc_qwen
```

This resumes after saved batches. There is no scheduler. Completed batches,
including exhausted invalid attempts, are preserved; only an interrupted unsaved
batch is regenerated. Changes to the inputs/configuration/model/runtime/code
require a new output directory. `--export-only` regenerates JSONL from the
checkpoint without GPU/model loading. `--check-inputs` validates inputs locally
without loading a model. Exit code 2 indicates structural errors remain, not
length warnings; exit code 0 may still describe a partial smoke test.

## Outputs on rama

Under `/home/jayanth/projects/nanovlm_new/data/generated/longdesc_qwen/`:

- `longdesc.jsonl`: generated long descriptions and source captions, IDs,
  assignments, timestamp, teacher model, seeds/configuration, validation/warnings,
  and attempt history.
- `checkpoint.sqlite3`: authoritative, transactionally saved batch progress.
- `run.json`: prompt, model/runtime/configuration and input fingerprints.

In these long-only records, `short_desc` is null and marked `not_generated`.
Existing short descriptions remain in `shortdesc_qwen/shortdesc.jsonl`.
The two artifacts correspond by `image_id` and unchanged assignment; no merged
artifact is created in this step and neither source is overwritten.
Keep checkpoints with their run manifests. JSONL is refreshed when a command
ends; after a forced kill it may lag the checkpoint until resume/export-only.

Once the full run finishes, inspect it with:

```bash
python -m src.data.report_descriptions --input data/generated/longdesc_qwen/longdesc.jsonl --field long_desc
```

No real LongDesc generation was performed locally. Local verification uses test
teachers in temporary directories. Phases 1F–1H remain outside this implementation.
