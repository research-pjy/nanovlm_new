# Phase 1E, step 1: ShortDesc

This step generates short descriptions only. LongDesc will be implemented
separately; each record currently has `long_desc: null` and a `not_generated`
validation status for it. Images and source data are never changed.

## Teacher and scientific choices

The default teacher is the Hugging Face `Qwen/Qwen3-VL-8B-Instruct` checkpoint,
loaded directly through Transformers, with BF16 and PyTorch SDPA on the L40S.
No Ollama server, paid API, scheduler, or environment modification is used.
The integration follows the official [Qwen model card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
and [Transformers Qwen3-VL documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_vl).

**Teacher input is captions only**, as agreed, matching the paper's input modality
for description generation. The VLM runs as a text generator; it receives all
original captions retained by Phase 1C (including a sixth where available), not
image pixels. The prompt requests 20–25 words, 2–3 simple sentences, a child-friendly
tone, and descriptions grounded in the supplied captions. It is an explicit
adaptation of the paper's prompt, not a claim to duplicate its GPT-4o outputs.

The teacher interface is `src/data/teachers/base.py::TeacherGenerator`.
The runner accepts any adapter supplying batched `TeacherOutput` values, model
identity/runtime provenance, and optional memory statistics (an empty dictionary
is sufficient). Qwen is the implemented production adapter. An OpenAI or other
local adapter can be added through this boundary without changing the generated
record schema; no other production backend is claimed to be implemented.

The supplied configuration uses:

- Batch size 16; a smaller final batch or retry subset is expected.
- Seed 42 as the base. Each batch/attempt uses
  `42 + batch_index * max_attempts + attempt_index`, with zero-based indices.
- Greedy decoding (`do_sample=False`, one beam), 128 new-token limit, no input
  truncation, and a 2,048 input-token guard.
- Up to three attempts per image. Only invalid responses are requested again,
  with the previous attempt and validation feedback added to the prompt.

Seed/batch settings are recorded, but GPU numerical behavior is not guaranteed to
be bitwise identical across different software/hardware. Resume refuses changed
package versions, model assets, runner/adapter code, generation settings, prompts,
or input fingerprints to avoid silently mixing experiments.

## Before the first GPU run

The model setting defaults to the **standard Hugging Face cache**. All model
resolution/loading uses `local_files_only=True`; nothing is downloaded. The cached
revision is located through `config.json`, rather than requiring a complete Hub
repository snapshot. Missing `README.md` or `.gitattributes` is harmless; missing
weight shards, tokenizer assets, or a usable chat template still fails preflight.
If your checkpoint is in a separate directory, copy the JSON configuration locally,
set `model` to that directory's full path, review/commit that configuration, and
use it consistently in every command. Do this before the first generation batch.
Both model weights and tokenizer/chat-template assets must be available.
Do not switch between Instruct and Thinking inside one output directory.

After reviewing and committing locally, push, then run on rama. Proceed only if
each command succeeds:

```bash
cd /home/jayanth/projects/nanovlm_new
git pull --ff-only
conda activate qwen-vl
python -m unittest discover -s tests -q
```

If the earlier data phases have not been run on rama, first run:

```bash
python -m src.data.verify_existing --config configs/data.rama.json
python -m src.data.export_metadata --config configs/data.rama.json --output data/processed/coco_metadata.json
python -m src.data.export_splits --config configs/data.rama.json --metadata data/processed/coco_metadata.json --output data/splits/coco_splits.json
```

The generation runner requires the exact Phase 1C metadata fingerprint in the
reviewed Phase 1D split file, matching assignment IDs/counts, and valid captions.
It generates all 28,100 selected images, preserving the 25,200 / 2,800 / 100
assignments. Held-out captions remain confined to held-out records; they must not
be used for model training in later phases.

## Preflight

```bash
python -m src.data.generate_shortdesc \
  --config configs/shortdesc.qwen.json \
  --metadata data/processed/coco_metadata.json \
  --splits data/splits/coco_splits.json \
  --output-dir data/generated/shortdesc_qwen \
  --preflight-only
```

This checks L40S detection, BF16 support with an actual CUDA matrix operation,
package versions, CUDA runtime, free VRAM (at least 24 GiB by default), output disk
space (at least 2 GiB), and offline model assets. It reads and hashes the model
files, which can take a minute. It does not allocate the full model or generate
text, so the smoke run is still needed to establish actual batch memory usage.
These checks run again when generation starts. It does not change the environment.
Use `--check-inputs` instead for a CPU-only input/config check, with no model imports
or generated files.

## Smoke run: two batches (32 images)

```bash
python -m src.data.generate_shortdesc \
  --config configs/shortdesc.qwen.json \
  --metadata data/processed/coco_metadata.json \
  --splits data/splits/coco_splits.json \
  --output-dir data/generated/shortdesc_qwen \
  --max-batches 2
```

Each committed batch prints elapsed time, saved count, invalid count, and GPU
allocated/reserved/peak/free memory. The final summary previews three descriptions.
Review the output style, grounding against source captions, invalid rate, and GPU
memory before scaling up. The automatic checks do not establish semantic quality.
The model is loaded only when a pending batch needs generation.

A CUDA out-of-memory error stops the run and preserves previous batches. Do not
silently lower batch size, truncate captions, quantize, or change the teacher.
If a hardware adjustment is necessary, review it and use a separate configuration
and output directory; keep scientific choices explicit.

## Full run or resume

Once the smoke results are acceptable, remove `--max-batches` and keep all other
arguments unchanged:

```bash
python -m src.data.generate_shortdesc \
  --config configs/shortdesc.qwen.json \
  --metadata data/processed/coco_metadata.json \
  --splits data/splits/coco_splits.json \
  --output-dir data/generated/shortdesc_qwen
```

This resumes after the smoke batches. `--max-batches` limits new batches per
invocation, not the total dataset. Run directly in a terminal on rama (a persistent
terminal session may be used for a long run); no scheduler is involved.

## Saved files and recovery

All files are under the output directory and ignored by Git:

- `checkpoint.sqlite3`: authoritative, transactionally committed batch results.
- `run.json`: immutable configuration, full prompt, input/model/code fingerprints,
  package versions, and teacher identity.
- `shortdesc.jsonl`: one generated record per line, in original metadata order.
- `.lock`: prevents concurrent writers to the same directory; the OS releases the
  lock when the process exits, including after a crash.

Keep the checkpoint and run manifest together. Completed batches, including
invalid records whose attempts are exhausted, are never regenerated on resume.
An interrupted, uncommitted batch is retried in full with its original seed scheme.
JSONL is refreshed atomically when the invocation ends, including handled errors
and Ctrl-C. After a power loss or forced kill it may lag behind SQLite; resume the
same command or refresh it without a GPU:

```bash
python -m src.data.generate_shortdesc \
  --config configs/shortdesc.qwen.json \
  --metadata data/processed/coco_metadata.json \
  --splits data/splits/coco_splits.json \
  --output-dir data/generated/shortdesc_qwen \
  --export-only
```

Every generated record retains `image_id`, portable `image_path`, assignment,
`source_captions`, source caption IDs, `short_desc`, `long_desc`, `teacher_model`,
UTC generation timestamp, actual attempt seed, generation configuration, run
signature, validation, and all raw attempts. Short-description validation counts
whitespace-delimited tokens containing a letter/digit as words (hyphenated words
and contractions count as one). It checks 20–25 words, detectable reasoning/list
formatting, and whether the token limit was reached without EOS. It does not check
English fluency, sentence count, child-level vocabulary, or visual truth;
`semantic_review` is explicitly `not_performed`.

Invalid attempts are retained and reported, never silently dropped or labeled
valid. Exit code 2 means saved output includes invalid descriptions and needs
review, 1 means an operational failure, and 130 means interruption. Exit code 0
can describe a successful partial smoke run; check the summary's `complete` field.
A completed run is not automatically a training-ready dataset. Later validation
and long-description phases are separate work.
