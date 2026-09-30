# Phase 1G: final teacher-free dataset

The earlier stages saved descriptions separately. This stage joins them by image
ID and exports the existing assignments; it does not generate text, filter records,
move images, or change split IDs. It imports no teacher, Torch, Transformers or API
client. Training must consume these files and local images only; no teacher call
belongs in the training pipeline.

## Run on rama after local review/commit/push

```bash
cd /home/jayanth/projects/nanovlm_new
git pull --ff-only
conda activate qwen-vl
python -m src.data.export_dataset \
  --metadata data/processed/coco_metadata.json \
  --splits data/splits/coco_splits.json \
  --short-dir data/generated/shortdesc_qwen \
  --long-dir data/generated/longdesc_qwen_v2 \
  --output-dir data/processed/final
```

Expected counts are 25,200 train, 2,800 val and 100 test. CPU only; no new model
downloads, GPU inference, or environment changes. The input directories must each
contain their original `run.json` and completed description JSONL.

## Output on rama

Under `/home/jayanth/projects/nanovlm_new/data/processed/final/`:

- `train.jsonl`: 25,200 records.
- `val.jsonl`: 2,800 records.
- `test.jsonl`: all 100 existing held-out records, not a newly selected subset.
- `manifest.json`: counts, assignment mapping, source fingerprints, generation
  run contracts and checksums for each exported JSONL file.

Each record has `schema_version`, `image_id`, `image_path`, `source_captions`,
`source_caption_ids`, `short_desc`, `long_desc`, `split`, and `source_split`.
`generation` and `validation` each have separate `short_desc` and `long_desc`
entries so their different prompts, generation timestamps, seeds/configurations,
teacher identity, run signatures and original validation results remain traceable.
Raw retry histories stay in the original generation artifacts, rather than being
duplicated in every downstream dataset. No teacher execution is needed to use
any of these records.

Image paths are relative to `data_root`, for example
`images/train2017/000000272081.jpg`. Resolve using the machine's configured root:
`/home/jayanth/datasets/coco` on rama or
`/home/jayanth/Research/datasets/coco` locally. Images are not copied or embedded.
The `source_split` is the original COCO folder, not the experimental split.
Records keep the reviewed split-manifest order, independent of the row order in
the generated input files. The 100 held-out IDs are renamed to `test` only for
this final representation. All original records are retained, including length
flags; the export is not a claim that semantic accuracy has been established.

## Integrity and reruns

The exporter checks metadata/split hashes, assignments, exact ID coverage,
duplicate/unexpected IDs, source captions and caption IDs, image paths, and
per-record generation provenance against the corresponding run contract. Missing
or mismatched records cause explicit failure before publication. It does not
silently intersect filtered short/long sets or call a teacher to fill gaps.

The four files are staged and published together as a directory. An interruption
before publication does not leave a partial final dataset. An identical existing
export reports `unchanged`; a differing export is refused. Use a new output
directory for an intentionally revised dataset. Do not add unrelated files inside
the export directory. File locking coordinates concurrent exporters on Linux.
No source file is modified.

These generated outputs are ignored by Git under `data/processed/`. Commit the
exporter, tests and documentation; preserve/back up the generated data separately.
The complete dataset has not been exported locally because the real generated
ShortDesc/LongDesc files are on rama. Local tests use temporary fixtures and cover
ID-based joins, coverage, assignments, provenance and safe reruns.
