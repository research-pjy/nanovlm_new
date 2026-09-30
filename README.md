# NanoVLM convolution placement study

An incremental research implementation based on *NanoVLMs: How small can we go
and still make coherent Vision Language Models?*

## Current scope

Phase 0 scaffolding, modified Phase 1A (existing download verification), and
Phase 1B (deterministic selection verification) are implemented. No selection generation, metadata export, model,
loss, training, or evaluation is implemented. Develop one phase at a time.

## Research question

The paper describes 224 × 224 images, 16 × 16 patches, 196 image tokens, and two
2D convolutional layers, but does not unambiguously specify convolution placement.
The planned visual encoder will support two explicit interpretations:

- **Global-Conv:** apply convolution to the complete image before patch extraction.
- **Patch-Conv:** extract patches first and apply convolution independently to them.

Neither interpretation is claimed to be an exact reproduction of the paper.
The initial comparison will vary only convolution placement. Keep the connector,
decoder, task, data and split, loss, optimizer, and other experimental settings
identical between branches. Resolve unspecified architectural details explicitly
during the MODEL bucket, before implementation.

## Bucket boundaries

The conceptual pipeline is DATA → TASK → MODEL → LOSS → LEARNING → EVALUATION.
Each listed component should remain independently replaceable.

1. **DATA** (`src/data/`): COCO loader, description generator, dataset validator,
   dataset splitter, and task dataset adapter. Task semantics belong in TASK.
2. **TASK** (`src/tasks/`): partial-text construction, target construction, and
   loss-mask construction.
3. **MODEL** (`src/models/`): vision encoder, configurable convolutional patch
   embedding, connector/projector, language decoder, and multimodal fusion.
4. **LOSS** (`src/losses/`): causal language-model loss consuming the task mask.
5. **LEARNING** (`src/learning/`): optimizer, scheduler, training and validation
   loops, and checkpointing.
6. **EVALUATION** (`src/evaluation/`): generation, automatic metrics, LLM
   evaluation, parameter counts, latency, memory, and ablations.

## Repository layout

```text
configs/                  Future experiment configuration
data/
  raw/                    Original source data
  generated/              Generated descriptions
  processed/              Validated and processed data
  splits/                 Future dataset split manifests
src/
  data/
  tasks/
  models/
    vision/               Both convolution strategies live here
    connector/
    decoder/
    nanovlm.py            Reserved composition entry point
  losses/
  learning/
  evaluation/
scripts/                  Future command-line entry points
experiments/              Future experiment specifications and notes
tests/                    Future component and integration tests
```

Empty directories contain `.gitkeep` files so they can be versioned. Local data
contents are ignored by default; decide which split manifests to version during
the DATA bucket. Dependencies and executable configurations will be introduced
when their corresponding implementation is authorized.

## Development and execution

- Develop locally in `/home/jayanth/codexprojects/nanovlm_new` and test/review
  locally where practical.
- The user reviews and commits changes. Provide a suggested commit message at
  each completed task, phase, or milestone; do not commit automatically.
- After commit and push, pull the reviewed commit into the repository cloned
  under `/home/jayanth/projects` on `rama` and execute there.
- Use `rama` as the remote compute machine: NVIDIA L40S, approximately 44 GiB
  usable VRAM, approximately 247 GiB system RAM, and BF16 support.
- Run directly in the existing `qwen-vl` Conda environment. Do not use Slurm,
  `sbatch`, partitions, or another job scheduler.
- Before experiments, verify current package versions, PyTorch/CUDA
  compatibility, GPU detection, available VRAM, and disk space. Previously
  reported versions are historical, not dependency pins: Python 3.11.16,
  PyTorch 2.14.0, Transformers 5.17.0, Accelerate 1.15.0, CUDA runtime 13.0.
- Preserve the environment unless a demonstrated compatibility issue requires
  changing it. Avoid uncommitted remote code edits except necessary debugging.
- Use BF16 where appropriate, monitor initial GPU memory use, and tune batch
  size, workers, and gradient accumulation from measurements. Keep execution
  tuning separate from scientific configuration and preserve the controlled
  comparison, including effective batch size when adjusting accumulation.

No remote execution or environment changes are part of Phase 0.


## DATA Phase 1A: reuse the existing download

Keep `images/train2017/` and `images/val2017/`. These are COCO source locations,
not experimental assignments. `image_selection.json` defines our custom train,
validation, and held-out assignments independently. Seed **42** is authoritative.
`number_of_images` counts train plus validation; the existing 100 held-out images
are additional to the 28,000 images.

Run from the repository root with Python 3.11+ and Pillow already available:

```bash
python3 -m src.data.verify_existing --config configs/data.local.json
```

After user review, commit, push, and pull on rama, activate `qwen-vl` and use:

```bash
python -m src.data.verify_existing --config configs/data.rama.json
```

Machine paths are confined to the configuration files; edit `data_root` for
another location. The checker reads caption annotations and the existing manifest,
checks counts and seed, detects repeated selected IDs/paths, compares saved
captions to source annotations, and verifies and fully decodes every selected
image. It checks decoded dimensions, reports image modes/formats, prints progress,
and exits nonzero on errors. Missing or unreadable inputs fail clearly. Sample
errors are collected and reported, never silently dropped. The checker does not
write to the dataset, regenerate selections, extract archives, or download data.
It verifies the extracted files needed by this experiment, not the unused ZIP.
It does not establish near-duplicate absence or verify remote source checksums.

A downloader is deferred because both machines already have the selected data.
Full COCO image archives are unnecessary for this existing subset. Check rama's
copy separately; local success does not establish remote integrity.

Remaining steps, to implement separately:

- **1C:** export portable metadata with relative image paths and all original
  captions. Some COCO images have more than five captions; the existing manifest
  retains the first five, but the source annotations must remain authoritative.
- **1D:** persist and verify the existing 25,200 / 2,800 / 100 assignments without
  reshuffling. Decide the final evaluation protocol separately.
- **1E–1H:** await specifications.

Validation:

```bash
python3 -m unittest discover -s tests -v
```


## DATA Phase 1B: preserve and verify selection

The existing `image_selection.json` remains the authoritative saved selection.
This phase never writes, replaces, or regenerates it, even when it is missing or
inconsistent. It reconstructs the expected ordered IDs **in memory** from the
complete caption annotation files, not from the downloaded image subset:

1. Include source images with at least five captions.
2. Sort unique image IDs, then shuffle with `random.Random(42)`.
3. Reserve the first 100 IDs for held-out evaluation.
4. Take the next 28,000 IDs and use `round(28000 * 0.9)` for the training boundary.
5. Compare every ordered ID and source filename/split with the existing manifest.

The configs now explicitly specify `number_of_held_out` and `train_fraction`, in
addition to `data_root`, `number_of_images`, and `random_seed`. Configuration and
manifest parameters must agree. Duplicate IDs, assignment changes, reordered IDs,
incorrect counts, and malformed source annotations cause verification to fail.
Images with fewer than five captions are counted as ineligible, matching the
original downloader; malformed captions are errors rather than silently excluded.

```bash
python3 -m src.data.verify_selection --config configs/data.local.json
```

On rama, after the normal reviewed-commit transfer, use the same command with
`python` in `qwen-vl` and `--config configs/data.rama.json`. No image decoding or
third-party dependencies are needed for this phase; use Phase 1A for image health.

The JSON report goes to standard output and includes candidate counts, assignment
counts, errors, and SHA-256 fingerprints of the exact manifest and annotation
files read. These hashes can identify differing local and remote copies; they are
not a replacement for a trusted source checksum or a permanently pinned dataset
version. Exit status is zero only on success. No remote validation has been run.

Selection IDs and assignments stay stored in the existing manifest. Portable
metadata export and repository split artifacts remain for Phases 1C and 1D.
