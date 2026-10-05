# Phase 2: TASK bucket

The task is **image + partial text → continuation**. The task layer does not know
vision token counts, convolution placement, projector dimensions, decoder layers,
or the eventual model architecture. It never calls a teacher. Student-model,
loss computation, optimization, and training-loop implementation remain separate.

## Text construction

`configs/task.json` selects base seed 42, SHORT prefixes of 6–7 words and LONG
prefixes of 18–20 words. SHORT uses `short_desc`; LONG uses `long_desc` from the
final DATA record. Word counting follows DATA: a whitespace-delimited token that
contains a letter, digit or underscore counts as one word. Hyphenated words and
contractions count as one; punctuation-only tokens are not words.

The prefix length is chosen using SHA-256 of `[seed, image_id, variant]`, mapped
into the inclusive configured range. It is stable across runs, process order,
workers, batch size and epochs. No global random generator is touched. Thus both
future encoder branches receive identical tasks. Varying prefixes by epoch is not
implemented or silently enabled.

The split occurs immediately after the chosen word, retaining original punctuation,
Unicode, newlines and spacing. `prompt_text + target_text` exactly equals the
original description; the continuation usually starts with its original separator
space/newline. It is not stripped or rewritten. There must be at least one word
after the chosen prefix; otherwise the record raises a clear error. There is no
silent shortening, filtering or token truncation. All observed generated lengths
are sufficient for the default prefix ranges.

## Injectable tokenization

The tokenizer has not been chosen for training. `src/tasks/builder.py::Tokenizer`
defines an `encode(text, add_special_tokens=False) -> list[int]` interface. A later
Hugging Face tokenizer or a custom tokenizer can satisfy it directly or via a
small adapter. Choosing an injectable interface does not exclude Hugging Face.
This keeps vocabulary selection independent of the task logic and avoids adopting
the Qwen teacher's tokenizer by accident.

Prefix and continuation are encoded **separately**. The prefix encoder receives
only the prefix, preventing tokenization of future continuation text from changing
the prompt. Preserve the continuation's leading whitespace when using a tokenizer
whose tokens depend on it. Concatenating these independently encoded sequences
is the task's defined token sequence; do not replace it with a fresh tokenization
of the full description, which can merge across the boundary.

No BOS/EOS is added by default. When a tokenizer is selected, callers explicitly
supply its `bos_token_id` and/or `eos_token_id` to `TaskBuilder` if desired. The
builder masks BOS as prompt and supervises EOS as part of the target. PAD is
supplied separately to the batching helper. No vocabulary is fitted, downloaded,
or inferred from validation/test text by this implementation.

## API and images

```python
from src.tasks.builder import TaskBuilder, TaskConfig, collate
from src.data.task_dataset import TaskDataset

# tokenizer is supplied by the caller after its selection/configuration.
builder = TaskBuilder(
    tokenizer=tokenizer,
    data_root="/home/jayanth/datasets/coco",
    config=TaskConfig(),
    # Optional explicit bos_token_id=..., eos_token_id=...
)
dataset = TaskDataset("data/processed/final", "train", "short", builder)
example = dataset[0]
# batch = collate([dataset[0], dataset[1]], pad_token_id=chosen_pad_id)
```

Each `TaskExample` exposes `image`, `prompt_text`, `target_text`,
`tokenized_prompt`, `tokenized_target`, and `loss_mask`, plus ID, portable image
path, split, variant and prefix word count. `input_ids` concatenates the two token
sequences. Token IDs and masks are Python lists, not framework-specific tensors.

The default image loader returns an independent Pillow RGB image at its original
resolution, including conversion of grayscale images. It performs no resize,
normalization, cropping or patch extraction. A callable image loader may be
injected without changing text semantics. Paths are resolved against `data_root`,
with missing files and paths escaping the root rejected. Loading is lazy through
`TaskDataset`: creating the dataset verifies the chosen final JSONL's checksum,
count and record assignments; indexing loads/tokenizes one example. The caller
owns returned images and should release them when no longer needed.

## Loss-mask contract (no loss implementation)

`loss_mask` has the same length and positions as `input_ids`:

```text
input_ids = [prompt tokens ...][continuation tokens ...][optional EOS]
loss_mask = [0, 0, ...       ][1, 1, ...             ][1           ]
```

It marks **label token positions**, not already-shifted prediction positions.
When causal next-token loss is implemented, align:

```text
prediction logits: logits[:, :-1]
label IDs:         input_ids[:, 1:]
supervision mask:  loss_mask[:, 1:]
```

This supervises the first continuation token using the last prompt position, and
includes a supplied EOS. Shift once, not twice. This is a contract for later code,
not a current loss function. A future multimodal wrapper must align extra image
positions and mask them out according to its architecture; TASK cannot know those
positions in advance. The mask alone does not implement causal attention.

`collate` right-pads text IDs, returns images as a list, and supplies a separate
`attention_mask` (1 for real text positions, 0 for padding). Padding loss-mask
entries are zero even if PAD and EOS use the same ID. The attention mask is a
padding indicator; the future decoder still needs causal masking. Image tensors,
causal attention matrices and model-specific labels are not constructed here.
At inference, provide the image and prompt tokens only; target tokens are training
supervision and must not be passed as the generation prompt.

## Tests and rama checks

After reviewing/committing/pushing locally, run on rama:

```bash
cd /home/jayanth/projects/nanovlm_new
git pull --ff-only
conda activate qwen-vl
python -m unittest discover -s tests -q
```

Check the saved 100-image dataset end to end without a GPU or teacher:

```bash
python -m src.tasks.check_tasks \
  --dataset data/processed/smoke100/final \
  --data-config configs/data.rama.json \
  --task-config configs/task.json \
  --report data/processed/smoke100/task_check.json
```

Then check the complete dataset:

```bash
python -m src.tasks.check_tasks \
  --dataset data/processed/final \
  --data-config configs/data.rama.json \
  --task-config configs/task.json \
  --report data/processed/task_check.json
```

The checker uses a **diagnostic UTF-8 byte tokenizer**, explicitly labeled in its
report. This is a reversible test probe, not the selected student tokenizer, and
its token counts should not be used to set the eventual model's context length.
It checks both task variants on every split, text/token round trips, masks, and
actual image decoding to RGB. With the full dataset it checks 56,200 examples;
image reads take time, but no model is loaded. For a shorter check,
`--num-images 100` checks at most 100 images **per split** for each variant;
this limits inspection and does not create a new dataset.

The report records the configuration and dataset-manifest fingerprint, prefix
length distributions, sample text pairs and failures. It never serializes image
pixels or changes source DATA files. An identical existing report is reused;
choose a new report path for a different configuration or subset.

Local tests cover both variants, deterministic ranges/order independence, exact
text preservation, separate tokenizer calls, special tokens, first-target/EOS
supervision, padding (including shared PAD/EOS IDs), image conversion and safe
paths, lazy DATA loading, checksums and the diagnostic integration path. Real
rama DATA artifacts have not been copied or regenerated during implementation.
