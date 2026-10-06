# 3H — Training tokenizer and vocabulary

Implements the approved 3G design with Hugging Face `tokenizers` byte-level BPE.
This student tokenizer is unrelated to the Qwen teacher tokenizer and replaces
neither saved descriptions nor experimental splits. No network or teacher calls.

Training reads only manifest.json and train.jsonl. It verifies the training file
hash, record count, train assignment and unique IDs, orders by image_id, then
uses short_desc followed by long_desc for each image. Validation/test files and
source captions are not used to fit vocabulary. Target vocabulary is 4,096 with
minimum frequency 2 and a complete byte alphabet. Actual size may be smaller.
There is no text normalization, inserted prefix space, or BPE dropout. The BPE
trainer does not expose a random seed; stable input ordering and alphabet, saved
library version and repeated-training artifact checks establish reproducibility
within the tested environment. Project seed 42 is recorded, not falsely passed
to an unsupported trainer option.

PAD=0, BOS=1, EOS=2 are distinct. Complete byte coverage handles ordinary Unicode
without an UNK token. Reserved spellings <pad>, <bos>, <eos> in ordinary text are
rejected. TASK inserts BOS/EOS explicitly. Prompt and target remain separately
encoded with leading whitespace preserved. Decode performs no cleanup. The loader
verifies checksums, settings, byte coverage, vocabulary IDs and library version;
it never substitutes a fallback tokenizer. Use the recorded library version if
loading fails due to a version difference; do not automatically upgrade qwen-vl.

Saved artifacts: tokenizer.json and manifest.json (published last). Manifest
records actual vocabulary, special IDs, tokenizer SHA256, library version,
training file/IDs/corpus hashes and dataset manifest hash. Re-running with the
same artifact and data loads it without fitting. A changed corpus, corrupted or
incomplete artifact fails; choose a new directory after investigating instead
of automatically deleting old files. A failed check may leave a valid tokenizer
available for diagnosis and reuse.

## Run on rama after review/commit/push/pull

```bash
conda activate qwen-vl
python -c "import tokenizers; print(tokenizers.__version__)"
python -m unittest discover -s tests -p 'test_student_tokenizer.py' -v
python -m src.tokenization.prepare \
  --dataset-dir data/processed/final \
  --output-dir data/processed/student_tokenizer \
  --model-config configs/model.debug.json \
  --resolved-model-config data/processed/model.debug.tokenized.json \
  --data-root /home/jayanth/datasets/coco \
  --report data/processed/student_tokenizer_check.json
```

If the import fails, report the missing dependency before modifying the existing
environment. Four tests should run without skips when tokenizers is installed.
The CLI checks 56,200 real TASK examples, validates image paths without decoding
pixels (3I handles pixels), and verifies exact prompt, target and combined text
recovery. It reports per-split/task counts, maximum and 95th-percentile text token
lengths and counts over 512, including BOS/EOS. Nothing is truncated. Any overflow
requires reviewing the shared decoder limit before 3K and exits nonzero. The
report's context_limit_sufficient must be true for the proposed 512 limit.
Validation/test text is used here only for validation, after fitting is complete.

Output JSON model configuration uses the ACTUAL tokenizer vocabulary size and
leaves the tracked debug template unchanged. The tokenizer path/hash remain in
its manifest and check report, to be bound into full model checkpoints in 3L.
Both convolution variants must use this same artifact and resolved vocabulary.
Outputs live under ignored data/processed; Git does not transfer these artifacts.

Re-run the same command with --check-only to require existing tokenizer assets
and perform checks without training. Identical published outputs are left intact;
differing outputs require new paths. Missing files fail clearly. No 3I–3L code is
implemented by this phase.

## Adapter usage

```python
from src.tokenization.student import StudentTokenizer
from src.tasks.builder import TaskBuilder
from src.models.config import ModelConfig

tokenizer = StudentTokenizer('data/processed/student_tokenizer')
config = tokenizer.resolve_model_config(ModelConfig.load('configs/model.debug.json'))
builder = TaskBuilder(tokenizer, data_root,
                      bos_token_id=tokenizer.bos_token_id,
                      eos_token_id=tokenizer.eos_token_id)
```

An already configured conflicting vocabulary fails rather than being overwritten.
The module imports neither torch nor a model implementation. GPU execution is
unnecessary for this phase. Tests cover train-only ingestion, reserved strings,
round trips, TASK boundary/masks, padding, vocabulary resolution, save/reload,
repeat fitting, corruption and incomplete-artifact refusal.
