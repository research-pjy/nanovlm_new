# 3L — Complete NanoVLM assembly and acceptance

`src.models.nanovlm.NanoVLM` composes the verified vision encoder, CLS connector
and causal decoder. It does not load a teacher, preprocess pixels internally,
retokenize text, compute loss or run an optimizer.

```python
from src.models.nanovlm import NanoVLM
from src.models.config import ModelConfig
from src.preprocessing.images import ImageConfig
from src.tokenization.student import StudentTokenizer
from src.losses.causal import causal_cross_entropy

config = ModelConfig.load('data/processed/model.debug.tokenized.json')
tokenizer = StudentTokenizer('data/processed/student_tokenizer')
pixel_config = ImageConfig()  # Or load the saved JSON configuration.
model = NanoVLM.from_artifacts(config, tokenizer, pixel_config)
logits = model(images, input_ids, attention_mask)
loss = causal_cross_entropy(logits, input_ids, loss_mask,
                            attention_mask=attention_mask)
```

Images are processed [B,3,H,W]; input IDs and required attention mask are [B,T].
Output is [B,T,V]. Vision -> contextualized CLS -> one projected visual prefix ->
causal decoder is entirely internal. The decoder removes the visual output slot;
BOS remains at text index zero. The existing loss shifts once, supervising the
first target from the final prompt position. Neither TASK nor LOSS needs image
offsets. All input/model tensors must share the execution device.

`from_artifacts` rejects mismatched vocabulary, special IDs, image size or RGB
channel configuration. The numeric constructor `NanoVLM(config)` is available
for synthetic unit tests; it cannot export an artifact checkpoint without binding
tokenizer/preprocessing identities. Use from_artifacts for actual DATA.

## Model-only checkpoints

`save_checkpoint(path, provenance=...)` atomically publishes a model artifact
without replacing differing files. It records resolved configuration (including
strategy), strict state_dict, component/total/trainable counts, tokenizer hash
and full manifest, preprocessing settings/hash, Pillow and tokenizers versions,
and caller provenance. Acceptance artifacts include seed 42, dataset hashes and
Git revision. They contain random initialization, not a trained model.

`load_checkpoint(..., tokenizer=..., preprocessing=..., expected_config=...)`
uses weights_only loading onto CPU, validates metadata and exact expected config,
loads strict weights and returns eval mode. Wrong strategy is rejected even when
weight shapes match. Same-sized but different tokenizers are rejected by identity.
Changed preprocessing/runtime versions fail explicitly rather than silently
changing behavior. Assets must exist locally; no downloads or teacher calls occur.
The tokenizer and images remain separate artifacts; a model checkpoint does not
embed raw training data or tokenizer JSON. Preserve those assets alongside it.

These are not resumable training checkpoints: optimizer/scheduler/RNG/data-order
state belongs to Phase 5. All current parameters are trainable. Explicit freezing
and its checkpoint/resume policy are not introduced by this phase.

## Verify on rama after local review/commit/push/pull

```bash
conda activate qwen-vl
python -m unittest discover -s tests -p 'test_nanovlm.py' -v
python -m unittest discover -s tests -v

python -m src.models.check_nanovlm \
  --dataset-dir data/processed/final \
  --data-root /home/jayanth/datasets/coco \
  --tokenizer-dir data/processed/student_tokenizer \
  --model-config data/processed/model.debug.tokenized.json \
  --preprocessing-config configs/image_preprocessing.json \
  --num-images 2 --device cpu \
  --output-dir data/processed/nanovlm_acceptance_cpu

python -m src.models.check_nanovlm \
  --dataset-dir data/processed/final \
  --data-root /home/jayanth/datasets/coco \
  --tokenizer-dir data/processed/student_tokenizer \
  --model-config data/processed/model.debug.tokenized.json \
  --preprocessing-config configs/image_preprocessing.json \
  --num-images 2 --device cuda --bf16 \
  --output-dir data/processed/nanovlm_acceptance_cuda
```

Four assembly tests should run without skips on rama. Local execution without
torch/tokenizers skips the numerical integration tests. The fixture test uses
saved image files, a small train-only BPE, final-DATA-format records and the real
TASK/preprocessing/model/loss path. The CLI additionally checks actual COCO data.

Each CLI run uses the same first two ascending training image IDs for SHORT and
LONG, verifies the training corpus against tokenizer provenance, resets seed 42
before each branch construction and checks exact matching initial state tensors.
The paired resolved settings differ only in convolution_strategy. It measures
component, total and trainable counts and reports signed Patch-minus-Global
counts; no components are resized to disguise differences.

Behavior checks run in eval FP32: text shape, finite values, future-token isolation,
masked-padding content and unpadded-example invariance, and changes to valid
text logits when image input changes. Backward checks run in train mode with
requested BF16 autocast, comparing loss against manually selected shifted labels
and requiring finite gradients and nonzero aggregate gradients in every component.
Both variants are saved/reloaded and checked for identical eval predictions.

Outputs: global.pt, patch.pt and report.json inside each new output directory.
The report records image IDs, seed, resolved configs, dataset identity, Git
revision, runtime, counts, task shapes/losses/gradients, conditioning effects,
reload results and GPU peak allocation where applicable. CUDA preflight checks
L40S/BF16 availability, free VRAM and disk. There are ZERO optimizer steps.
A new directory is required for reruns; retain failed-run artifacts for diagnosis.
Missing report.json or any command failure means acceptance is incomplete.

For the debug configuration with vocabulary 4,096, analytical counts are:
vision 749,984; connector 16,512; decoder 1,515,136; complete model 2,281,632.
The acceptance report must verify these by counting actual parameters, identically
for both variants. This debug model is not labeled the paper's Mini.

Only after unit tests AND both real-data acceptance runs pass on rama is the
MODEL bucket ready for LEARNING. These checks verify implementation, not learned
language quality. The tiny-data overfitting gate remains Phase 5.
