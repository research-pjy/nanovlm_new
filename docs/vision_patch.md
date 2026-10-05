# Experimental Variant B: Patch-Wise Convolution

This is an experimental interpretation, not an exact implementation from the paper.
The only architectural difference from Variant A is convolution placement.

Default flow:

```text
RGB floating images [B, 3, 224, 224]
→ extract non-overlapping 16×16 patches [B, 196, 768]
→ fold patches into batch [B×196, 3, 16, 16]
→ shared Conv2d + channel LayerNorm + ReLU [B×196, 16, 16, 16]
→ shared Conv2d + channel LayerNorm + ReLU [B×196, 32, 16, 16]
→ flatten each patch and project [B, 196, 512]
→ same CLS, positions, normalization, and transformer as Variant A
→ output [B, 197, 512]
```

One convolutional stem is shared across all patches and all images. Each
convolution uses stride one and zero padding at each patch boundary. Normalization
operates only over channels at each pixel. Consequently, neither convolution nor
normalization communicates between patches. The subsequent vision transformer
can attend across patches; the isolation guarantee applies before that stage.

Patch tokens remain in row-major order, with CLS prepended at index zero.
The 16×16 patch size produces a 14×14 grid (196 patches), not 256 patches.

`configs/vision.patch.json` differs from `configs/vision.global.json` only in
`conv_strategy`. All other settings, learned tensor shapes, initialization order,
and parameter names are shared. The default parameter count is 13,760,576, equal
to Variant A. Reset seed 42 before constructing each model to obtain equal initial
weights for a controlled comparison. Save the configuration with any future
checkpoint: weights alone cannot identify the convolution strategy.

```python
from src.models.vision.encoder import VisionEncoder

encoder = VisionEncoder(conv_strategy='patch')
tokens = encoder(images)
```

As with Variant A, image preprocessing is external. No teacher, tokenizer,
connector, decoder, or training pipeline is introduced here. See
[Variant A](vision_global.md) for the shared architectural defaults.

## Verify on rama

After reviewing, committing and pushing locally, pull the commit on rama. From
`/home/jayanth/projects/nanovlm_new`, run:

```bash
conda activate qwen-vl
python -m unittest discover -s tests -p 'test_vision_*.py' -v
python -m src.models.vision.check_patch \
  --config configs/vision.patch.json \
  --device cuda --bf16 --batch-size 4 \
  --report data/processed/vision_patch_check.json
```

The numerical tests cover both branches, patch isolation, full-image versus
patch convolution ordering, a patch-by-patch reference including shared gradients,
identical-patch embeddings, parameter parity, single-patch branch equivalence,
backpropagation, and checkpoint reload. These tests skip explicitly without
PyTorch; configuration checks still run locally.

The synthetic smoke check reuses Variant A's runtime checks and memory reporting,
requires the patch configuration, and performs no optimizer updates. Expected
output is `[4, 197, 512]`; the report uses the Variant B label. The report path
must be new. Forward/backward success verifies component operation, not training
quality or comparative scientific performance.
