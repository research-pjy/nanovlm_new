# Experimental Variant A: Global-Image Convolution

This is a configurable experimental interpretation, **not the exact implementation
from the paper**. Only the global vision encoder is implemented at this stage.

The default flow is:

```text
floating RGB images [B, 3, 224, 224]
→ Conv2d + channel LayerNorm + ReLU [B, 16, 224, 224]
→ Conv2d + channel LayerNorm + ReLU [B, 32, 224, 224]
→ non-overlapping 16×16 patches [B, 196, 8192]
→ linear patch projection [B, 196, 512]
→ prepend learned CLS; add learned positions; LayerNorm; dropout
→ 3 bidirectional, pre-LayerNorm transformer blocks
→ final LayerNorm [B, 197, 512]
```

Here 16×16 means the spatial size of each patch, not the number of tokens.
The token grid is 14×14. Token zero is CLS; the remaining tokens follow row-major
image order. Returning both preserves later connector choices without pooling.

Both convolutions use 3×3 kernels, stride one and zero padding at the full image
boundary. There is no pooling. They run before any patch extraction, so features
can cross future patch boundaries (up to two pixels with these kernels).
Channel LayerNorm normalizes each pixel independently, avoiding additional spatial
mixing through normalization statistics. Patch extraction uses PyTorch
[Unfold](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Unfold.html).

`configs/vision.global.json` records all defaults: convolution widths 16/32,
embedding dimension 512, depth 3, eight attention heads, MLP expansion four and
dropout 0.1. These, the normalization, activations, CLS, and positional embeddings
are explicit experimental choices, not claims about unspecified paper details or
a finalized model size. Transformer MLPs use GELU. CLS and positions initialize
from a normal distribution with standard deviation 0.02; other layers use PyTorch
defaults. Future controlled comparisons must retain these choices while changing
only convolution placement.

```python
import json
from src.models.vision.config import VisionConfig
from src.models.vision.encoder import VisionEncoder

config = VisionConfig.from_dict(json.load(open('configs/vision.global.json')))
encoder = VisionEncoder(config, conv_strategy='global')
tokens = encoder(images)
```

`VisionEncoder(conv_strategy='global')` also works with defaults. Requesting
`patch` raises an explicit not-implemented error. RGB conversion, resizing,
scaling/normalization and batching are external; a shared preprocessing policy
must be chosen before training. No tokenizer, teacher, connector or decoder is
loaded by this component.

## Verify on rama

After reviewing, committing and pushing locally, pull your commit on rama and run
from `/home/jayanth/projects/nanovlm_new`:

```bash
conda activate qwen-vl
python -m unittest discover -s tests -p 'test_vision_global.py' -v
python -m src.models.vision.check_global \
  --config configs/vision.global.json \
  --device cuda --bf16 --batch-size 4 \
  --report data/processed/vision_global_check.json
```

The check verifies CUDA, L40S and BF16 availability, reports PyTorch/CUDA versions,
free GPU memory and working-directory disk space, then performs one synthetic
forward/backward pass. It reports finite gradients and peak GPU memory without
updating weights or saving a model. Expected default output is `[4, 197, 512]`.
Choose a new report path to repeat the check. This is not a training experiment or
proof of scientific performance.

Tests check actual full-image convolution order, influence across future patch
boundaries, patch ordering, gradients, checkpoint reload, and invalid inputs.
If PyTorch is absent, numerical tests explicitly skip; configuration tests still
run. The CUDA smoke check must pass on rama before this component is accepted.
