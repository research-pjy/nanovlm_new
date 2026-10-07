# Model configuration

`configs/model.debug.json` is the only model-size preset introduced now. It is a
small debugging configuration, not a claim to reproduce the paper's approximately
5M Mini, 16M Base, or 25M Large models. Those are full-model size targets; we must
count the complete implementation before assigning corresponding size labels.

The debug preset retains resolution 224 and patch size 16. It uses one vision
transformer block, two planned decoder layers, four attention heads, text embedding
width 128, image embedding width 128, dropout 0.1, convolution widths 8/16, two
3×3 convolutions and MLP expansion four. This is a separate preset: the earlier
512-wide vision configurations and their verified counts are unchanged.

`src.models.config.ModelConfig` validates and serializes the shared settings:

- `image_size`, `patch_size`: square input size and spatial patch size.
- `vision_blocks`: number of vision transformer blocks.
- `decoder_layers`: decoder depth.
- `max_text_tokens`: text context capacity (default 512), plus one image position.
- `attention_heads`: shared head count for vision and the future decoder.
- `embedding_dimension`: future language-decoder/text embedding width.
- `image_embedding_dimension`: vision output width, before any connector.
- `dropout`: shared configured dropout for the model components.
- `convolution_strategy`: `global` or `patch`.
- `in_channels`, `conv_channels`, `conv_kernel_size`, `mlp_ratio`: explicit
  architectural choices carried forward from the vision implementation.
- `vocabulary_size`: currently null; set from the selected training tokenizer
  when it is integrated, including its special tokens.

Both embedding widths must be divisible by the head count. Resolution must be
patch-divisible. Invalid values and unknown configuration keys fail explicitly.
Missing fields use the documented debug defaults; reports serialize resolved
values. Configurations can round-trip through JSON without loading PyTorch.

```python
from dataclasses import replace
from src.models.config import ModelConfig
from src.models.vision.encoder import VisionEncoder

config = ModelConfig.load('configs/model.debug.json')
global_encoder = VisionEncoder(config.to_vision_config())
patch_config = replace(config, convolution_strategy='patch')
patch_encoder = VisionEncoder(patch_config.to_vision_config())
```

For an actual controlled initialization, reset seed 42 before each construction
(or copy the shared initial weights), as specified in the comparison contract.
Both configurations must otherwise match. The debug vision output is
`[B, 197, 128]`: one CLS token and 196 visual tokens.

The vision encoder and the 3K decoder now consume their corresponding settings.
The connector is implemented separately. Complete NanoVLM assembly remains 3L.
The future connector must map image embedding width to text embedding width if
needed. The decoder must require a resolved vocabulary before construction.

## Verify on rama

After review, commit, push locally and pull on rama. From the repository root:

```bash
conda activate qwen-vl
python -m unittest discover -s tests -v

python -m src.models.vision.check_global \
  --model-config configs/model.debug.json \
  --device cuda --bf16 --batch-size 4 \
  --report data/processed/vision_debug_global_check.json

python -m src.models.vision.check_patch \
  --model-config configs/model.debug.json \
  --convolution-strategy patch \
  --device cuda --bf16 --batch-size 4 \
  --report data/processed/vision_debug_patch_check.json
```

The override changes only convolution placement; each report preserves the fully
resolved model and vision configurations and seed (default 42). Expected shape is
`[4, 197, 128]` for both. Each report records the measured vision parameter count;
`parameter_count_scope` is `vision_encoder_only` and `complete_model_parameters`
is null. Compare the measured counts, which should match, before moving forward.
These commands perform synthetic forward/backward checks, with no optimizer
updates or teacher calls. Report paths must be new.

Existing `--config configs/vision.global.json` and `--config
configs/vision.patch.json` commands remain supported. Use either `--config` or
`--model-config`, never both. Explicit placement overrides require `--model-config`;
the global and patch entry points each reject the wrong resolved strategy.

Further sizes will be added only after this configuration is integrated and
validated. Do not change other components merely to force an approximate paper
parameter target or conceal a branch count difference.
