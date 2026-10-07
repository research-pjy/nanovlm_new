# 3K — Causal language decoder

`src.models.decoder.language.CausalLanguageDecoder` implements the approved
scratch decoder. It accepts one projected image prefix [B,1,D], int64 text IDs
[B,T] and an optional binary text attention mask [B,T]. It returns logits [B,T,V]
aligned to text positions. Loss and label shifting remain external.

```python
from src.models.config import ModelConfig
from src.models.decoder.language import CausalLanguageDecoder
from src.losses.causal import causal_cross_entropy

config = ModelConfig.load('data/processed/model.debug.tokenized.json')
decoder = CausalLanguageDecoder(config)
logits = decoder(input_ids, visual_prefix, attention_mask)
loss = causal_cross_entropy(logits, input_ids, loss_mask,
                            attention_mask=attention_mask)
```

Vocabulary must be resolved from the saved student tokenizer; null vocabulary
fails. The module can accept smaller explicitly configured vocabularies in unit
tests, but it never downloads assets, calls a teacher or imports pretrained weights.
It does not inspect the convolution strategy. The approved PAD ID is zero.

## Architecture and masks

Text embeddings follow the visual token in the combined sequence. Learned
absolute positions start at image position zero and text position one. The new
`max_text_tokens` configuration field defaults to 512 (including BOS/EOS and
batch padding), giving 513 position entries. Existing saved configurations that
omit this field remain loadable with the same approved default. No tokenized
configuration or tokenizer needs regeneration.

The configured number of blocks uses pre-LayerNorm self-attention and an MLP
with configured expansion (default four), GELU and residuals. Dropout uses the
configured value on embeddings, attention probabilities, attention output and
MLP intermediate/output. LayerNorm epsilon is 1e-5. Final LayerNorm is followed
by an untied biased vocabulary projection. The visual position is removed before
that projection; BOS remains a text position. There is no KV cache, cross-attention
or full-model assembly in this phase.

Each attention query can see valid keys at or before its own position. The image
query sees only itself. All valid text queries can see the image and their past
text, never future targets. The image key is always valid, avoiding fully masked
rows. Mask value one means valid, zero means padding; require nonempty valid text
and right padding. If omitted, the mask is inferred from PAD=0. With an explicit
mask, masked positions may contain any in-vocabulary ID; they cannot influence
valid predictions. PAD cannot be marked valid. Padded-query outputs are finite
but meaningless and must stay excluded by the TASK loss mask.

A text logit at position t predicts text ID t+1 through the existing loss. The
last prompt position therefore predicts the first continuation token. No visual
offset or second label shift should be added by the caller. At later inference,
select the last VALID text position, not a padded position, for the next token.

All new embedding/linear/attention weights initialize from normal(0,0.02), biases
zero; LayerNorm weights one and biases zero. PAD embedding starts at zero and
has no embedding gradient. Input embeddings and output weights are not tied.
Reset seed 42 before each paired model construction. Vision initialization is
unchanged. BF16 visual prefixes are cast to text embedding dtype with autograd
preserved; attention/projection precision follows the caller's autocast context.

## Verify on rama

After local review/commit/push and pull on rama:

```bash
conda activate qwen-vl
python -m unittest discover -s tests -p 'test_decoder.py' -v
python -m unittest discover -s tests -v
```

Seven tests should pass without skips on rama. They check configuration bounds,
future-token invariance, prefix-only agreement, padding content/length invariance,
visual conditioning, text-logit shape, existing masked-loss integration, gradients,
configured depth/positions, untied weights, checkpoint reload, invalid inputs and
CUDA BF16 backward execution. The CPU numerical tests use small synthetic
vocabularies and visual features; no real image or teacher is needed.

Local configuration checks run without torch; numerical checks require rama.
Complete image -> encoder -> connector -> decoder integration, tokenizer binding,
full parameter counts and checkpoint metadata remain 3L. No training loop or
claim about learned language quality is introduced here.
