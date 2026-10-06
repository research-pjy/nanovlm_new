# Causal continuation loss

`src.losses.causal.causal_cross_entropy` is independent of the vision strategy,
connector, decoder, tokenizer and teacher. It accepts text-aligned floating logits
`[B, T, V]`, int64 input IDs `[B, T]`, and a binary task loss mask `[B, T]` on the
same device. The optional binary attention mask has the same shape.

```python
from src.losses.causal import causal_cross_entropy

loss = causal_cross_entropy(
    logits,
    input_ids,
    loss_mask,
    attention_mask=attention_mask,
)
loss.backward()
```

Convert the TASK collator's lists to tensors at the training boundary. Pass the
full prompt-plus-target sequence, not already-shifted labels. The loss pairs
`logits[:, :-1]` with `input_ids[:, 1:]` and `loss_mask[:, 1:]` exactly once.
For `[prompt_0, prompt_1, target_0, target_1]` and mask `[0, 0, 1, 1]`, predictions
at positions 1 and 2 supervise target_0 and target_1. Masking prompt labels does
not stop gradients flowing through their representations as context.

Default reduction is the sum of supervised token cross entropies divided by the
number of supervised tokens across the batch. `reduction='sum'` is also supported.
For future gradient accumulation or distributed training, normalize sums using
the total supervised-token count across the intended effective batch; averaging
unequal microbatch means would change this objective.

Only selected predictions enter cross entropy. Masked labels may use sentinel IDs;
supervised labels must be valid vocabulary IDs. FP16/BF16 selected logits are
promoted to FP32 for the loss; autograd remains connected. Padding is determined
by masks, not a token ID, so a genuine EOS remains supervised even if EOS equals
PAD. The optional attention mask rejects supervision on padding or immediately
after a padded position. It does not itself create causal attention.

Position zero must be masked because there is no preceding logit in this API.
Use a BOS position if supervision of the first text token is required. Empty
supervision, malformed shapes/masks, out-of-range supervised IDs and non-finite
supervised logits fail clearly rather than returning a misleading zero loss.

The future model must prevent future-token attention and provide logits aligned
to text positions. If its internal sequence includes visual tokens, the model
integration layer must align the text logits before invoking this function. The
loss does not infer image-token offsets or know model architecture. This phase
implements the objective, not a training loop or a complete language model.

## Verify on rama

After reviewing, committing, pushing and pulling the commit:

```bash
conda activate qwen-vl
python -m unittest discover -s tests -p 'test_causal_loss.py' -v
python -m unittest discover -s tests -v
```

The seven new tests include a hand-computed loss, shift and gradient checks,
mask invariance, TASK collation with shared EOS/PAD ID, input rejection,
low-precision loss and CUDA BF16 backward execution. On rama with CUDA available,
all seven should run without skips. No dataset generation or model training is
required. Local tests skip numerical checks if PyTorch is unavailable.
