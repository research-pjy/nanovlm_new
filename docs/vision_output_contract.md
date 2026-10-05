# Vision encoder output contract

Both experimental variants expose the same `VisionEncoder` interface. Convolution
strategy is a construction setting; connector and decoder code must not branch
on it or inspect the patch-embedding implementation.

For the current configuration, floating RGB input `[B, 3, 224, 224]` produces a
single floating tensor `[B, 197, 512]` after the vision transformer and final
normalization:

- `sequence[:, 0, :]`: contextualized CLS representation `[B, 512]`.
- `sequence[:, 1:, :]`: 196 contextualized visual tokens `[B, 196, 512]`.

CLS is additional to the 196 visual tokens, not one of them. Patches retain their
14×14 row-major sequence positions. CLS is inserted before the transformer and
participates in attention alongside patch tokens. The returned CLS is therefore
not the initial learned CLS parameter.

```python
sequence = encoder(images)
cls_representation = sequence[:, 0, :]
visual_tokens = sequence[:, 1:, :]
```

This code works unchanged for either strategy. No tokens are detached or moved
to CPU. Dtype follows the execution precision; BF16 autocast does not guarantee a
BF16 output. Input preprocessing remains external and must match between variants.

For configurable resolutions and dimensions, the general contract is
`[B, 1 + encoder.num_visual_tokens, encoder.output_dim]`, where
`num_visual_tokens = (image_size // patch_size)**2`. The two properties add no
parameters and do not change existing checkpoints or numerical computation.

The future connector will choose whether to consume CLS, visual tokens, or both.
That choice must be identical in the controlled comparison. This contract does
not introduce pooling or commit to a connector architecture.

After pulling the reviewed commit on rama, run:

```bash
conda activate qwen-vl
python -m unittest discover -s tests -p 'test_vision_*.py' -v
```

Contract tests exercise an identical consumer for both branches, check the 196
visual-token plus CLS layout, and verify gradients from both representations to
the input. Existing branch tests cover row-major patch ordering and placement.
