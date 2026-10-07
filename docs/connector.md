# 3J — Visual-textual connector

`VisualTextualConnector` implements the approved contextualized-CLS policy:

```text
vision sequence [B,1+N,Dv]
-> select sequence[:,0:1,:]
-> Linear(Dv,Dt,bias=True)
-> GELU
-> visual prefix [B,1,Dt]
```

At 224×224 with 16×16 patches, N=196. The connector accepts the complete encoder
sequence, retaining the batch and singleton visual-token dimensions. It explicitly
selects CLS; it does not average, flatten, or project the patch tokens. Those
patch tokens have already influenced contextualized CLS through encoder attention.
The future decoder will receive this one visual prefix token before text.

This is the approved interpretation of the paper's CLS summary and single-layer
GELU projector, not a claim of exact author implementation. See the evidence and
remaining ambiguities in model_design_review.md. The connector has no strategy,
teacher, tokenizer, or decoder dependency. The same instance can consume either
encoder's features. It preserves autograd and does not modify the input sequence.

```python
from src.models.connector.projector import VisualTextualConnector

connector = VisualTextualConnector(
    image_embedding_dimension=model_config.image_embedding_dimension,
    embedding_dimension=model_config.embedding_dimension,
)
visual_prefix = connector(encoder(images))
```

Both equal and unequal widths are supported. Equal widths still use a learned
projection. Weights initialize from normal(0,0.02), biases to zero, as approved.
For controlled branch initialization reset seed 42 before constructing each whole
model, or explicitly load matching shared weights. Keep configuration identical
except convolution strategy. No connector dropout or additional normalization
has been added. Output precision follows the PyTorch/autocast context.

Parameter count is Dv*Dt+Dt, all trainable: the debug 128->128 connector has
16,512 parameters for either variant. Combined with the previously verified
749,984-parameter debug encoder, this is 766,496 encoder-plus-connector parameters.
These are analytical counts for this addition, not a complete VLM count; decoder
and vocabulary-head parameters are still pending. Tests count actual instances.

## Verify on rama

After user review/commit/push and pulling on rama:

```bash
conda activate qwen-vl
python -m unittest discover -s tests -p 'test_connector.py' -v
python -m unittest discover -s tests -v
```

Five connector tests should pass without skips on the CUDA machine. They cover
same/unequal widths, actual parameter counts, matching initialization, checkpoint
reload, exact CLS selection, input preservation, gradients and invalid inputs.
Integration uses the same connector with both encoders and checks image and
convolution gradients. A separate CUDA BF16 test covers forward/backward with
unequal widths. Local numerical tests skip when PyTorch is unavailable.

The connector's direct gradient to non-CLS input slots is zero by design; the
end-to-end encoder test confirms that image patches still receive gradients via
CLS attention. Connector-only tests cannot establish decoder conditioning or
training quality; those remain 3K/3L and Phase 5 checks.
