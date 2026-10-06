# 3G — Paper evidence and proposed small-model design

Status: review complete; design proposed, awaiting user agreement before 3H–3L.
Date: 2026-10-06. No dependent modules or training implemented by this review.

## Evidence and provenance

Primary source: the supplied `nanoVLMs.pdf`, arXiv:2502.07838v2, dated
13 February 2025. Page references below are printed PDF pages (also file pages).
Local SHA256: `4456a01b5f271d491eb04634b0ce76521d899e01cc50825c3a9c79e64a4b2953`.
[Paper identity and version history](https://arxiv.org/abs/2502.07838).
The relevant text was read on pages 3–5; figures and tables on pages 4–5 were
also visually inspected. The complete extracted text was searched for tokenizer,
vocabulary, normalization, resizing, positional encoding and implementation links.

A limited paper-title/author search did not establish an author-authenticated
implementation. No public repository is used as implementation evidence here.
This is not a claim that no code exists. A similarly named project is insufficient
provenance: require an author/paper link and inspect a pinned revision before
using its behavior to resolve ambiguity.

## What the paper actually specifies

### Vision and CLS — section 2.2.1, pages 3–4; Figures 4–5, page 4

Images are 224×224, with 16×16 patches producing 196 tokens. Patch embedding
includes two convolutions, normalization, ReLU and a fully connected mapping.
A CLS token is prepended, then positions, normalization and transformer blocks
are applied. The paragraph continuing on page 4 describes aggregating CLS into a
compact image representation. This supports selecting contextualized CLS, though
it does not define an exact aggregation operator or tensor shape.

Figure 4 depicts image -> two convolutions -> flatten -> fully connected output.
Figure 5 separately depicts patch projection and the transformer. They do not
jointly establish precise patch/convolution tensor operations. Preserve our
required ambiguity statement and both existing experimental variants.

### Connector and fusion — section 2.2.2, page 4

The paper specifies one learnable projection layer followed by GELU, mapping
visual embeddings to the text embedding dimension. Visual and text embeddings
are then concatenated and sent to the decoder. This is more specific than an
arbitrary connector or cross-attention mechanism.

Not specified: exact projector input shape, visual-token count, concatenation
axis/order, projection bias, or a detailed CLS-to-projector operation. One
projected CLS prefix token concatenated along sequence length is our proposed
interpretation, not an explicitly published tensor contract.

### Decoder — sections 2.2 and 2.2.3, pages 3 and 5

Layer normalization precedes attention and MLP sublayers. The fused embeddings
receive positional embeddings and pass through causal self-attention blocks.
Final layer normalization and a linear vocabulary projection produce next-token
logits; training uses cross entropy.

Not specified: vocabulary/tokenizer, BOS/EOS/PAD policy, exact position encoding
formula, context limit, MLP expansion/activation details for the decoder,
embedding/output weight tying, projection biases, padding implementation, or
whether loss excludes prefix labels. Our accepted continuation-only loss remains
an explicit project decision.

### Sizes — Table 1, page 4; section 2.3 and Table 2, page 5

Reported Mini: vision blocks 1, decoder layers 4, heads 8, head size 12, text width
96, image width 400, approximately 5M total parameters. Base: 3, 8, 8, 16, 128,
512, approximately 16M. Large: 5, 10, 16, 12, 192, 512, approximately 25M.
The head sizes multiply to the listed text widths; do not assume that table
fully specifies vision attention internals. Dropout 0.1, image/patch sizes and
learning rate 1e-3 are reported fixed. Learning-rate adoption belongs to LEARNING.

Table 2 reports projector shares of 14%, 8%, 6%. A simple CLS linear projection
400->96 has only 38,496 parameters including bias, far below 14% of 5M. Base and
Large show a similar discrepancy under this interpretation; the Base percentages
also sum to 102%. These figures do not resolve the missing projector dimensions.
Do not invent a larger connector to match them or claim exact parameter parity
with the authors. Measure our own full model when it exists.

### Tokenization and pixel preprocessing

No tokenizer algorithm, vocabulary size/artifact, special-token IDs, pixel
normalization constants, resize interpolation or crop/aspect-ratio policy was
found in the supplied paper. Image resolution alone does not specify those
operations. Model layer normalization is not evidence of pixel normalization.

## Recommended design — project choices requiring agreement

### 3H: one student tokenizer shared by SHORT and LONG

Train byte-level BPE using the Hugging Face `tokenizers` library on TRAIN records
only, ordered by image_id, with short_desc then long_desc per image. Exclude source
captions, validation and test text from fitting. Use one frozen tokenizer for both
tasks and both convolution branches; task comparisons thus do not change vocabulary.
Propose a target vocabulary of 4,096 including special tokens, minimum frequency
2, complete byte alphabet, no lowercasing/Unicode normalization, no BPE dropout,
and no inserted prefix space. Record the actual vocabulary size, not merely the
target; inspect token-length distributions before freezing the context limit.

This is learned BPE, not the existing diagnostic byte-ID probe. The upstream
[ByteLevel BPE implementation](https://github.com/huggingface/tokenizers/blob/main/bindings/python/py_src/tokenizers/implementations/byte_level_bpe.py)
provides a byte alphabet and configurable prefix-space/normalization behavior.
Pin and record the installed version and saved tokenizer hash in 3H; this source
is library evidence only, not evidence of the paper's tokenizer.

Reserve distinct PAD=0, BOS=1, EOS=2; require these IDs in the saved artifact.
Use complete byte coverage instead of an unknown token for ordinary UTF-8 text.
Reject literal reserved special-token spellings in corpus/input text rather than
silently treating them as content or allowing ambiguous round trips.
TaskBuilder explicitly adds BOS to prompt and EOS to continuation. Continue to
encode prompt/continuation separately, preserving leading whitespace. Right-pad;
mask BOS/prompt/padding 0 and continuation/EOS 1. Decode with no whitespace cleanup.
Store corpus manifest/hash, training settings, library version and tokenizer hash.
Use seed 42 where applicable, stable corpus order and a reproducibility test;
do not assume seed alone guarantees BPE artifact identity across versions.

Why this proposal: compact student vocabulary, no imported pretrained weights or
teacher vocabulary, and byte coverage for unseen Unicode. It is not a paper claim.

### 3I: deterministic shared pixel transform

Decode with Pillow, apply EXIF orientation, convert to RGB, resize directly to
224×224 using bicubic interpolation, convert to float32 CHW in [0,1], then
normalize each channel with mean 0.5 and standard deviation 0.5 (range [-1,1]).
Use the same operation for train/validation/test and both branches, without
random augmentation or cropping initially. Record library versions and transform
settings. Direct resize preserves all image content but distorts aspect ratio;
this is an explicit simplicity tradeoff, not specified by the paper. Fail clearly
on unreadable images; do not silently skip or modify source images.

### 3J: contextualized CLS -> Linear -> GELU

Preserve encoder output [B,197,Dv]. The connector takes the full sequence,
selects `sequence[:,0:1,:]`, and applies Linear(Dv,Dt,bias=True) then GELU.
Return [B,1,Dt]. No patch averaging, flattening of all patches, extra MLP, or
cross-attention. Patch tokens still contribute through vision self-attention to
CLS; they are not directly passed as 196 decoder prefix tokens in this design.
This combines the paper's CLS summary and single-layer/GELU connector while
making the unresolved tensor layout explicit. A patch-token-prefix experiment
would be a separate later design change, shared by A/B if undertaken.

Keep Dv=Dt=128 for the accepted debug preset; equal widths mean the projector
maps rather than reduces dimension. Retain a learned projection even at equal
widths and test unequal widths. This differs from the paper's reduction sizes.

### 3K: small decoder trained from scratch

Use the existing two-layer, width-128, four-head debug settings. Sequence:
[projected image CLS, BOS, prompt tokens, continuation tokens, EOS, PAD...].
Use learned absolute positions over this combined sequence, image position 0,
BOS position 1, and subsequent text positions increasing by one. Propose maximum
512 text positions including BOS/EOS, with 513 learned position entries including
the image. Check actual tokenized lengths before freezing; fail on overflow,
never silently truncate. If inadequate, explicitly revise one shared setting.

Each block: pre-LayerNorm -> causal multi-head self-attention -> residual;
pre-LayerNorm -> Linear(Dt,4Dt) -> GELU -> Linear(4Dt,Dt) -> residual.
Use dropout 0.1 on embedding output, attention probabilities, attention residual
output and MLP intermediate/output. LayerNorm epsilon 1e-5; affine normalization;
linear biases enabled. Final LayerNorm -> untied Linear(Dt,V,bias=True).
Use a learned token embedding with padding_idx=0; zero its PAD row. No pretrained
language weights, rotary positions, cross-attention, weight tying or KV cache
in the first debug implementation. These details fill paper gaps.

Retain existing vision initialization. Proposed new linear/attention/embedding
weights: normal standard deviation 0.02; biases zero; LayerNorm weight one/bias
zero. Reset seed 42 before constructing each complete branch and test identical
shared initial weights. Do not change the verified vision to force paper counts.

## Tensor and masking contract for 3L

Inputs: images [B,3,224,224], input_ids [B,T] (BOS+prompt+target+EOS+right-PAD),
text attention mask [B,T], task loss mask [B,T]. Masks and lengths are text-only.
Require nonempty valid text and right padding for this initial implementation.
Vision -> [B,197,Dv]; connector -> [B,1,Dt]; text embedding -> [B,T,Dt];
concatenated decoder input -> [B,T+1,Dt].

Allowed attention: key position j may be attended to by query i only if j<=i
and key j is valid. The image key is always valid; text validity is the text
attention mask. Image query 0 sees only itself. A text query sees the image and
all valid text at or before itself, never future targets. Padded queries may
attend previous valid keys but their outputs are excluded from loss; no valid
query may attend padded keys. This avoids all-masked query rows. Do not replace
this with bidirectional prompt attention without a separately reviewed change.

After final normalization, select text hidden states `hidden[:,1:,:]` and apply
the vocabulary head, returning [B,T,V]. Discard only the visual position, not BOS.
Use existing causal_cross_entropy(logits,input_ids,loss_mask,attention_mask=...).
It pairs logits[:, :-1] with input_ids[:, 1:] exactly once. The logit at the last
prompt position predicts the first continuation token. No visual-token offset
enters the loss and no BOS prediction is supervised. EOS is supervised.
At generation time, the last valid text position predicts the next token; stop
at EOS or an explicit length limit. Batched padding must not select a PAD logit.

## Compatibility, departures and acceptance

Preserve all verified DATA selection/splits/descriptions and TASK/loss behavior.
The dataset uses the accepted Qwen captions-only teacher rather than GPT-4o;
length differences and the 100-image held-out split are existing project choices.
The small debug architecture is not the paper's Mini and is not a 5M target.
Keep the exact ambiguity statement in README/code and label both vision variants.
No unverified public implementation or teacher tokenizer is adopted.

Record full-model/component parameter counts in 3L; the measured debug encoder
count is 749,984 in each branch. A proposed 128->128 biased connector would add
16,512 parameters, an analytical estimate until instantiated. No measured complete
model count exists yet. Do not change components to reproduce Table 2 percentages.

Acceptance tests to add in 3H–3L: tokenizer round-trip and train-only provenance;
known preprocessing values; connector shapes/gradients at unequal widths; causal
future-token and padding invariance; CLS/image contribution to text logits;
first-target loss alignment; gradients through every component; checkpoint reload;
matching initial parameters/counts and both CPU and rama BF16 execution. A real
tiny-data overfit test remains a Phase 5 gate, not part of this documentation step.

Decision requested: accept this proposed tokenizer, pixel transform, single-CLS
prefix connector and small causal decoder as the first debug design, or identify
changes before 3H. Until agreed, configuration/code remain unchanged and 3G is
not marked accepted. Implement 3H, 3I, 3J, 3K and 3L one at a time thereafter.
