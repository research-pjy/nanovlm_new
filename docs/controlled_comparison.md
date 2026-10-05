# First controlled comparison

Compare **Experimental Variant A: Global-Image Convolution** with
**Experimental Variant B: Patch-Wise Convolution**. Only convolution placement
may change. Neither variant is claimed to reproduce the paper exactly.

## Shared settings

Both runs must use identical:

- Input resolution and image preprocessing.
- Patch size, convolution widths, kernels, strides, activations and normalization.
- Number of transformer blocks, transformer dimensions, attention heads, MLP
  expansion, dropout, positional embeddings and CLS handling.
- Connector and decoder, including initialization.
- Tokenizer, vocabulary and special-token policy.
- Task variant, prefix construction and loss-mask policy.
- Dataset artifacts, description versions, filtering and actual train/validation/
  test image IDs. The held-out test set remains separate.
- Seed **42**, initial shared weights and data ordering.
- Loss and its reduction, optimizer and optimizer settings, learning rate and
  schedule, effective batch size, number of optimizer updates and epochs.
- Precision and checkpoint/evaluation policies.

For each task comparison, use the same SHORT or LONG task on both sides.
Do not compare Global-Conv SHORT against Patch-Conv LONG as a placement-only
experiment. Preserve the existing 25,200 / 2,800 / 100 image assignments.

The shipped vision configs differ only in `conv_strategy`. Both use resolution
224, patch size 16, convolution widths 16/32, kernel size 3, embedding dimension
512, three transformer blocks, eight heads, MLP expansion four and dropout 0.1.
The patch branch shares its convolutional weights across patches. Padding at
patch boundaries versus full-image boundaries is a consequence of placement.

Reset seed 42 before constructing each branch or explicitly copy matching initial
weights. Save configuration alongside checkpoints: parameter names and shapes
alone do not identify the branch. A shared seed does not imply bitwise identical
outputs across different computational paths.

## Recorded parameter counts

User-provided L40S smoke-check results for the shipped configurations report:

- Global-Conv vision encoder: **13,760,576** parameters.
- Patch-Conv vision encoder: **13,760,576** parameters.
- Difference (Patch minus Global): **0** parameters.

These are vision-encoder counts, not complete VLM counts. Source reports on rama:
`data/processed/vision_global_check.json` and
`data/processed/vision_patch_check.json`. Both passed forward/backward checks;
all 16 vision tests passed on rama after Variant B was added.

Once connector and decoder exist, record total and trainable counts for each
component and the complete model for both branches. If a future implementation
produces unequal counts, report the absolute difference and its cause. Do not
alter widths, depths, vocabulary or any other component to disguise it.

## Before the first training comparison

Connector, decoder, vocabulary, loss, optimizer, learning rate, training schedule
and epoch count have not yet been selected. They must be specified once and shared
by both branches when their buckets are implemented; no defaults are implied here.

Save resolved run configurations, code revision, dataset/task artifact hashes,
tokenizer identity, initialization seed and parameter counts with the results.
The future training entry point must check that paired configurations differ only
in convolution strategy and bookkeeping fields such as output paths and run names.
Record runtime versions and hardware as well. Resource tuning must not silently
change the scientific settings or effective batch size between runs.

Current automated checks verify shipped vision-config equality except for strategy
and matching encoder parameter layouts/initialization. They do not yet enforce
the full training contract because the remaining components are not implemented.
