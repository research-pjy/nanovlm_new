# 3I — Shared image preprocessing

`src.preprocessing.images.ImagePreprocessor` implements the approved deterministic
pixel policy. It takes a filesystem path or Pillow image and returns a contiguous
CPU float32 tensor [3,H,W]. Its batch method stacks outputs to [B,3,H,W]. It does
not inspect convolution strategy, tokenize text, use a teacher or modify images.

The order is: decode -> EXIF transpose -> RGB -> bicubic stretch resize -> CHW
float32 / 255 -> channel normalization. Default mean and std are each
(0.5,0.5,0.5), producing values in [-1,1]. Grayscale channels are replicated by
Pillow RGB conversion. RGBA conversion discards alpha without background
compositing; RGB values are retained even for transparent pixels. Orientation is
applied to a copy, preserving original image objects and source files.

Stretch resize keeps the full frame but distorts aspect ratio. There is no crop,
random augmentation, or split-specific transform. `configs/image_preprocessing.json`
records the policy, image_size and mean/std. The size and normalization values are
configurable; currently only the approved bicubic/stretch/EXIF policy is accepted.
A later policy change must be explicit and shared between branches. Bad images
fail with context instead of being skipped. No torchvision dependency is needed.

## Use with TASK without changing its interface

```python
import torch
from src.preprocessing.images import ImageConfig, ImagePreprocessor
from src.tasks.builder import TaskBuilder, collate

transform = ImagePreprocessor(ImageConfig(image_size=model_config.image_size))
builder = TaskBuilder(tokenizer, data_root,
                      bos_token_id=tokenizer.bos_token_id,
                      eos_token_id=tokenizer.eos_token_id,
                      image_loader=transform)
# After building examples using the existing TaskDataset:
batch = collate(examples, pad_token_id=tokenizer.pad_token_id)
images = torch.stack(batch['images'])
```

Inject the adapter as image_loader so it sees the original file and EXIF before
RGB conversion. Do not run the existing default RGB loader first: some source
mode/metadata information may then be lost. TASK still accepts any loader and
has no model-specific dependency. Device transfer and BF16 autocast remain at
the execution boundary; preprocessing always returns float32 CPU tensors.

## Verify on rama

After review, commit, push locally and pull on rama:

```bash
conda activate qwen-vl
python -m unittest discover -s tests -p 'test_image_preprocessing.py' -v
python -m src.preprocessing.check_images \
  --dataset-dir data/processed/final \
  --data-root /home/jayanth/datasets/coco \
  --config configs/image_preprocessing.json \
  --model-config data/processed/model.debug.tokenized.json \
  --num-images 4 \
  --report data/processed/image_preprocessing_check.json
```

Five tests should run without skips. The CPU smoke check reads the first four
training records in ascending image-ID order after verifying the training file
checksum/count/IDs. It checks source file hashes before/after, exact repeated
pixel tensors and finite encoder outputs. It feeds the same batch tensor to both
strategies, checking that neither changes it. Expect input [4,3,224,224] and each
output [4,197,128]. Outputs need not match across strategies.

The report records source hashes, selected IDs, transform and model configuration,
Pillow/PyTorch versions, shape, dtype, pixel range and repeatability. Preserve it
with later experiment metadata. An existing differing report is not overwritten;
use a new path after an intentional configuration/runtime change. This is a small
real-image smoke check, not a full dataset image-integrity scan or a training run.
