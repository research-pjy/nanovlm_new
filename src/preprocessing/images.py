"""Shared deterministic pixel adapter; no vision-strategy or TASK dependency."""
from dataclasses import asdict, dataclass
import math
from pathlib import Path


@dataclass(frozen=True)
class ImageConfig:
    image_size: int = 224
    mean: tuple = (0.5, 0.5, 0.5)
    std: tuple = (0.5, 0.5, 0.5)
    interpolation: str = 'bicubic'
    resize_policy: str = 'stretch'
    apply_exif_orientation: bool = True

    def __post_init__(self):
        if type(self.image_size) is not int or self.image_size <= 0:
            raise ValueError('image_size must be a positive integer')
        for name in ('mean', 'std'):
            values = getattr(self, name)
            if not isinstance(values, tuple) or len(values) != 3 or any(
                    type(v) not in (int, float) or not math.isfinite(v) for v in values):
                raise ValueError(f'{name} requires three finite numbers')
        if any(v <= 0 for v in self.std):
            raise ValueError('std values must be positive')
        if (self.interpolation != 'bicubic' or self.resize_policy != 'stretch'
                or self.apply_exif_orientation is not True):
            raise ValueError('Approved transform requires bicubic stretch and EXIF orientation')

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        for name in ('mean', 'std'):
            if name in values:
                values[name] = tuple(values[name])
        return cls(**values)

    def to_dict(self):
        return asdict(self)


class ImagePreprocessor:
    """Callable PIL/path -> CPU float32 [3,H,W]; injectable as TASK image_loader.

    RGB conversion drops alpha without compositing. Stretch resizing preserves
    the full frame but changes aspect ratio. No crop or random augmentation.
    """
    def __init__(self, config=ImageConfig()):
        self.config = config

    def __call__(self, source):
        import torch
        from PIL import Image, ImageOps
        try:
            if isinstance(source, (str, Path)):
                with Image.open(source) as original:
                    original.load()
                    image = ImageOps.exif_transpose(original).convert('RGB')
            elif isinstance(source, Image.Image):
                # Work on a copy: never remove EXIF or alter pixels on caller's image.
                image = ImageOps.exif_transpose(source.copy()).convert('RGB')
            else:
                raise ValueError('Expected an image path or Pillow image')
            image = image.resize((self.config.image_size, self.config.image_size),
                                 resample=Image.Resampling.BICUBIC)
            pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
            tensor = pixels.reshape(self.config.image_size, self.config.image_size, 3)
            tensor = tensor.permute(2, 0, 1).to(dtype=torch.float32).div_(255).contiguous()
            mean = torch.tensor(self.config.mean, dtype=torch.float32)[:, None, None]
            std = torch.tensor(self.config.std, dtype=torch.float32)[:, None, None]
            tensor = (tensor - mean) / std
            if not torch.isfinite(tensor).all():
                raise ValueError('Non-finite normalized pixels')
            return tensor
        except (OSError, SyntaxError, ValueError) as exc:
            raise ValueError(f'Image preprocessing failed for {source!s}: {exc}') from exc

    def batch(self, sources):
        import torch
        tensors = [self(source) for source in sources]
        if not tensors:
            raise ValueError('Cannot preprocess an empty batch')
        return torch.stack(tensors)
