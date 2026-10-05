"""Explicit experimental choices for the standalone vision encoder."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class VisionConfig:
    conv_strategy: str = 'global'
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    conv_channels: tuple[int, int] = (16, 32)
    conv_kernel_size: int = 3
    embed_dim: int = 512
    depth: int = 3
    num_heads: int = 8
    mlp_ratio: int = 4
    dropout: float = 0.1

    def __post_init__(self):
        if self.conv_strategy not in ('global', 'patch'):
            raise ValueError('conv_strategy must be global or patch')
        for name in ('image_size','patch_size','in_channels','conv_kernel_size','embed_dim','depth','num_heads','mlp_ratio'):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if not isinstance(self.conv_channels, tuple) or len(self.conv_channels) != 2 or any(type(c) is not int or c <= 0 for c in self.conv_channels):
            raise ValueError('conv_channels must contain exactly two positive channel widths')
        if self.conv_kernel_size % 2 != 1:
            raise ValueError('conv_kernel_size must be odd for same-size symmetric padding')
        if self.image_size % self.patch_size:
            raise ValueError('image_size must be divisible by patch_size')
        if self.embed_dim % self.num_heads:
            raise ValueError('embed_dim must be divisible by num_heads')
        if type(self.dropout) not in (int,float) or not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError('dropout must be in [0, 1)')

    @property
    def num_patches(self):
        return (self.image_size // self.patch_size) ** 2

    @classmethod
    def from_dict(cls, config):
        values = dict(config)
        if 'conv_channels' in values:
            values['conv_channels'] = tuple(values['conv_channels'])
        return cls(**values)
