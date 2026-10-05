"""Shared model dimensions; debug defaults are not a paper-sized reproduction."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from .vision.config import VisionConfig


@dataclass(frozen=True)
class ModelConfig:
    image_size: int = 224
    patch_size: int = 16
    vision_blocks: int = 1
    decoder_layers: int = 2
    attention_heads: int = 4
    embedding_dimension: int = 128
    image_embedding_dimension: int = 128
    dropout: float = 0.1
    convolution_strategy: str = 'global'
    in_channels: int = 3
    conv_channels: tuple[int, int] = (8, 16)
    conv_kernel_size: int = 3
    mlp_ratio: int = 4
    # No training tokenizer has been chosen. Do not invent its vocabulary size.
    vocabulary_size: int | None = None

    def __post_init__(self):
        for name in ('decoder_layers', 'embedding_dimension'):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if self.vocabulary_size is not None and (
                type(self.vocabulary_size) is not int or self.vocabulary_size <= 0):
            raise ValueError('vocabulary_size must be null or a positive integer')
        # Reuse the vision component's constraints rather than diverging from them.
        self.to_vision_config()
        if self.embedding_dimension % self.attention_heads:
            raise ValueError('embedding_dimension must be divisible by attention_heads')

    def to_vision_config(self):
        return VisionConfig(
            conv_strategy=self.convolution_strategy, image_size=self.image_size,
            patch_size=self.patch_size, in_channels=self.in_channels,
            conv_channels=self.conv_channels, conv_kernel_size=self.conv_kernel_size,
            embed_dim=self.image_embedding_dimension, depth=self.vision_blocks,
            num_heads=self.attention_heads, mlp_ratio=self.mlp_ratio, dropout=self.dropout)

    def to_dict(self):
        values = asdict(self)
        values['conv_channels'] = list(self.conv_channels)
        return values

    @classmethod
    def from_dict(cls, values):
        if not isinstance(values, dict):
            raise ValueError('Model configuration must be a JSON object')
        values = dict(values)
        if 'conv_channels' in values:
            if not isinstance(values['conv_channels'], (list, tuple)):
                raise ValueError('conv_channels must contain two channel widths')
            values['conv_channels'] = tuple(values['conv_channels'])
        try:
            return cls(**values)
        except TypeError as exc:
            raise ValueError(f'Invalid model configuration: {exc}') from exc

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))
