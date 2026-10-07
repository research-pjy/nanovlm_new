"""Tiny-data learning gate; deliberately not a full-training configuration."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class LearningConfig:
    seed: int = 42
    train_images: int = 10
    validation_images: int = 10
    batch_size: int = 10
    epochs: int = 300
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    gradient_clip: float = 1.0
    checkpoint_every: int = 10
    required_relative_reduction: float = 0.8
    maximum_final_loss: float = 1.0
    scheduler: str = 'constant'

    def __post_init__(self):
        for name in ('seed', 'train_images', 'validation_images', 'batch_size', 'epochs', 'checkpoint_every'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.seed != 42 or self.train_images > 20 or self.validation_images > 20:
            raise ValueError('Tiny gate requires seed 42 and at most 20 images per split')
        for name in ('learning_rate', 'weight_decay', 'gradient_clip', 'required_relative_reduction', 'maximum_final_loss'):
            value = getattr(self, name)
            if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if self.learning_rate <= 0 or self.gradient_clip <= 0 or self.maximum_final_loss <= 0:
            raise ValueError('Learning rate, gradient clip and loss threshold must be positive')
        if not 0 < self.required_relative_reduction < 1 or self.scheduler != 'constant':
            raise ValueError('Require reduction in (0,1) and constant scheduler')

    def to_dict(self):
        return asdict(self)


def gate(initial, final, config):
    return {variant: (math.isfinite(final[variant]) and
                     final[variant] <= config.maximum_final_loss and
                     final[variant] <= initial[variant]*(1-config.required_relative_reduction))
            for variant in ('short', 'long')}
