"""One shared contextualized-CLS projector for either vision strategy."""
import torch
from torch import nn


class VisualTextualConnector(nn.Module):
    """[B, 1+N, Dv] -> select CLS at index 0 -> Linear + GELU -> [B,1,Dt].

    Patch tokens are not pooled or directly projected. They condition CLS inside
    the vision encoder before this module. No convolution-strategy dependency.
    """
    def __init__(self, image_embedding_dimension: int, embedding_dimension: int):
        super().__init__()
        for name, value in (('image_embedding_dimension', image_embedding_dimension),
                            ('embedding_dimension', embedding_dimension)):
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        self.input_dim = image_embedding_dimension
        self.output_dim = embedding_dimension
        self.projection = nn.Linear(self.input_dim, self.output_dim, bias=True)
        self.activation = nn.GELU()
        nn.init.normal_(self.projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.projection.bias)

    def forward(self, sequence):
        if not isinstance(sequence, torch.Tensor) or sequence.ndim != 3:
            raise ValueError('Expected vision sequence [B, 1+N, Dv]')
        if (sequence.shape[0] < 1 or sequence.shape[1] < 2
                or sequence.shape[2] != self.input_dim or not sequence.is_floating_point()):
            raise ValueError('Expected nonempty floating CLS-plus-patch sequence with configured width')
        return self.activation(self.projection(sequence[:, 0:1, :]))
