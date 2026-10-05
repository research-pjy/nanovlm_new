"""Experimental global-image and patch-wise convolution vision encoders.

The paper describes the image as being divided into 16x16 patches and also describes two 2D convolutional layers as part of patch embedding, but does not provide sufficient implementation detail to uniquely determine whether those convolutions operate before patch extraction or independently after patch extraction. This implementation therefore treats the two interpretations as experimental variants.

Neither variant may be described as the authors' exact implementation unless
additional implementation evidence is found and documented.
"""
import torch
from torch import nn

from .config import VisionConfig


class ChannelLayerNorm(nn.Module):
    """Normalize channels independently at each pixel, never across patch regions."""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ConvStem(nn.Module):
    """Two spatially preserving convolutions; no pooling, patching, or stride reduction."""
    def __init__(self, config: VisionConfig):
        super().__init__()
        layers = []
        incoming = config.in_channels
        for outgoing in config.conv_channels:
            layers.extend([
                nn.Conv2d(incoming, outgoing, config.conv_kernel_size,
                          stride=1, padding=config.conv_kernel_size // 2, bias=True),
                ChannelLayerNorm(outgoing), nn.ReLU(),
            ])
            incoming = outgoing
        self.layers = nn.Sequential(*layers)

    def forward(self, images):
        return self.layers(images)


class GlobalConvPatchEmbedding(nn.Module):
    """Convolve the full image first, then form/project non-overlapping patches."""
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.stem = ConvStem(config)
        self.unfold = nn.Unfold(kernel_size=config.patch_size, stride=config.patch_size)
        self.projection = nn.Linear(config.conv_channels[-1] * config.patch_size**2, config.embed_dim)

    def _validate_images(self, images):
        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise ValueError('Expected a floating image tensor [batch, channels, height, width]')
        expected = (self.config.in_channels, self.config.image_size, self.config.image_size)
        if tuple(images.shape[1:]) != expected or images.shape[0] < 1:
            raise ValueError(f'Expected nonempty batch with image shape {expected}; got {tuple(images.shape)}')
        if not images.is_floating_point():
            raise ValueError('Images must be floating tensors; RGB conversion/scaling is external')

    def forward(self, images):
        self._validate_images(images)
        features = self.stem(images)  # [B, C, 224, 224], BEFORE any patch extraction.
        patches = self.unfold(features).transpose(1, 2)  # [B, 196, C*16*16], row-major grid.
        return self.projection(patches)


class PatchConvPatchEmbedding(GlobalConvPatchEmbedding):
    """Extract patches first; reuse one stem independently for every patch.

    Inherits the identical parameter layout and initialization from Variant A.
    Folding patches into the batch axis shares weights without spatial mixing.
    """
    def forward(self, images):
        self._validate_images(images)
        batch = images.shape[0]
        size = self.config.patch_size
        patches = self.unfold(images).transpose(1, 2)
        patches = patches.reshape(batch * self.config.num_patches,
                                  self.config.in_channels, size, size)
        features = self.stem(patches)
        flattened = features.reshape(batch, self.config.num_patches, -1)
        return self.projection(flattened)


class VisionTransformerBlock(nn.Module):
    """Pre-LayerNorm residual block with bidirectional self-attention."""
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.attention = nn.MultiheadAttention(config.embed_dim, config.num_heads,
                                              dropout=config.dropout, batch_first=True)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.norm2 = nn.LayerNorm(config.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim * config.mlp_ratio), nn.GELU(),
            nn.Dropout(config.dropout), nn.Linear(config.embed_dim * config.mlp_ratio, config.embed_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        normalized = self.norm1(x)
        attention, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        x = x + self.attention_dropout(attention)
        return x + self.mlp(self.norm2(x))


class VisionEncoder(nn.Module):
    label = 'Experimental Variant A: Global-Image Convolution'

    def __init__(self, config: VisionConfig | None = None, *, conv_strategy: str | None = None):
        super().__init__()
        if config is None:
            config = VisionConfig(conv_strategy='global' if conv_strategy is None else conv_strategy)
        elif conv_strategy is not None and conv_strategy != config.conv_strategy:
            raise ValueError('conv_strategy disagrees with the supplied config')
        self.config = config
        if config.conv_strategy == 'global':
            self.label = 'Experimental Variant A: Global-Image Convolution'
            self.patch_embedding = GlobalConvPatchEmbedding(config)
        else:
            self.label = 'Experimental Variant B: Patch-Wise Convolution'
            self.patch_embedding = PatchConvPatchEmbedding(config)
        self.cls_token = nn.Parameter(torch.empty(1, 1, config.embed_dim))
        self.position_embedding = nn.Parameter(torch.empty(1, config.num_patches + 1, config.embed_dim))
        self.embedding_norm = nn.LayerNorm(config.embed_dim)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(VisionTransformerBlock(config) for _ in range(config.depth))
        self.final_norm = nn.LayerNorm(config.embed_dim)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)

    def forward(self, images):
        patches = self.patch_embedding(images)
        cls = self.cls_token.to(dtype=patches.dtype).expand(patches.shape[0], -1, -1)
        tokens = torch.cat([cls, patches], dim=1)
        tokens = self.embedding_dropout(self.embedding_norm(tokens + self.position_embedding.to(dtype=tokens.dtype)))
        for block in self.blocks:
            tokens = block(tokens)
        return self.final_norm(tokens)  # [B, 197, D]; token 0 is CLS, tokens 1: are patches.
