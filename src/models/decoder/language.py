"""Scratch causal decoder conditioned on one projected image token."""
import torch
from torch import nn
from src.models.config import ModelConfig


class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.embedding_dimension
        self.norm1 = nn.LayerNorm(width, eps=1e-5)
        self.attention = nn.MultiheadAttention(width, config.attention_heads,
                                               dropout=config.dropout, batch_first=True)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.norm2 = nn.LayerNorm(width, eps=1e-5)
        self.mlp = nn.Sequential(nn.Linear(width, width*config.mlp_ratio), nn.GELU(),
                                 nn.Dropout(config.dropout), nn.Linear(width*config.mlp_ratio, width),
                                 nn.Dropout(config.dropout))

    def forward(self, x, causal_mask, padding_mask):
        normalized = self.norm1(x)
        attended, _ = self.attention(normalized, normalized, normalized,
                                    attn_mask=causal_mask, key_padding_mask=padding_mask,
                                    need_weights=False)
        x = x + self.attention_dropout(attended)
        return x + self.mlp(self.norm2(x))


class CausalLanguageDecoder(nn.Module):
    """(input_ids [B,T], visual_prefix [B,1,D], text mask [B,T]) -> [B,T,V].

    Positions include image at zero and text starting at one. Return only text
    logits; the external loss performs the causal label shift once. Mask uses
    1=valid, 0=padding and must be right padded with nonempty text per sample.
    PAD ID is zero under the approved student tokenizer contract.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.vocabulary_size is None or config.vocabulary_size < 3:
            raise ValueError('Resolve vocabulary_size from the student tokenizer before construction')
        self.config = config
        width = config.embedding_dimension
        self.token_embedding = nn.Embedding(config.vocabulary_size, width, padding_idx=0)
        self.position_embedding = nn.Embedding(config.max_text_tokens+1, width)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(DecoderBlock(config) for _ in range(config.decoder_layers))
        self.final_norm = nn.LayerNorm(width, eps=1e-5)
        self.lm_head = nn.Linear(width, config.vocabulary_size, bias=True)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=0.02)
                if isinstance(module, nn.Linear):
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                nn.init.normal_(module.in_proj_weight, std=0.02)
                nn.init.zeros_(module.in_proj_bias)
        with torch.no_grad():
            self.token_embedding.weight[0].zero_()

    def forward(self, input_ids, visual_prefix, attention_mask=None):
        if (not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2
                or input_ids.dtype != torch.long):
            raise ValueError('input_ids must be int64 [B,T]')
        batch, length = input_ids.shape
        if batch < 1 or not 1 <= length <= self.config.max_text_tokens:
            raise ValueError('Empty batch/text or text context limit exceeded; no truncation allowed')
        if (not isinstance(visual_prefix, torch.Tensor)
                or tuple(visual_prefix.shape) != (batch, 1, self.config.embedding_dimension)
                or not visual_prefix.is_floating_point()):
            raise ValueError('visual_prefix must be floating [B,1,embedding_dimension]')
        if input_ids.device != self.token_embedding.weight.device or visual_prefix.device != input_ids.device:
            raise ValueError('Inputs and decoder must share a device')
        if not torch.isfinite(visual_prefix).all():
            raise ValueError('visual_prefix must be finite')
        if torch.any((input_ids < 0) | (input_ids >= self.config.vocabulary_size)):
            raise ValueError('Input token ID outside vocabulary')
        if attention_mask is None:
            attention_mask = input_ids != 0
        if (not isinstance(attention_mask, torch.Tensor) or attention_mask.shape != input_ids.shape
                or attention_mask.device != input_ids.device
                or not torch.all((attention_mask == 0) | (attention_mask == 1))):
            raise ValueError('attention_mask must be binary [B,T] on the input device')
        valid = attention_mask.bool()
        if not valid[:, 0].all() or torch.any(valid[:, 1:] & ~valid[:, :-1]):
            raise ValueError('Require nonempty, right-padded text per sample')
        if torch.any(valid & (input_ids == 0)):
            raise ValueError('PAD ID cannot be marked valid')
        text = self.token_embedding(input_ids)
        # Cast retains gradients and supports BF16 connector outputs with FP32 embeddings.
        x = torch.cat([visual_prefix.to(text.dtype), text], dim=1)
        positions = torch.arange(length+1, device=input_ids.device)
        x = self.embedding_dropout(x + self.position_embedding(positions)[None])
        causal = torch.ones(length+1, length+1, dtype=torch.bool, device=x.device).triu(1)
        padding = torch.cat([torch.zeros(batch, 1, dtype=torch.bool, device=x.device), ~valid], dim=1)
        for block in self.blocks:
            x = block(x, causal, padding)
        return self.lm_head(self.final_norm(x)[:, 1:, :])
