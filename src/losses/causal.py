"""Architecture-independent, continuation-masked causal language-model loss."""
import torch
from torch.nn import functional as F


def causal_cross_entropy(logits, input_ids, loss_mask, *, attention_mask=None,
                         reduction='mean'):
    """Predict input_ids[:, 1:] from logits[:, :-1], using loss_mask[:, 1:].

    logits: floating [B, T, V], aligned to the text input positions.
    input_ids: int64 [B, T]. loss_mask: binary [B, T], aligned to labels.
    Optional binary attention_mask rejects supervision on padding positions.
    Mean reduction averages over supervised tokens, not samples or padded length.
    Position zero cannot be predicted here and must be masked out (prepend BOS
    if it needs supervision). Empty supervision is an error, never a silent zero.
    This function neither constructs an attention mask nor enforces causal model
    attention. The caller must supply causally computed, text-aligned logits.
    """
    if reduction not in ('mean', 'sum'):
        raise ValueError('reduction must be mean or sum')
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3 or not logits.is_floating_point():
        raise ValueError('logits must be a floating [B, T, V] tensor')
    batch, length, vocab = logits.shape
    if batch < 1 or length < 2 or vocab < 1:
        raise ValueError('Need a nonempty batch, at least two positions and a vocabulary')
    for name, tensor in (('input_ids', input_ids), ('loss_mask', loss_mask),
                         ('attention_mask', attention_mask)):
        if tensor is None and name == 'attention_mask':
            continue
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != (batch, length):
            raise ValueError(f'{name} must have shape [B, T] matching logits')
        if tensor.device != logits.device:
            raise ValueError(f'{name} must be on the logits device')
        if name != 'input_ids' and not torch.all((tensor == 0) | (tensor == 1)):
            raise ValueError(f'{name} must be binary')
    if input_ids.dtype != torch.long:
        raise ValueError('input_ids must have dtype torch.int64')
    if torch.any(loss_mask[:, 0] != 0):
        raise ValueError('Position zero has no preceding prediction; mask it out or prepend BOS')
    selected = loss_mask[:, 1:].bool()
    if attention_mask is not None:
        if torch.any(loss_mask.bool() & ~attention_mask.bool()):
            raise ValueError('loss_mask supervises padding')
        if torch.any(selected & ~attention_mask[:, :-1].bool()):
            raise ValueError('A supervised token has a padded preceding position')
    if not selected.any():
        raise ValueError('No supervised continuation tokens in this batch')
    targets = input_ids[:, 1:][selected]
    if torch.any((targets < 0) | (targets >= vocab)):
        raise ValueError('Supervised target ID is outside the vocabulary')
    # Select before CE: masked labels (including sentinel IDs) and masked logits
    # do not enter the arithmetic. There is no special-case pad/EOS ID heuristic.
    predictions = logits[:, :-1, :][selected]
    if not torch.isfinite(predictions).all():
        raise ValueError('Supervised prediction logits must be finite')
    if predictions.dtype in (torch.float16, torch.bfloat16):
        predictions = predictions.float()
    return F.cross_entropy(predictions, targets, reduction=reduction)
