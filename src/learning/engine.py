"""Deterministic epoch-boundary training and token-weighted validation helpers."""
import os
import random
import tempfile
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from src.tasks.builder import collate
from src.losses.causal import causal_cross_entropy


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    # Use the same deterministic attention implementation on resume.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def tensor_collate(examples):
    batch = collate(examples, 0)
    return {'images': torch.stack(batch['images']),
            **{key: torch.tensor(batch[key], dtype=torch.long)
               for key in ('input_ids', 'attention_mask', 'loss_mask')}}


def loader(examples, batch_size, seed, shuffle):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(examples, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, generator=generator, collate_fn=tensor_collate,
                      drop_last=False)


def move(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def evaluate(model, examples, batch_size, device):
    model.eval()
    total, tokens = 0., 0
    with torch.no_grad():
        for batch in loader(examples, batch_size, 42, False):
            b = move(batch, device)
            logits = model(b['images'], b['input_ids'], b['attention_mask'])
            loss = causal_cross_entropy(logits, b['input_ids'], b['loss_mask'],
                                        attention_mask=b['attention_mask'], reduction='sum')
            total += loss.item()
            tokens += int(b['loss_mask'][:, 1:].sum())
    if tokens == 0:
        raise ValueError('No evaluation target tokens')
    return total/tokens


def train_epoch(model, optimizer, examples, config, epoch, device, bf16):
    model.train()
    total, tokens, steps = 0., 0, 0
    for batch in loader(examples, config.batch_size, config.seed+epoch, True):
        b = move(batch, device)
        optimizer.zero_grad(set_to_none=True)
        count = int(b['loss_mask'][:, 1:].sum())
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=bf16):
            logits = model(b['images'], b['input_ids'], b['attention_mask'])
            summed = causal_cross_entropy(logits, b['input_ids'], b['loss_mask'],
                                          attention_mask=b['attention_mask'], reduction='sum')
            loss = summed / count
        if not torch.isfinite(loss):
            raise ValueError('Non-finite training loss; stop and debug')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip, error_if_nonfinite=True)
        optimizer.step()
        total += summed.item()
        tokens += count
        steps += 1
    return {'training_loss': total/tokens, 'optimizer_steps': steps, 'last_gradient_norm': float(norm)}


def atomic_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.checkpoint-', delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_training(path, model, optimizer, contract, epoch, steps, initial, history):
    atomic_save(path, {'schema_version': 1, 'contract': contract, 'epoch': epoch,
        'steps': steps, 'initial': initial, 'history': history,
        'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
        'scheduler': None, 'torch_rng': torch.get_rng_state(),
        'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        'python_rng': random.getstate()})


def restore_training(path, model, optimizer, contract):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state.get('schema_version') != 1 or state['contract'] != contract:
        raise ValueError('Resume contract mismatch: config/assets/code/runtime must match')
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['torch_rng'])
    if state['cuda_rng']:
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    random.setstate(state['python_rng'])
    return state
