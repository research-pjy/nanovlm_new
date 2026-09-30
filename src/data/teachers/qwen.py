"""Offline Hugging Face Qwen3-VL teacher. Heavy imports happen only on rama."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil

from .base import TeacherOutput


def prepare(config, output_parent):
    import torch
    from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration  # noqa: F401
    from huggingface_hub import snapshot_download

    if not torch.cuda.is_available():
        raise ValueError('CUDA GPU unavailable; activate qwen-vl on rama and check PyTorch/CUDA')
    if not torch.cuda.is_bf16_supported():
        raise ValueError('This configuration requires CUDA BF16 support')
    device = torch.cuda.get_device_name(0)
    if 'L40S' not in device:
        raise ValueError(f'Expected L40S on cuda:0, detected {device}')
    # Actually execute CUDA/BF16 kernels, rather than trusting reported versions.
    x = torch.ones((16, 16), device='cuda:0', dtype=torch.bfloat16)
    _ = x @ x
    torch.cuda.synchronize()
    del x, _
    free, total = torch.cuda.mem_get_info(0)
    if free < config['minimum_free_vram_gib'] * 2**30:
        raise ValueError(f'Only {free / 2**30:.2f} GiB VRAM free; free resources before running')
    parent = Path(output_parent).resolve()
    while not parent.exists():
        parent = parent.parent
    disk = shutil.disk_usage(parent).free
    if disk < config['minimum_free_disk_gib'] * 2**30:
        raise ValueError(f'Only {disk / 2**30:.2f} GiB output disk space free')
    model_path = Path(config['model']).expanduser()
    if not model_path.is_dir():
        model_path = Path(snapshot_download(config['model'], revision=config['revision'], local_files_only=True))
    model_config = json.loads((model_path / 'config.json').read_text())
    if model_config.get('model_type') != 'qwen3_vl':
        raise ValueError('Expected a Qwen3-VL Hugging Face checkpoint')
    # Hash local model assets so a changed checkpoint cannot silently join an old run.
    files = sorted(p for p in model_path.iterdir() if p.is_file() and p.suffix in ('.json', '.safetensors', '.jinja', '.txt', '.model'))
    if not any(p.suffix == '.safetensors' for p in files):
        raise ValueError('No cached safetensors weights found; supply the complete local model directory')
    index_path = model_path / 'model.safetensors.index.json'
    if index_path.exists():
        shards = set(json.loads(index_path.read_text())['weight_map'].values())
        if any(Path(name).name != name or not (model_path / name).is_file() for name in shards):
            raise ValueError('Incomplete or invalid cached model shards')
    asset_hash = hashlib.sha256()
    print('Fingerprinting cached teacher assets (read-only; may take a minute)', flush=True)
    for path in files:
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        asset_hash.update(f'{path.name}:{digest.hexdigest()}\n'.encode())
    runtime = {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'huggingface-hub')}
    for name in ('accelerate', 'Pillow'):
        try:
            runtime[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            runtime[name] = 'not installed'
    runtime.update(python=platform.python_version(), cuda=torch.version.cuda, gpu=device)
    identity = {'backend': 'qwen_hf', 'teacher_model': config['model'],
                'assets_sha256': asset_hash.hexdigest(), 'runtime': runtime,
                'adapter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'inference_settings': {'dtype': 'bfloat16', 'attention': 'sdpa',
                                       'device': 'cuda:0', 'do_sample': False,
                                       'num_beams': 1, 'use_cache': True,
                                       'padding_side': 'left', 'local_files_only': True}}
    print(json.dumps({'runtime': runtime, 'free_vram_gib': free / 2**30,
                      'total_vram_gib': total / 2**30, 'free_disk_gib': disk / 2**30,
                      'model_directory': str(model_path), 'assets_sha256': asset_hash.hexdigest()}, indent=2), flush=True)
    return model_path, identity


class QwenTeacher:
    def __init__(self, config, model_path, identity):
        self.config, self.path, self._identity = config, model_path, identity
        self.model = self.tokenizer = None

    @property
    def identity(self):
        return self._identity

    def _load(self):
        import torch
        from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration
        if self.model is not None:
            return
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True, padding_side='left')
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.path, local_files_only=True, dtype=torch.bfloat16,
            attn_implementation='sdpa', device_map={'': 'cuda:0'})
        self.model.eval()

    def generate_batch(self, prompts, seed):
        import torch
        from transformers import GenerationConfig, set_seed
        self._load()
        set_seed(seed)
        torch.cuda.reset_peak_memory_stats()
        messages = [[{'role': 'user', 'content': prompt}] for prompt in prompts]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, add_special_tokens=False)
        if inputs['input_ids'].shape[1] > self.config['max_input_tokens']:
            raise ValueError('Prompt exceeds max_input_tokens; no source captions were truncated')
        inputs = inputs.to('cuda:0')
        eos = self.model.generation_config.eos_token_id
        if eos is None:
            eos = self.tokenizer.eos_token_id
        if eos is None:
            raise ValueError('Teacher has no EOS token configuration')
        eos_ids = {eos} if isinstance(eos, int) else set(eos)
        settings = GenerationConfig(do_sample=False, num_beams=1,
                                    max_new_tokens=self.config['max_new_tokens'],
                                    eos_token_id=eos, pad_token_id=self.tokenizer.pad_token_id,
                                    use_cache=True)
        with torch.inference_mode():
            generated = self.model.generate(**inputs, generation_config=settings)
        continuations = generated[:, inputs['input_ids'].shape[1]:]
        decoded = self.tokenizer.batch_decode(continuations, skip_special_tokens=True,
                                              clean_up_tokenization_spaces=False)
        return [TeacherOutput(text, not any(token in eos_ids for token in tokens.tolist()))
                for text, tokens in zip(decoded, continuations)]

    def memory_stats(self):
        import torch
        free, total = torch.cuda.mem_get_info()
        return {'allocated_gib': torch.cuda.memory_allocated() / 2**30,
                'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                'reserved_gib': torch.cuda.memory_reserved() / 2**30,
                'free_gib': free / 2**30}
