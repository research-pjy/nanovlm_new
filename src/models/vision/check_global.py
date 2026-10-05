"""Synthetic forward/backward check, not a training run or scientific evaluation."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import shutil

from .config import VisionConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    report_path = Path(args.report)
    if report_path.exists():
        parser.error('Report already exists; choose a new report path')
    try:
        import torch
        from .encoder import VisionEncoder
        config = VisionConfig.from_dict(json.loads(Path(args.config).read_text()))
        if args.device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable; check the qwen-vl environment and GPU')
        if args.bf16 and (args.device != 'cuda' or not torch.cuda.is_bf16_supported()):
            raise ValueError('This BF16 smoke check requires a BF16-capable CUDA GPU')
        runtime = {'python': platform.python_version(), 'torch': torch.__version__,
                   'cuda': torch.version.cuda, 'gpu': None}
        if args.device == 'cuda':
            runtime['gpu'] = torch.cuda.get_device_name()
            if 'L40S' not in runtime['gpu']:
                raise ValueError(f"Expected the L40S execution environment; found {runtime['gpu']}")
            free, total = torch.cuda.mem_get_info()
            runtime.update(free_vram_gib=free / 2**30, total_vram_gib=total / 2**30)
            torch.cuda.reset_peak_memory_stats()
        runtime['free_disk_gib'] = shutil.disk_usage(Path.cwd()).free / 2**30
        print(json.dumps({'preflight': runtime}), flush=True)
        torch.manual_seed(args.seed)
        model = VisionEncoder(config).to(args.device)
        model.train()
        images = torch.randn(args.batch_size, config.in_channels, config.image_size,
                             config.image_size, device=args.device)
        with torch.autocast(device_type=args.device, dtype=torch.bfloat16, enabled=args.bf16):
            tokens = model(images)
            loss = (tokens.float() * torch.randn_like(tokens, dtype=torch.float32)).mean()
        expected_shape = (args.batch_size, config.num_patches + 1, config.embed_dim)
        if tuple(tokens.shape) != expected_shape or not torch.isfinite(tokens).all():
            raise ValueError('Incorrect output shape or non-finite output')
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise ValueError(f'Missing/non-finite gradient: {name}')
        report = {'ok': True, 'label': model.label, 'input': 'synthetic random images',
                  'random_seed': args.seed, 'config': asdict(config), 'runtime': runtime,
                  'bf16_autocast': args.bf16, 'input_shape': list(images.shape),
                  'output_shape': list(tokens.shape), 'output_dtype': str(tokens.dtype),
                  'parameters': sum(p.numel() for p in model.parameters()),
                  'all_parameter_gradients_finite': True, 'optimizer_steps': 0}
        if args.device == 'cuda':
            report['gpu'] = {
                'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
                'free_gib': torch.cuda.mem_get_info()[0] / 2**30}
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open('x') as stream:
            stream.write(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
    except (ImportError, ValueError, RuntimeError, OSError, TypeError) as exc:
        parser.exit(1, f'ERROR: {exc}\n')


if __name__ == '__main__':
    main()
