"""Read-only real-COCO batching and paired vision input smoke check."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from src.preprocessing.images import ImageConfig, ImagePreprocessor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--model-config', required=True)
    parser.add_argument('--num-images', type=int, default=4)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    try:
        import torch
        import PIL
        from src.models.config import ModelConfig
        from src.models.vision.encoder import VisionEncoder
        from src.tasks.builder import resolve_image
        from src.data.artifacts import publish
        if args.num_images < 1:
            raise ValueError('num-images must be positive')
        config = ImageConfig.from_dict(json.loads(Path(args.config).read_text()))
        model_config = ModelConfig.load(args.model_config)
        if model_config.image_size != config.image_size or model_config.in_channels != 3:
            raise ValueError('Preprocessing and model image configuration disagree')
        root = Path(args.dataset_dir)
        manifest_raw = (root/'manifest.json').read_bytes()
        manifest = json.loads(manifest_raw)
        raw = (root/'train.jsonl').read_bytes()
        if manifest['schema_version'] != 1 or hashlib.sha256(raw).hexdigest() != manifest['files']['train.jsonl']:
            raise ValueError('Dataset schema/checksum mismatch')
        records = [json.loads(line) for line in raw.splitlines()]
        ids = [r['image_id'] for r in records]
        if (len(records) != manifest['counts']['train'] or len(set(ids)) != len(ids)
                or any(type(i) is not int or i < 0 for i in ids)
                or any(r['split'] != 'train' for r in records) or args.num_images > len(records)):
            raise ValueError('Invalid training records or requested sample count')
        selected = sorted(records, key=lambda r: r['image_id'])[:args.num_images]
        paths = [resolve_image(Path(args.data_root), r['image_path']) for r in selected]
        hashes_before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
        transform = ImagePreprocessor(config)
        images = transform.batch(paths)
        torch.testing.assert_close(images, transform.batch(paths), rtol=0, atol=0)
        original = images.clone()
        shapes = {}
        torch.set_num_threads(2)
        for strategy in ('global', 'patch'):
            torch.manual_seed(42)
            model = VisionEncoder(replace(model_config, convolution_strategy=strategy).to_vision_config()).eval()
            with torch.no_grad():
                output = model(images)  # Exact same batch object for both branches.
            expected = (args.num_images, model.num_visual_tokens+1, model.output_dim)
            if tuple(output.shape) != expected or not torch.isfinite(output).all():
                raise ValueError('Invalid encoder output')
            torch.testing.assert_close(images, original, rtol=0, atol=0)
            shapes[strategy] = list(output.shape)
        hashes_after = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
        if hashes_before != hashes_after:
            raise ValueError('Source files changed during verification')
        report = {'ok': True, 'config': config.to_dict(), 'model_config': model_config.to_dict(),
                  'random_seed': 42, 'selection': 'first training image IDs in ascending order',
                  'image_ids': [r['image_id'] for r in selected], 'source_sha256': hashes_before,
                  'dataset_manifest_sha256': hashlib.sha256(manifest_raw).hexdigest(),
                  'pillow': PIL.__version__, 'torch': torch.__version__, 'device': 'cpu',
                  'batch_shape': list(images.shape), 'dtype': str(images.dtype),
                  'pixel_min': images.min().item(), 'pixel_max': images.max().item(),
                  'repeat_exact': True, 'source_files_unchanged': True,
                  'same_input_for_both_strategies': True, 'output_shapes': shapes}
        publish(args.report, (json.dumps(report, indent=2)+'\n').encode())
        print(json.dumps(report, indent=2))
    except (OSError, ValueError, TypeError, KeyError, ImportError, RuntimeError) as exc:
        parser.exit(1, f'ERROR: {exc}\n')


if __name__ == '__main__':
    main()
