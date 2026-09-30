"""Read-only checks for the existing selected COCO download (Phase 1A)."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys


def verify(config, progress=print):
    from PIL import Image

    root = Path(config['data_root']).expanduser()
    if not root.is_absolute():
        raise ValueError('data_root must be an absolute path')
    for key in ('number_of_images', 'random_seed'):
        if type(config[key]) is not int:
            raise ValueError(f'{key} must be an integer')
    if config['number_of_images'] <= 0:
        raise ValueError('number_of_images must be positive')
    manifest_path = root / 'image_selection.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['seed'] != config['random_seed']:
        raise ValueError('Existing selection seed differs from configuration; selection was not changed')
    if manifest['num_images'] != config['number_of_images']:
        raise ValueError('Existing selection size differs from configuration; selection was not changed')
    groups = ('train', 'val', 'held_out')
    if len(manifest['train']) + len(manifest['val']) != config['number_of_images']:
        raise ValueError('Manifest train/val counts do not match number_of_images')
    if len(manifest['held_out']) != manifest['num_held_out'] or not manifest['held_out']:
        raise ValueError('Missing or inconsistent held-out set')

    metadata = {}
    captions = {}
    source_splits = manifest['splits']
    if not source_splits or len(set(source_splits)) != len(source_splits):
        raise ValueError('Invalid source split list')
    for split in source_splits:
        if split not in ('train2017', 'val2017'):
            raise ValueError(f'Unsupported COCO source split: {split}')
        annotation_path = root / 'annotations' / f'captions_{split}.json'
        progress(f'Reading {annotation_path.name}', flush=True)
        annotation = json.loads(annotation_path.read_text(encoding='utf-8'))
        for record in annotation['images']:
            image_id = record['id']
            if image_id in metadata:
                raise ValueError(f'Duplicate annotation image ID: {image_id}')
            metadata[image_id] = {**record, 'split': split}
        for record in annotation['annotations']:
            captions.setdefault(record['image_id'], []).append(record['caption'])

    errors = []
    seen_ids = set()
    seen_paths = set()
    modes, formats = Counter(), Counter()
    checked = 0
    total = sum(len(manifest[g]) for g in groups)
    for group in groups:
        for index, entry in enumerate(manifest[group]):
            label = f'{group}[{index}]'
            try:
                image_id = entry['image_id']
                label += f' image_id={image_id}'
                if image_id in seen_ids:
                    raise ValueError('Repeated image ID within or across assignments')
                seen_ids.add(image_id)
                meta = metadata[image_id]
                split, filename = entry['split'], entry['file_name']
                if split != meta['split'] or filename != meta['file_name']:
                    raise ValueError('Source split or filename disagrees with annotations')
                if Path(filename).name != filename or '\\' in filename:
                    raise ValueError('Image filename must not contain a directory')
                path = root / 'images' / split / filename
                resolved = path.resolve()
                if not resolved.is_relative_to(root.resolve()):
                    raise ValueError('Image path escapes data_root')
                if resolved in seen_paths:
                    raise ValueError('Repeated image path')
                seen_paths.add(resolved)
                source_captions = captions[image_id]
                if len(source_captions) < 5 or any(
                    not isinstance(c, str) or not c.strip() for c in source_captions
                ):
                    raise ValueError('Expected at least five nonempty source captions')
                if entry['coco_captions'] != source_captions[:5]:
                    raise ValueError('Saved captions disagree with source annotations')
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    image.load()
                    if image.size != (meta['width'], meta['height']):
                        raise ValueError('Decoded dimensions disagree with annotations')
                    modes[image.mode] += 1
                    formats[image.format] += 1
                checked += 1
            except (OSError, ValueError, KeyError, TypeError) as exc:
                errors.append(f'{label}: {exc}')
            processed = checked + len(errors)
            if processed % 2000 == 0 or processed == total:
                progress(f'Checked {processed}/{total} selected images', flush=True)
    return {
        'ok': not errors,
        'random_seed': manifest['seed'],
        'assignments': {g: len(manifest[g]) for g in groups},
        'images_verified': checked,
        'image_modes': dict(modes),
        'image_formats': dict(formats),
        'errors': errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text(encoding='utf-8'))
        report = verify(config)
    except ImportError:
        print('ERROR: Pillow is required for image verification.', file=sys.stderr)
        return 1
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f'ERROR: Cannot verify existing dataset: {exc}', file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
