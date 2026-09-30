"""Export portable metadata for the verified selection, retaining all COCO captions."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile

from .verify_selection import read_json, verify


def build_metadata(config):
    """Validate all inputs before returning a portable, deterministic document."""
    verification = verify(config)
    if not verification['ok']:
        raise ValueError('Selection verification failed:\n' + '\n'.join(verification['errors']))
    root = Path(config['data_root']).expanduser().resolve()
    manifest, digest = read_json(root / 'image_selection.json')
    if digest != verification['manifest_sha256']:
        raise ValueError('Selection changed during verification; retry with stable inputs')
    captions = defaultdict(list)
    for split in manifest['splits']:
        filename = f'captions_{split}.json'
        annotation, digest = read_json(root / 'annotations' / filename)
        if digest != verification['annotation_sha256'][filename]:
            raise ValueError(f'{filename} changed during verification; retry with stable inputs')
        for record in annotation['annotations']:
            captions[record['image_id']].append((record['id'], record['caption']))

    records, errors = [], []
    for assignment in ('train', 'val', 'held_out'):
        for entry in manifest[assignment]:
            image_id = entry['image_id']
            filename = entry['file_name']
            original = captions[image_id]
            texts = [text for _, text in original]
            if len(texts) < 5 or any(not isinstance(c, str) or not c.strip() for c in texts):
                errors.append(f'image_id={image_id}: expected at least five nonempty captions')
            if entry.get('coco_captions') != texts[:5]:
                errors.append(f'image_id={image_id}: saved first five captions disagree with annotations')
            if (not isinstance(filename, str) or not filename or
                    PurePosixPath(filename).name != filename or '\\' in filename or
                    filename in ('.', '..')):
                errors.append(f'image_id={image_id}: unsafe source filename {filename!r}')
                continue
            relative = PurePosixPath('images', entry['split'], filename)
            path = root / str(relative)
            if not path.resolve().is_relative_to(root) or not path.is_file():
                errors.append(f'image_id={image_id}: missing image or path outside data_root: {relative}')
            records.append({
                'image_id': image_id,
                'image_path': str(relative),
                'source_split': entry['split'],
                'assignment': assignment,
                'caption_ids': [caption_id for caption_id, _ in original],
                'captions': texts,
            })
    if errors:
        raise ValueError('Metadata validation failed (no export written):\n' + '\n'.join(errors))
    return {
        'schema_version': 1,
        'image_path_base': 'data_root',
        'random_seed': config['random_seed'],
        'source_manifest_sha256': verification['manifest_sha256'],
        'source_annotation_sha256': verification['annotation_sha256'],
        'assignments': verification['assignments'],
        'records': records,
    }


def export(config, output):
    output = Path(output).expanduser()
    root = Path(config['data_root']).expanduser().resolve()
    # Keep generated artifacts separate from the read-only source dataset.
    if output.resolve().is_relative_to(root):
        raise ValueError('Export output must be outside the source data_root')
    document = build_metadata(config)
    payload = (json.dumps(document, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix='.metadata-', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # Publish a complete file atomically, refusing to replace an existing file.
            os.link(temporary, output)
            status = 'created'
        except FileExistsError:
            if output.read_bytes() != payload:
                raise ValueError(f'Existing export differs: {output}; use a new output path')
            status = 'unchanged'
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    counts = Counter(len(record['captions']) for record in document['records'])
    return {
        'status': status,
        'output': str(output),
        'images': len(document['records']),
        'captions': sum(n * count for n, count in counts.items()),
        'caption_count_distribution': dict(sorted(counts.items())),
        'sha256': hashlib.sha256(payload).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True,
                        help='Destination JSON outside data_root; existing differing files are never replaced')
    args = parser.parse_args()
    try:
        config, _ = read_json(args.config)
        print(json.dumps(export(config, args.output), indent=2))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f'ERROR: Cannot export metadata: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
