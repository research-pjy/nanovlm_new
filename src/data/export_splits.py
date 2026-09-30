"""Preserve verified COCO assignments in a portable, versionable split manifest."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

from .artifacts import publish
from .export_metadata import build_metadata
from .verify_selection import integer, read_json


GROUPS = ('train', 'val', 'held_out')


def validate_assignments(metadata, expected):
    """Require exact membership, order, and counts; report duplicate image IDs."""
    ids = {group: [] for group in GROUPS}
    seen = {}
    for record in metadata['records']:
        image_id = integer(record['image_id'], 'metadata image ID')
        assignment = record['assignment']
        if assignment not in ids:
            raise ValueError(f'Unknown assignment {assignment!r} for image {image_id}')
        if image_id in seen:
            raise ValueError(f'Duplicate image ID {image_id}: {seen[image_id]} and {assignment}')
        seen[image_id] = assignment
        ids[assignment].append(image_id)
    for group in GROUPS:
        if ids[group] != expected[group]:
            raise ValueError(f'{group}: metadata IDs or order disagree with verified selection')
        if metadata['assignments'][group] != len(ids[group]):
            raise ValueError(f'{group}: metadata count disagrees with records')
    return ids


def export(config, metadata_path, output):
    output = Path(output).expanduser()
    metadata_path = Path(metadata_path).expanduser()
    root = Path(config['data_root']).expanduser().resolve()
    if output.resolve().is_relative_to(root):
        raise ValueError('Split output must be outside the source data_root')
    if output.resolve() == metadata_path.resolve():
        raise ValueError('Split output must not be the metadata input')
    metadata, metadata_hash = read_json(metadata_path)
    # Rebuild in memory only, checking source annotations, selection and image paths.
    reference = build_metadata(config)
    expected = {group: [r['image_id'] for r in reference['records']
                        if r['assignment'] == group] for group in GROUPS}
    ids = validate_assignments(metadata, expected)
    if metadata != reference:
        raise ValueError('Metadata differs from verified source data; rerun Phase 1C to a new path')
    document = {
        'schema_version': 1,
        'random_seed': config['random_seed'],
        'train_fraction': config['train_fraction'],
        'source_manifest_sha256': reference['source_manifest_sha256'],
        'source_annotation_sha256': reference['source_annotation_sha256'],
        'metadata_sha256': metadata_hash,
        'counts': {group: len(ids[group]) for group in GROUPS},
        'image_ids': ids,
    }
    payload = (json.dumps(document, indent=2) + '\n').encode('utf-8')
    status = publish(output, payload)
    return {
        'status': status,
        'output': str(output),
        'counts': document['counts'],
        'pairwise_overlap': {
            f'{a}/{b}': len(set(ids[a]) & set(ids[b]))
            for a, b in (('train', 'val'), ('train', 'held_out'), ('val', 'held_out'))
        },
        'sha256': hashlib.sha256(payload).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--metadata', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        config, _ = read_json(args.config)
        print(json.dumps(export(config, args.metadata, args.output), indent=2))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f'ERROR: Cannot export splits: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
