"""Verify the saved COCO selection without creating or modifying dataset files."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import sys


def integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{label} must be an integer >= {minimum}')
    return value


def read_json(path):
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def reproduce_selection(candidate_ids, number_of_images, number_of_held_out, seed, train_fraction):
    """Match the original downloader: sorted IDs, local RNG, held-out first."""
    ids = sorted(candidate_ids)
    if len(ids) != len(set(ids)):
        raise ValueError('Candidate image IDs are not unique')
    required = number_of_images + number_of_held_out
    if required > len(ids):
        raise ValueError(f'Need {required} candidates; only {len(ids)} are eligible')
    random.Random(seed).shuffle(ids)
    selected = ids[number_of_held_out:required]
    boundary = round(number_of_images * train_fraction)
    return {
        'held_out': ids[:number_of_held_out],
        'train': selected[:boundary],
        'val': selected[boundary:],
    }


def verify(config):
    root = Path(config['data_root']).expanduser()
    if not root.is_absolute():
        raise ValueError('data_root must be an absolute path')
    size = integer(config['number_of_images'], 'number_of_images', 1)
    seed = integer(config['random_seed'], 'random_seed')
    held_out = integer(config['number_of_held_out'], 'number_of_held_out', 1)
    fraction = config['train_fraction']
    if type(fraction) not in (int, float) or not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError('train_fraction must be a finite number between 0 and 1')
    manifest, manifest_hash = read_json(root / 'image_selection.json')
    for key, expected in (('seed', seed), ('num_images', size), ('num_held_out', held_out)):
        integer(manifest[key], f'manifest.{key}')
        if manifest[key] != expected:
            raise ValueError(f'Manifest {key} differs from configuration; manifest was not changed')
    if type(manifest['train_fraction']) not in (int, float) or manifest['train_fraction'] != fraction:
        raise ValueError('Manifest train_fraction differs from configuration')
    splits = manifest['splits']
    if not isinstance(splits, list) or not splits or any(
        s not in ('train2017', 'val2017') for s in splits
    ) or len(splits) != len(set(splits)):
        raise ValueError('Manifest must list unique COCO 2017 source splits')

    metadata, counts, annotation_hashes = {}, Counter(), {}
    for split in splits:
        filename = f'captions_{split}.json'
        annotation, annotation_hashes[filename] = read_json(root / 'annotations' / filename)
        source_ids = set()
        for image in annotation['images']:
            image_id = integer(image['id'], 'annotation image ID')
            if image_id in metadata:
                raise ValueError(f'Duplicate annotation image ID: {image_id}')
            metadata[image_id] = (split, image['file_name'])
            source_ids.add(image_id)
        annotation_ids = set()
        for caption in annotation['annotations']:
            caption_id = integer(caption['id'], 'annotation caption ID')
            if caption_id in annotation_ids:
                raise ValueError(f'Duplicate caption ID in {filename}: {caption_id}')
            annotation_ids.add(caption_id)
            image_id = integer(caption['image_id'], 'caption image ID')
            if image_id not in source_ids:
                raise ValueError(f'Caption refers to unknown source image: {image_id}')
            if not isinstance(caption['caption'], str) or not caption['caption'].strip():
                raise ValueError(f'Malformed caption {caption_id} for image {image_id}')
            counts[image_id] += 1
    # Eligibility mirrors the downloader; it does not depend on downloaded files.
    candidates = [i for i in metadata if counts[i] >= 5]
    expected = reproduce_selection(candidates, size, held_out, seed, fraction)
    errors, seen, assignments = [], {}, {}
    for group, expected_ids in expected.items():
        entries = manifest[group]
        if not isinstance(entries, list):
            raise ValueError(f'Manifest {group} must be a list')
        actual = []
        for position, entry in enumerate(entries):
            image_id = integer(entry['image_id'], f'{group}[{position}].image_id')
            actual.append(image_id)
            if image_id in seen:
                errors.append(f'Repeated image ID {image_id} in {seen[image_id]} and {group}')
            seen[image_id] = group
            if metadata.get(image_id) != (entry['split'], entry['file_name']):
                errors.append(f'{group}[{position}]: source metadata mismatch for image {image_id}')
        if len(actual) != len(expected_ids):
            errors.append(f'{group}: expected {len(expected_ids)} IDs, found {len(actual)}')
        mismatches = sum(a != b for a, b in zip(actual, expected_ids))
        if mismatches:
            first = next(i for i, (a, b) in enumerate(zip(actual, expected_ids)) if a != b)
            errors.append(f'{group}: {mismatches} ordered ID mismatches; first at index {first}: '
                          f'expected {expected_ids[first]}, found {actual[first]}')
        assignments[group] = len(actual)
    return {
        'ok': not errors,
        'algorithm': 'sorted IDs with >=5 captions; random.Random(seed).shuffle; held-out first; round(N * train_fraction)',
        'random_seed': seed,
        'candidate_count': len(candidates),
        'excluded_fewer_than_five_captions': len(metadata) - len(candidates),
        'assignments': assignments,
        'manifest_sha256': manifest_hash,
        'annotation_sha256': annotation_hashes,
        'errors': errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    try:
        config, _ = read_json(args.config)
        report = verify(config)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f'ERROR: Cannot verify selection: {exc}', file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
