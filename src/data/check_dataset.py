"""CPU-only checks and reload support for final DATA artifacts."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys

from .artifacts import publish
from .export_splits import validate_assignments
from .verify_selection import read_json
from .export_metadata import build_metadata
from .prepare_small import subset


def reload_dataset(directory):
    """Read all three splits and verify their checksums and counts; no teacher imports."""
    directory = Path(directory)
    manifest, _ = read_json(directory / 'manifest.json')
    if manifest['schema_version'] != 1:
        raise ValueError('Unsupported final dataset schema')
    result = {}
    for split in ('train', 'val', 'test'):
        name = split + '.jsonl'
        raw = (directory / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest['files'][name]:
            raise ValueError(f'{name}: checksum mismatch')
        rows = [json.loads(line) for line in raw.splitlines()]
        if len(rows) != manifest['counts'][split]:
            raise ValueError(f'{name}: manifest count mismatch')
        result[split] = rows
    return result, manifest


def check(directory, data_root, metadata_path, splits_path):
    metadata, mh = read_json(Path(metadata_path))
    splits, sh = read_json(Path(splits_path))
    if splits['metadata_sha256'] != mh:
        raise ValueError('Metadata/split fingerprint mismatch')
    validate_assignments(metadata, splits['image_ids'])
    data, manifest = reload_dataset(directory)
    if manifest['inputs']['metadata_sha256'] != mh or manifest['inputs']['splits_sha256'] != sh:
        raise ValueError('Final dataset input fingerprints mismatch')
    expected = {r['image_id']: r for r in metadata['records']}
    errors, seen = [], set()
    lengths = {'short_desc': Counter(), 'long_desc': Counter()}
    root = Path(data_root).expanduser().resolve()
    for split, assignment in [('train','train'), ('val','val'), ('test','held_out')]:
        rows = data[split]
        if [r.get('image_id') for r in rows] != splits['image_ids'][assignment]:
            errors.append(f'{split}: IDs/count/order differ from saved assignment')
        for row in rows:
            image_id = row.get('image_id')
            if type(image_id) is not int or image_id not in expected:
                errors.append(f'{split}: invalid/unexpected image ID {image_id!r}')
                continue
            if image_id in seen:
                errors.append(f'duplicate image ID / split leakage: {image_id}')
            seen.add(image_id)
            ref = expected[image_id]
            if row.get('split') != split or ref['assignment'] != assignment:
                errors.append(f'{image_id}: incorrect split')
            path = row.get('image_path')
            if not isinstance(path, str) or not path or PurePosixPath(path).is_absolute() or '..' in PurePosixPath(path).parts or '\\' in path:
                errors.append(f'{image_id}: unsafe image path')
            elif not (root/path).resolve().is_relative_to(root) or not (root/path).is_file():
                errors.append(f'{image_id}: missing image file or escaping symlink')
            if path != ref['image_path']:
                errors.append(f'{image_id}: image path mismatch')
            captions = row.get('source_captions')
            if not isinstance(captions, list) or len(captions) < 5 or any(not isinstance(c,str) or not c.strip() for c in captions):
                errors.append(f'{image_id}: missing/malformed captions')
            if captions != ref['captions'] or row.get('source_caption_ids') != ref['caption_ids']:
                errors.append(f'{image_id}: source caption mismatch')
            for field in lengths:
                text = row.get(field)
                if not isinstance(text,str) or not text.strip():
                    errors.append(f'{image_id}: unavailable {field}')
                else:
                    lengths[field][sum(bool(re.search(r'\w',w)) for w in text.split())] += 1
    if seen != set(expected):
        errors.append('Final dataset does not cover the selected IDs exactly')
    # Explicitly reload a second time to test reproducible saved-data consumption.
    again, _ = reload_dataset(directory)
    if data != again:
        errors.append('Dataset changed between reloads')
    return {'ok': not errors, 'counts': {k:len(v) for k,v in data.items()},
            'selected_images':len(expected), 'unique_image_ids':len(seen),
            'reload_verified': data == again,
            'word_count_histograms': {k:dict(sorted(v.items())) for k,v in lengths.items()},
            'length_warnings': {
                'short_outside_20_27': sum(n for w,n in lengths['short_desc'].items() if not 20<=w<=27),
                'long_outside_60_70': sum(n for w,n in lengths['long_desc'].items() if not 60<=w<=70)},
            'errors':errors, 'semantic_review':'not_performed'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','data-config','metadata','splits','report'):
        parser.add_argument('--'+name, type=Path, required=True)
    args=parser.parse_args()
    try:
        config,_=read_json(args.data_config)
        if config['random_seed'] != 42:
            raise ValueError('Project seed is 42, not 404')
        reference=build_metadata(config)  # Includes deterministic seed-42 source verification.
        metadata,_=read_json(args.metadata)
        if 'small_data' in metadata:
            reference=subset(reference,metadata['small_data']['num_images'],config['random_seed'])
        if metadata != reference:
            raise ValueError('Metadata differs from the verified source selection or deterministic subset')
        result=check(args.dataset,config['data_root'],args.metadata,args.splits)
        result['deterministic_source_selection_verified']=True
        result['random_seed']=42
        # Reports must stay outside the immutable exported directory and source data.
        report=args.report.resolve()
        if report.is_relative_to(args.dataset.resolve()) or report.is_relative_to(Path(config['data_root']).resolve()) or report in {args.metadata.resolve(),args.splits.resolve(),args.data_config.resolve()}:
            raise ValueError('Choose a separate report path')
        publish(args.report,(json.dumps(result,indent=2)+'\n').encode())
        print(json.dumps(result,indent=2))
        return 0 if result['ok'] else 1
    except (OSError,ValueError,TypeError,KeyError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr)
        return 1


if __name__=='__main__': sys.exit(main())
