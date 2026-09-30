"""Phase 1G: join existing descriptions by ID into teacher-free split JSONL files."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile

from .export_splits import validate_assignments
from .verify_selection import integer, read_json


def encode(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def load_generated(directory, field, expected, provenance):
    directory = Path(directory)
    contract, run_hash = read_json(directory / 'run.json')
    if contract['sources'] != provenance or contract['phase'] != ('short' if field == 'short_desc' else 'long'):
        raise ValueError(f'{field}: run provenance or phase disagrees with input dataset')
    signature = hashlib.sha256(encode(contract).encode()).hexdigest()
    path = directory / ('shortdesc.jsonl' if field == 'short_desc' else 'longdesc.jsonl')
    digest = hashlib.sha256()
    rows = {}
    with path.open('rb') as stream:
        for line_number, line in enumerate(stream, 1):
            digest.update(line)
            row = json.loads(line)
            image_id = integer(row['image_id'], f'{field} line {line_number} image_id')
            if image_id in rows or image_id not in expected:
                raise ValueError(f'{field}: duplicate or unexpected image ID {image_id}')
            ref = expected[image_id]
            for name, source in [('assignment', 'assignment'), ('image_path', 'image_path'),
                                 ('source_captions', 'captions'), ('source_caption_ids', 'caption_ids')]:
                if row[name] != ref[source]:
                    raise ValueError(f'{field}: {name} mismatch for image {image_id}')
            if not isinstance(row[field], str) or not row[field].strip():
                raise ValueError(f'{field}: empty/non-string description for image {image_id}')
            if row['run_signature'] != signature or row['generation_configuration'] != contract['configuration']:
                raise ValueError(f'{field}: run signature/configuration mismatch for image {image_id}')
            if row['teacher_model'] != contract['teacher']['teacher_model']:
                raise ValueError(f'{field}: teacher mismatch for image {image_id}')
            integer(row['generation_seed'], f'{field} generation seed')
            if not isinstance(row['generation_timestamp'], str) or not row['generation_timestamp']:
                raise ValueError(f'{field}: missing generation timestamp for image {image_id}')
            if not isinstance(row['validation'][field], dict):
                raise ValueError(f'{field}: missing validation for image {image_id}')
            rows[image_id] = row
    missing = set(expected) - set(rows)
    if missing:
        raise ValueError(f'{field}: missing {len(missing)} IDs; examples: {sorted(missing)[:10]}')
    if contract['total_records'] != len(rows):
        raise ValueError(f'{field}: run count mismatch')
    return rows, {'jsonl_sha256': digest.hexdigest(), 'run_json_sha256': run_hash, 'contract': contract}


def export(metadata_path, splits_path, short_dir, long_dir, output_dir):
    output = Path(output_dir).expanduser()
    sources = [Path(metadata_path), Path(splits_path), Path(short_dir), Path(long_dir)]
    if any(output.resolve() == p.resolve() or output.resolve().is_relative_to(p.resolve()) or
           p.resolve().is_relative_to(output.resolve()) for p in sources):
        raise ValueError('Output must be separate from all input files/directories')
    metadata, metadata_hash = read_json(Path(metadata_path))
    splits, splits_hash = read_json(Path(splits_path))
    if metadata['schema_version'] != 1 or splits['schema_version'] != 1:
        raise ValueError('Unsupported input schema')
    if splits['metadata_sha256'] != metadata_hash:
        raise ValueError('Metadata fingerprint does not match splits')
    for key in ('random_seed', 'source_manifest_sha256', 'source_annotation_sha256'):
        if metadata[key] != splits[key]:
            raise ValueError(f'Split provenance mismatch: {key}')
    ids = validate_assignments(metadata, splits['image_ids'])
    if {g: len(v) for g, v in ids.items()} != splits['counts']:
        raise ValueError('Split count mismatch')
    expected = {r['image_id']: r for r in metadata['records']}
    for record in expected.values():
        path = record['image_path']
        if not isinstance(path, str) or not path or PurePosixPath(path).is_absolute() or '..' in PurePosixPath(path).parts or '\\' in path:
            raise ValueError('Metadata must contain safe relative POSIX image paths')
    provenance = {'metadata_sha256': metadata_hash, 'splits_sha256': splits_hash}
    short, short_provenance = load_generated(short_dir, 'short_desc', expected, provenance)
    long, long_provenance = load_generated(long_dir, 'long_desc', expected, provenance)
    payloads, counts = {}, {}
    for assignment, split in [('train', 'train'), ('val', 'val'), ('held_out', 'test')]:
        lines = []
        for image_id in ids[assignment]:
            ref = expected[image_id]
            row = {'schema_version': 1, 'image_id': image_id, 'image_path': ref['image_path'],
                   'source_captions': ref['captions'], 'source_caption_ids': ref['caption_ids'],
                   'short_desc': short[image_id]['short_desc'], 'long_desc': long[image_id]['long_desc'],
                   'split': split, 'source_split': ref['source_split'], 'generation': {}, 'validation': {}}
            for field, generated in [('short_desc', short[image_id]), ('long_desc', long[image_id])]:
                row['generation'][field] = {k: generated[k] for k in (
                    'teacher_model', 'generation_timestamp', 'generation_seed', 'generation_configuration', 'run_signature')}
                row['validation'][field] = generated['validation'][field]
            lines.append(encode(row)+'\n')
        payloads[f'{split}.jsonl'] = ''.join(lines).encode()
        counts[split] = len(lines)
    manifest = {'schema_version': 1, 'image_path_base': 'data_root', 'counts': counts,
                'assignment_mapping': {'train': 'train', 'val': 'val', 'held_out': 'test'},
                'inputs': {**provenance, 'short': short_provenance, 'long': long_provenance},
                'filtering': 'none; all reviewed selection IDs retained',
                'semantic_review': 'not_performed',
                'files': {name: hashlib.sha256(raw).hexdigest() for name, raw in payloads.items()}}
    payloads['manifest.json'] = (json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True)+'\n').encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Directory publication prevents an interrupted export leaving a mixed split set.
    with (output.parent / f'.{output.name}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if output.exists():
            if not output.is_dir() or {p.name for p in output.iterdir()} != set(payloads) or any(
                    not (output / name).is_file() or (output / name).read_bytes() != raw for name, raw in payloads.items()):
                raise ValueError('Existing dataset differs; choose a new output directory')
            return {'status': 'unchanged', 'counts': counts, 'output': str(output)}
        staging = Path(tempfile.mkdtemp(dir=output.parent, prefix=f'.{output.name}-'))
        try:
            for name, raw in payloads.items():
                with (staging / name).open('wb') as stream:
                    stream.write(raw)
                    stream.flush()
                    os.fsync(stream.fileno())
            os.rename(staging, output)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    return {'status': 'created', 'counts': counts, 'output': str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('metadata', 'splits', 'short-dir', 'long-dir', 'output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(export(args.metadata, args.splits, args.short_dir, args.long_dir, args.output_dir), indent=2))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f'ERROR: Cannot export final dataset: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
