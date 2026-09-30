"""Phase 1F: audit descriptions and optionally export a configured subset."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata

from .artifacts import publish

FLAGS = {'malformed_json', 'malformed_record', 'invalid_image_id', 'invalid_assignment',
         'empty_output', 'malformed_output', 'word_count_outside_range',
         'duplicate_output', 'duplicate_image_id', 'truncated_output'}


def normalize(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def audit(raw, field, policy):
    if set(policy) != {'min_words', 'max_words', 'exclude_flags'}:
        raise ValueError('Policy requires min_words, max_words and exclude_flags')
    low, high = policy['min_words'], policy['max_words']
    if type(low) is not int or type(high) is not int or not 0 <= low <= high:
        raise ValueError('Invalid word-count range')
    excluded = policy['exclude_flags']
    if not isinstance(excluded, list) or any(not isinstance(f, str) or f not in FLAGS for f in excluded):
        raise ValueError('Unknown exclude_flags policy')
    if field not in ('short_desc', 'long_desc'):
        raise ValueError('Unsupported description field')
    lines = raw.splitlines(keepends=True)
    entries, text_groups, id_groups = [], defaultdict(list), defaultdict(list)
    for index, line in enumerate(lines):
        entry = {'line': index+1, 'image_id': None, 'assignment': None, 'word_count': None, 'flags': []}
        entries.append(entry)
        try:
            row = json.loads(line)
        except (ValueError, UnicodeError):
            entry['flags'].append('malformed_json')
            continue
        if not isinstance(row, dict):
            entry['flags'].append('malformed_record')
            continue
        image_id = row.get('image_id')
        if type(image_id) is not int or image_id < 0:
            entry['flags'].append('invalid_image_id')
        else:
            entry['image_id'] = image_id
            id_groups[image_id].append(index)
        assignment = row.get('assignment')
        if not isinstance(assignment, str) or assignment not in ('train', 'val', 'held_out'):
            entry['flags'].append('invalid_assignment')
        else:
            entry['assignment'] = assignment
        text = row.get(field)
        if text is None or isinstance(text, str) and not text.strip():
            entry['flags'].append('empty_output')
        if not isinstance(text, str):
            entry['flags'].append('malformed_output')
        else:
            count = sum(bool(re.search(r'\w', token)) for token in text.split())
            entry['word_count'] = count
            if not low <= count <= high:
                entry['flags'].append('word_count_outside_range')
            if (not count and text.strip()) or re.search(
                r'<\/?think>|```|^\s*(?:#{1,6}\s|[-*]\s|\d+[.)]\s|(?:shortdesc|longdesc|description)\s*:)',
                text, re.I | re.M
            ) or any(ord(c) < 32 and c not in '\n\r\t' for c in text):
                entry['flags'].append('malformed_output')
            key = normalize(text)
            if key and count:
                text_groups[key].append(index)
        # Preserve useful token-level evidence that cannot be inferred from plain text.
        history = row.get('attempts')
        if isinstance(history, list) and history and isinstance(history[-1], dict):
            if history[-1].get('text') == text and history[-1].get('truncated') is True:
                entry['flags'].append('truncated_output')
    duplicates = []
    for key, members in text_groups.items():
        if len(members) < 2:
            continue
        for i in members:
            entries[i]['flags'].append('duplicate_output')
        duplicates.append({'normalized_text_sha256': hashlib.sha256(key.encode()).hexdigest(),
                           'lines': [i+1 for i in members],
                           'image_ids': [entries[i]['image_id'] for i in members],
                           'assignments': sorted({entries[i]['assignment'] for i in members
                                                  if entries[i]['assignment'] is not None}),
                           'cross_assignment': len({entries[i]['assignment'] for i in members
                                                    if entries[i]['assignment'] is not None}) > 1})
    repeated_ids = []
    for image_id, members in id_groups.items():
        if len(members) > 1:
            repeated_ids.append({'image_id': image_id, 'lines': [i+1 for i in members]})
            for i in members:
                entries[i]['flags'].append('duplicate_image_id')
    histogram, flags = Counter(), Counter()
    retained = []
    for entry, line in zip(entries, lines):
        entry['flags'] = sorted(set(entry['flags']))
        entry['selected_by_policy'] = not bool(set(entry['flags']) & set(excluded))
        flags.update(entry['flags'])
        if entry['word_count'] is not None:
            histogram[entry['word_count']] += 1
        if entry['selected_by_policy']:
            retained.append(line)
    summary = {'records': len(entries), 'flagged_records': sum(bool(e['flags']) for e in entries),
               'flag_counts': dict(sorted(flags.items())), 'word_count_histogram': dict(sorted(histogram.items())),
               'duplicate_output_groups': len(duplicates),
               'cross_assignment_duplicate_groups': sum(g['cross_assignment'] for g in duplicates),
               'selected_by_policy': len(retained), 'excluded_by_policy': len(entries)-len(retained)}
    return {'schema_version': 1, 'field': field, 'input_sha256': hashlib.sha256(raw).hexdigest(),
            'policy': policy, 'duplicate_normalization': 'Unicode NFKC, casefold, collapse whitespace; punctuation retained',
            'semantic_review': 'not_performed', 'summary': summary, 'records': entries,
            'duplicate_output_groups': duplicates, 'duplicate_image_id_groups': repeated_ids}, b''.join(retained)


def validate(input_path, config_path, field, report_path, filtered_output=None):
    source, config_path, report_path = map(Path, (input_path, config_path, report_path))
    outputs = [report_path] + ([Path(filtered_output)] if filtered_output is not None else [])
    if len({p.resolve() for p in outputs}) != len(outputs) or any(
        p.resolve() in (source.resolve(), config_path.resolve()) for p in outputs
    ):
        raise ValueError('Outputs must be distinct from input/configuration and from each other')
    config = json.loads(config_path.read_text())
    raw = source.read_bytes()
    report, filtered = audit(raw, field, config[field])
    report['filtered_export_requested'] = filtered_output is not None
    report_bytes = (json.dumps(report, indent=2, ensure_ascii=False)+'\n').encode()
    payloads = [report_bytes] + ([filtered] if filtered_output is not None else [])
    # Check all destinations before publishing either artifact.
    for path, payload in zip(outputs, payloads):
        if path.exists() and path.read_bytes() != payload:
            raise ValueError(f'Existing output differs: {path}; choose a new output path')
    for path, payload in zip(outputs, payloads):
        publish(path, payload)
    return report['summary']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--field', choices=['short_desc', 'long_desc'], required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--filtered-output', type=Path, help='Optional separate subset; never rewrites source')
    args = parser.parse_args()
    try:
        print(json.dumps(validate(args.input, args.config, args.field, args.report, args.filtered_output), indent=2))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
