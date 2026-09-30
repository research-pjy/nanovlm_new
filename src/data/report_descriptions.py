"""Read-only summary of saved descriptions; never reclassify or rewrite records."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys


def report(path, field):
    errors, statuses, counts, warnings = Counter(), Counter(), Counter(), Counter()
    seen = set()
    length_only = 0
    within_20_27 = 0
    with Path(path).open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            image_id = row['image_id']
            if image_id in seen:
                raise ValueError(f'Duplicate image ID {image_id} on line {line_number}')
            seen.add(image_id)
            v = row['validation'][field]
            if not isinstance(row[field], str):
                raise ValueError(f'{field} not generated for image {image_id}')
            statuses[v['status']] += 1
            counts[v['word_count']] += 1
            errors.update(v['errors'])
            warnings.update(v.get('warnings', []))
            length_only += bool(v['errors']) and all(e.startswith('word_count_outside_') for e in v['errors'])
            within_20_27 += field == 'short_desc' and 20 <= v['word_count'] <= 27 and all(
                e.startswith('word_count_outside_') for e in v['errors'])
    return {'records': len(seen), 'field': field, 'saved_validation_counts': dict(statuses),
            'error_counts': dict(errors), 'warning_counts': dict(warnings),
            'length_only_flagged': length_only,
            'short_20_27_without_other_errors': within_20_27 if field == 'short_desc' else None,
            'word_count_histogram': dict(sorted(counts.items())),
            'semantic_review': 'not_performed', 'source_modified': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--field', choices=['short_desc', 'long_desc'], required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(report(args.input, args.field), indent=2))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
