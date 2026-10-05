"""CPU-only task integration checks using an explicitly diagnostic byte tokenizer."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

from src.data.artifacts import publish
from src.data.task_dataset import TaskDataset
from .builder import TaskBuilder, TaskConfig, collate


class DiagnosticByteTokenizer:
    """Test probe, not a selected training tokenizer; UTF-8 bytes are reversible."""
    def encode(self, text, *, add_special_tokens=False):
        if add_special_tokens:
            raise ValueError('Diagnostic tokenizer does not add special tokens')
        return list(text.encode('utf-8'))


def check(dataset_dir, data_root, task_config, num_images=None):
    if num_images is not None and (type(num_images) is not int or num_images < 1):
        raise ValueError('num_images must be positive')
    builder = TaskBuilder(DiagnosticByteTokenizer(), data_root, task_config)
    counts, prefixes, previews, errors = {}, {}, [], []
    for split in ('train', 'val', 'test'):
        for variant in ('short', 'long'):
            dataset = TaskDataset(dataset_dir, split, variant, builder)
            limit = len(dataset) if num_images is None else min(num_images, len(dataset))
            key = split + '/' + variant
            histogram, successes = Counter(), 0
            for index in range(limit):
                example = None
                try:
                    example = dataset[index]
                    source = dataset.records[index][variant + '_desc']
                    if example.prompt_text + example.target_text != source:
                        raise ValueError('Text split did not preserve the source')
                    if bytes(example.input_ids).decode('utf-8') != source:
                        raise ValueError('Diagnostic token round-trip failed')
                    if example.loss_mask != [0]*len(example.tokenized_prompt) + [1]*len(example.tokenized_target):
                        raise ValueError('Incorrect prompt/continuation mask')
                    if example.image.mode != 'RGB':
                        raise ValueError('Image was not converted to RGB')
                    collate([example], pad_token_id=256)
                    histogram[example.prefix_word_count] += 1
                    successes += 1
                    if index == 0:
                        previews.append({'split': split, 'variant': variant, 'image_id': example.image_id,
                                         'prompt_text': example.prompt_text, 'target_text': example.target_text,
                                         'prefix_words': example.prefix_word_count,
                                         'prompt_tokens': len(example.tokenized_prompt),
                                         'target_tokens': len(example.tokenized_target)})
                except (OSError, ValueError, KeyError, TypeError, UnicodeError) as exc:
                    errors.append(f'{key}[{index}]: {exc}')
                finally:
                    if example is not None:
                        example.image.close()
            counts[key] = {'checked': limit, 'passed': successes, 'available': len(dataset)}
            prefixes[key] = dict(sorted(histogram.items()))
    return {'ok': not errors, 'random_seed': task_config.random_seed,
            'task_config': asdict(task_config),
            'dataset_manifest_sha256': hashlib.sha256((Path(dataset_dir)/'manifest.json').read_bytes()).hexdigest(),
            'tokenizer': 'diagnostic UTF-8 byte probe; not for training',
            'counts': counts, 'prefix_word_histograms': prefixes, 'previews': previews, 'errors': errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--data-config', type=Path, required=True)
    parser.add_argument('--task-config', type=Path, required=True)
    parser.add_argument('--num-images', type=int, help='Maximum images PER SPLIT, for both variants; omit to check all')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    try:
        data_config = json.loads(args.data_config.read_text())
        config = TaskConfig.from_dict(json.loads(args.task_config.read_text()))
        if config.random_seed != data_config['random_seed']:
            raise ValueError('Task and data seeds disagree')
        output = args.report.resolve()
        if output.is_relative_to(args.dataset.resolve()) or output.is_relative_to(Path(data_config['data_root']).expanduser().resolve()) or output in {args.data_config.resolve(), args.task_config.resolve()}:
            raise ValueError('Report must be separate from source data/configs')
        result = check(args.dataset, data_config['data_root'], config, args.num_images)
        publish(args.report, (json.dumps(result, indent=2, ensure_ascii=False)+'\n').encode())
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result['ok'] else 1
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f'ERROR: Cannot check tasks: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
