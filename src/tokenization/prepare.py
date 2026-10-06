"""Train/reuse student BPE and validate real TASK tokenization without a teacher."""
import argparse
from pathlib import Path

from src.data.artifacts import publish
from src.models.config import ModelConfig
from src.tokenization.student import StudentTokenizer, train, json_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--model-config', required=True)
    parser.add_argument('--resolved-model-config', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    try:
        # Artifacts must not overwrite or be created inside their input directory.
        dataset = Path(args.dataset_dir).resolve()
        output = Path(args.output_dir).resolve()
        resolved = Path(args.resolved_model_config).resolve()
        report_path = Path(args.report).resolve()
        if (output == dataset or output.is_relative_to(dataset) or dataset.is_relative_to(output)
                or resolved == Path(args.model_config).resolve()
                or any(p.is_relative_to(dataset) or p.is_relative_to(output) for p in (resolved, report_path))
                or resolved == report_path):
            raise ValueError('Use distinct output paths outside source dataset/tokenizer directories')
        model_config = ModelConfig.load(args.model_config)
        if args.check_only:
            tokenizer, status = StudentTokenizer(output), 'loaded'
            from src.tokenization.student import training_corpus
            _, provenance = training_corpus(dataset)
            if tokenizer.manifest['provenance'] != provenance:
                raise ValueError('Tokenizer provenance differs from dataset')
        else:
            tokenizer, status = train(dataset, output)
        print(f'Tokenizer {status}; actual vocabulary={tokenizer.vocabulary_size}', flush=True)
        from src.tasks.builder import TaskBuilder
        from src.data.task_dataset import TaskDataset
        # resolve_image still verifies every image path; pixel decoding is a 3I concern.
        builder = TaskBuilder(tokenizer, args.data_root, bos_token_id=1, eos_token_id=2,
                              image_loader=lambda path: None)
        report = {'ok': True, 'tokenizer_sha256': tokenizer.artifact_sha256,
                  'vocabulary_size': tokenizer.vocabulary_size, 'random_seed': 42,
                  'context_limit_including_bos_eos': 512, 'pixel_decoding': False, 'counts': {}}
        for split in ('train', 'val', 'test'):
            for variant in ('short', 'long'):
                data = TaskDataset(dataset, split, variant, builder)
                lengths = []
                for index in range(len(data)):
                    example = data[index]
                    if tokenizer.decode(example.input_ids) != example.prompt_text + example.target_text:
                        raise ValueError(f'{example.image_id}: text round-trip failed')
                    if tokenizer.decode(example.tokenized_prompt) != example.prompt_text:
                        raise ValueError('Prompt round-trip failed')
                    if tokenizer.decode(example.tokenized_target) != example.target_text:
                        raise ValueError('Target round-trip failed')
                    lengths.append(len(example.input_ids))
                lengths.sort()
                report['counts'][split + '/' + variant] = {
                    'checked': len(lengths), 'max_tokens': max(lengths, default=0),
                    'p95_tokens': lengths[int((len(lengths)-1)*.95)] if lengths else 0,
                    'over_512': sum(n > 512 for n in lengths)}
                print(f'Checked {split}/{variant}: {len(lengths)} examples', flush=True)
        report['context_limit_sufficient'] = not any(v['over_512'] for v in report['counts'].values())
        config = tokenizer.resolve_model_config(model_config)
        publish(resolved, json_bytes(config.to_dict()))
        publish(report_path, json_bytes(report))
        print(json_bytes(report).decode())
        if not report['context_limit_sufficient']:
            parser.exit(1, 'Context overflow found; review the shared limit before decoder implementation. No text truncated.\n')
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        parser.exit(1, f'ERROR: {exc}\nNo fallback tokenizer or network download was used.\n')


if __name__ == '__main__':
    main()
