"""Tiny-data overfitting gate only; never starts full-dataset training."""
import argparse
from dataclasses import replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import time


def atomic_json(path, value):
    import tempfile
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset-dir', 'data-root', 'tokenizer-dir', 'model-config', 'preprocessing-config', 'learning-config', 'output-dir'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--strategy', choices=('global', 'patch'), required=True)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    p.add_argument('--bf16', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--stop-after-epochs', type=int, help='Controlled interruption for resume checks; not a pass')
    args = p.parse_args()
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    lock = None
    try:
        import torch
        from src.learning.config import LearningConfig, gate
        from src.learning.engine import seed_all, evaluate, train_epoch, save_training, restore_training
        from src.models.config import ModelConfig
        from src.models.nanovlm import NanoVLM
        from src.tokenization.student import StudentTokenizer, training_corpus
        from src.preprocessing.images import ImageConfig, ImagePreprocessor
        from src.data.task_dataset import TaskDataset
        from src.tasks.builder import TaskBuilder, resolve_image
        config = LearningConfig(**json.loads(Path(args.learning_config).read_text()))
        if args.stop_after_epochs is not None and args.stop_after_epochs < 1:
            raise ValueError('stop-after-epochs must be positive')
        if args.device == 'cuda' and (not torch.cuda.is_available() or 'L40S' not in torch.cuda.get_device_name()):
            raise ValueError('Expected an available L40S on rama')
        if args.bf16 and (args.device != 'cuda' or not torch.cuda.is_bf16_supported()):
            raise ValueError('BF16 requires supported CUDA')
        runtime = {'python': platform.python_version(), 'torch': str(torch.__version__),
                   'cuda': torch.version.cuda, 'device': args.device, 'bf16': args.bf16,
                   'gpu': torch.cuda.get_device_name() if args.device == 'cuda' else None,
                   'attention_backend': 'math', 'deterministic_algorithms': True}
        print(json.dumps({'preflight': runtime, 'free_disk_gib': shutil.disk_usage('.').free/2**30,
            'free_vram_gib': torch.cuda.mem_get_info()[0]/2**30 if args.device == 'cuda' else None}), flush=True)
        torch.set_num_threads(2)
        seed_all(config.seed)
        if args.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        tokenizer = StudentTokenizer(args.tokenizer_dir)
        _, corpus = training_corpus(args.dataset_dir)
        if corpus != tokenizer.manifest['provenance']:
            raise ValueError('Dataset differs from tokenizer training provenance')
        model_config = replace(ModelConfig.load(args.model_config), convolution_strategy=args.strategy)
        pixels = ImageConfig.from_dict(json.loads(Path(args.preprocessing_config).read_text()))
        builder = TaskBuilder(tokenizer, args.data_root, bos_token_id=1, eos_token_id=2,
                              image_loader=ImagePreprocessor(pixels))
        examples, selections, image_hashes = {}, {}, {}
        for split, number in (('train', config.train_images), ('val', config.validation_images)):
            datasets = {variant: TaskDataset(args.dataset_dir, split, variant, builder) for variant in ('short', 'long')}
            order = sorted(range(len(datasets['short'])), key=lambda i: datasets['short'].records[i]['image_id'])
            random.Random(config.seed).shuffle(order)
            if len(order) < number:
                raise ValueError(f'Not enough {split} images')
            order = order[:number]
            selections[split] = [datasets['short'].records[i]['image_id'] for i in order]
            image_hashes[split] = [hashlib.sha256(resolve_image(Path(args.data_root), datasets['short'].records[i]['image_path']).read_bytes()).hexdigest() for i in order]
            for variant, dataset in datasets.items():
                examples[split+'/'+variant] = [dataset[i] for i in order]
        if set(selections['train']) & set(selections['val']):
            raise ValueError('Training/validation image leakage')
        if any(len(e.input_ids) > model_config.max_text_tokens for group in examples.values() for e in group):
            raise ValueError('Context overflow; no truncation allowed')
        # Cache only this tiny subset in RAM; no full dataset or teacher generation.
        training = [e for pair in zip(examples['train/short'], examples['train/long']) for e in pair]
        model = NanoVLM.from_artifacts(model_config, tokenizer, pixels).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                      betas=(0.9, 0.999), eps=1e-8, weight_decay=config.weight_decay,
                                      foreach=False, fused=False)
        source_root = Path(__file__).resolve().parents[2]
        sources = {str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in sorted((source_root/'src').rglob('*.py'))}
        revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=source_root, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=source_root, capture_output=True, text=True, check=True).stdout
        contract = {'learning': config.to_dict(), 'model': model_config.to_dict(),
                    'identity': model._artifact_identity, 'dataset': corpus, 'selected_ids': selections,
                    'image_hashes': image_hashes, 'runtime': runtime, 'source_hashes': sources,
                    'optimizer': {'name': 'AdamW', 'betas': [0.9,0.999], 'eps': 1e-8, 'foreach': False, 'fused': False},
                    'tasks': ['short','long'], 'num_workers': 0, 'gradient_accumulation': 1,
                    'evaluation_precision': 'float32', 'scheduler': None}
        # Normalize tuple/list representation for stable equality after save/reload.
        contract = json.loads(json.dumps(contract))
        output = Path(args.output_dir)
        if args.resume:
            if not (output/'latest.pt').is_file():
                raise ValueError('Resume requires latest.pt; no silent fresh start')
        else:
            output.mkdir(parents=True, exist_ok=False)
        lock = (output/'.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def evaluation(split):
            return {v: evaluate(model, examples[split+'/'+v], config.batch_size, args.device) for v in ('short','long')}
        epoch, steps, history = 0, 0, []
        if args.resume:
            state = restore_training(output/'latest.pt', model, optimizer, contract)
            epoch, steps, initial, history = state['epoch'], state['steps'], state['initial'], state['history']
        else:
            initial = evaluation('train')
            initial_validation = evaluation('val')
            atomic_json(output/'run.json', {'contract': contract, 'code_revision': revision,
                        'working_tree_dirty': bool(dirty), 'parameter_counts': model.parameter_counts(),
                        'initial_train_loss': initial, 'initial_validation_loss': initial_validation})
            save_training(output/'latest.pt', model, optimizer, contract, 0, 0, initial, history)
        atomic_json(output/'metrics.json', history)  # Checkpoint history is authoritative after interruption.
        stop = min(config.epochs, epoch+args.stop_after_epochs) if args.stop_after_epochs else config.epochs
        while epoch < stop:
            started = time.monotonic()
            epoch += 1
            result = train_epoch(model, optimizer, training, config, epoch, args.device, args.bf16)
            steps += result['optimizer_steps']
            record = {'epoch': epoch, 'steps': steps, **result, 'learning_rate': config.learning_rate,
                      'train_eval': evaluation('train'), 'validation': evaluation('val')}
            record['seconds'] = time.monotonic()-started
            if args.device == 'cuda':
                record['peak_allocated_gib'] = torch.cuda.max_memory_allocated()/2**30
            history.append(record)
            print(json.dumps(record), flush=True)
            if epoch % config.checkpoint_every == 0 or epoch == stop:
                save_training(output/'latest.pt', model, optimizer, contract, epoch, steps, initial, history)
                atomic_json(output/'metrics.json', history)
        if epoch < config.epochs:
            atomic_json(output/'status.json', {'status': 'interrupted', 'epoch': epoch, 'gate_passed': False})
            print('Saved controlled interruption; resume with the same command plus --resume.')
            return
        final = history[-1]['train_eval']
        passed = gate(initial, final, config)
        summary = {'status': 'passed' if all(passed.values()) else 'failed', 'gate_passed': all(passed.values()),
                   'per_task_pass': passed, 'initial_train_loss': initial, 'final_train_loss': final,
                   'relative_reduction': {v: 1-final[v]/initial[v] for v in final},
                   'final_validation_loss': history[-1]['validation'], 'epochs': epoch, 'steps': steps,
                   'selected_ids': selections, 'strategy': args.strategy, 'seed': 42,
                   'full_training_started': False}
        atomic_json(output/'status.json', summary)
        # Export only once; latest.pt is the authoritative resume artifact.
        if not (output/'model.pt').exists():
            model.save_checkpoint(output/'model.pt', provenance={'kind': 'tiny_overfit', 'code_revision': revision,
                'seed': 42, 'selected_ids': selections, 'gate_passed': summary['gate_passed']})
        print(json.dumps(summary, indent=2))
        if not summary['gate_passed']:
            p.exit(1, 'Overfit gate FAILED. Stop and debug; do not start full training.\n')
    except (OSError, ValueError, RuntimeError, ImportError, KeyError, TypeError) as exc:
        p.exit(1, f'ERROR: {exc}\nStop and debug; saved checkpoints are preserved.\n')
    finally:
        if lock is not None:
            lock.close()


if __name__ == '__main__':
    main()
