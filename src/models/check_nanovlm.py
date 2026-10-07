"""Real training samples -> TASK -> pixels -> paired VLMs -> loss/backward.

Acceptance smoke test only: no optimizer updates or teacher calls.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess


def check_behavior(model, images, ids, attention):
    import torch
    model.eval()
    with torch.no_grad():
        logits = model(images, ids, attention)
        if tuple(logits.shape) != (*ids.shape, model.config.vocabulary_size) or not torch.isfinite(logits).all():
            raise ValueError('Invalid model logits')
        cut = min(3, ids.shape[1]-1)
        changed = ids.clone()
        changed[:, cut:] = (changed[:, cut:] % (model.config.vocabulary_size-3))+3
        altered = model(images, changed, attention)
        torch.testing.assert_close(logits[:, :cut], altered[:, :cut], atol=2e-5, rtol=2e-4)
        # Alter masked content, and compare each example with its unpadded form.
        padded = ids.clone()
        padded[~attention.bool()] = 3
        other = model(images, padded, attention)
        torch.testing.assert_close(logits[attention.bool()], other[attention.bool()], atol=2e-5, rtol=2e-4)
        for i in range(ids.shape[0]):
            length = int(attention[i].sum())
            single = model(images[i:i+1], ids[i:i+1, :length], attention[i:i+1, :length])
            torch.testing.assert_close(logits[i:i+1, :length], single, atol=2e-5, rtol=2e-4)
        different = model(-images, ids, attention)
        visual_effect = (different[attention.bool()]-logits[attention.bool()]).abs().max().item()
        if visual_effect <= 1e-8:
            raise ValueError('No detected visual conditioning in text logits')
    return logits, visual_effect


def check_backward(model, images, ids, attention, loss_mask, bf16=False):
    import torch
    from src.losses.causal import causal_cross_entropy
    model.train()
    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type=images.device.type, dtype=torch.bfloat16, enabled=bf16):
        logits = model(images, ids, attention)
        loss = causal_cross_entropy(logits, ids, loss_mask, attention_mask=attention)
        selected = loss_mask[:, 1:].bool()
        manual = torch.nn.functional.cross_entropy(logits[:, :-1][selected].float(), ids[:, 1:][selected])
    torch.testing.assert_close(loss, manual)
    loss.backward()
    gradients = {}
    for name in ('vision', 'connector', 'decoder'):
        total = 0.
        for parameter in getattr(model, name).parameters():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise ValueError(f'Missing/non-finite gradient in {name}')
            total += parameter.grad.float().abs().sum().item()
        if total == 0:
            raise ValueError(f'No gradient reaches {name}')
        gradients[name] = total
    return {'loss': loss.item(), 'gradient_abs_sums': gradients, 'optimizer_steps': 0}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-dir', required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--tokenizer-dir', required=True)
    p.add_argument('--model-config', required=True)
    p.add_argument('--preprocessing-config', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--num-images', type=int, default=2)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--bf16', action='store_true')
    args = p.parse_args()
    try:
        import torch
        from src.models.nanovlm import NanoVLM
        from src.models.config import ModelConfig
        from src.tokenization.student import StudentTokenizer, training_corpus
        from src.preprocessing.images import ImageConfig, ImagePreprocessor
        from src.tasks.builder import TaskBuilder, collate
        from src.data.task_dataset import TaskDataset
        from src.data.artifacts import publish
        output = Path(args.output_dir)
        if output.exists() or args.num_images < 1:
            raise ValueError('Use a new output directory and a positive sample count')
        if args.bf16 and (args.device != 'cuda' or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
            raise ValueError('BF16 check requires supported CUDA')
        runtime = {'python': platform.python_version(), 'torch': str(torch.__version__),
                   'cuda': torch.version.cuda, 'free_disk_gib': shutil.disk_usage(output.parent if output.parent.exists() else '.').free/2**30}
        if args.device == 'cuda':
            if not torch.cuda.is_available() or 'L40S' not in torch.cuda.get_device_name():
                raise ValueError('Expected available L40S GPU on rama')
            runtime['gpu'] = torch.cuda.get_device_name()
            runtime['free_vram_gib'] = torch.cuda.mem_get_info()[0]/2**30
            torch.cuda.reset_peak_memory_stats()
        print(json.dumps({'preflight': runtime}), flush=True)
        torch.set_num_threads(2)
        tokenizer = StudentTokenizer(args.tokenizer_dir)
        _, provenance = training_corpus(args.dataset_dir)
        if provenance != tokenizer.manifest['provenance']:
            raise ValueError('Tokenizer corpus identity differs from dataset')
        config = ModelConfig.load(args.model_config)
        pixel_config = ImageConfig.from_dict(json.loads(Path(args.preprocessing_config).read_text()))
        builder = TaskBuilder(tokenizer, args.data_root, bos_token_id=1, eos_token_id=2,
                              image_loader=ImagePreprocessor(pixel_config))
        datasets = {v: TaskDataset(args.dataset_dir, 'train', v, builder) for v in ('short', 'long')}
        order = sorted(range(len(datasets['short'])), key=lambda i: datasets['short'].records[i]['image_id'])[:args.num_images]
        if len(order) != args.num_images:
            raise ValueError('Not enough training samples')
        batches = {}
        for variant, dataset in datasets.items():
            batch = collate([dataset[i] for i in order], tokenizer.pad_token_id)
            batches[variant] = (torch.stack(batch['images']).to(args.device),
                                *(torch.tensor(batch[k], device=args.device) for k in ('input_ids', 'attention_mask', 'loss_mask')))
        revision = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
        report = {'ok': True, 'runtime': runtime, 'seed': 42, 'bf16': args.bf16,
                  'image_ids': [datasets['short'].records[i]['image_id'] for i in order],
                  'dataset': provenance, 'code_revision': revision, 'branches': {}}
        initial = None
        configs = []
        for strategy in ('global', 'patch'):
            paired = replace(config, convolution_strategy=strategy)
            configs.append(paired)
            torch.manual_seed(42)
            model = NanoVLM.from_artifacts(paired, tokenizer, pixel_config).to(args.device)
            if initial is None:
                initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(value.cpu(), initial[key], rtol=0, atol=0)
            branch = {'config': paired.to_dict(), 'counts': model.parameter_counts(), 'tasks': {}}
            for variant, (images, ids, attention, loss_mask) in batches.items():
                expected, effect = check_behavior(model, images, ids, attention)
                result = check_backward(model, images, ids, attention, loss_mask, args.bf16)
                branch['tasks'][variant] = {**result, 'logits_shape': list(expected.shape),
                                           'visual_effect_max_abs': effect}
                print(f'Passed {strategy}/{variant} forward and backward', flush=True)
            checkpoint = output/(strategy+'.pt')
            model.save_checkpoint(checkpoint, provenance={'seed': 42, 'code_revision': revision,
                                  'dataset': provenance, 'kind': 'untrained_acceptance_model'})
            restored = NanoVLM.load_checkpoint(checkpoint, tokenizer=tokenizer,
                         preprocessing=pixel_config, expected_config=paired).to(args.device)
            model.eval()
            images, ids, attention, _ = batches['short']
            with torch.no_grad():
                torch.testing.assert_close(restored(images, ids, attention), model(images, ids, attention), rtol=0, atol=0)
            branch['checkpoint_reload'] = True
            report['branches'][strategy] = branch
            del model, restored
        if replace(configs[0], convolution_strategy='patch') != configs[1]:
            raise ValueError('Paired configurations differ beyond placement')
        report['initial_weights_identical'] = True
        report['parameter_differences_patch_minus_global'] = {
            name: {kind: report['branches']['patch']['counts'][name][kind]-report['branches']['global']['counts'][name][kind]
                   for kind in ('total', 'trainable')} for name in ('vision', 'connector', 'decoder', 'model')}
        report['absolute_parameter_differences'] = {
            name: {kind: abs(value) for kind, value in counts.items()}
            for name, counts in report['parameter_differences_patch_minus_global'].items()}
        if args.device == 'cuda':
            report['peak_allocated_gib'] = torch.cuda.max_memory_allocated()/2**30
        publish(output/'report.json', (json.dumps(report, indent=2)+'\n').encode())
        print(json.dumps(report, indent=2))
    except (ValueError, OSError, RuntimeError, ImportError, KeyError, TypeError, AssertionError) as exc:
        p.exit(1, f'ERROR: {exc}\nAcceptance not complete; preserve outputs for diagnosis.\n')


if __name__ == '__main__':
    main()
