from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HAS_TORCH = importlib.util.find_spec('torch') is not None
HAS_TOKENIZERS = importlib.util.find_spec('tokenizers') is not None
if HAS_TORCH:
    import torch
    from src.models.nanovlm import NanoVLM
    from src.models.check_nanovlm import check_behavior, check_backward
from src.models.config import ModelConfig


@unittest.skipUnless(HAS_TORCH, 'PyTorch required on rama')
class NanoVLMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.config = ModelConfig(image_size=32, image_embedding_dimension=16,
                                 embedding_dimension=24, attention_heads=4,
                                 conv_channels=(4, 8), decoder_layers=1, dropout=0,
                                 vocabulary_size=32, max_text_tokens=128)

    def test_both_strategies_alignment_counts_and_gradients(self):
        from src.losses.causal import causal_cross_entropy
        images = torch.randn(2, 3, 32, 32)
        ids = torch.tensor([[1, 4, 5, 2], [1, 6, 2, 0]])
        attention = ids != 0
        loss_mask = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 0]])
        reference = None
        for strategy in ('global', 'patch'):
            torch.manual_seed(42)
            model = NanoVLM(replace(self.config, convolution_strategy=strategy))
            state = model.state_dict()
            if reference is None:
                reference = {k: v.clone() for k, v in state.items()}
                counts = model.parameter_counts()
            else:
                self.assertEqual(counts, model.parameter_counts())
                for name, value in state.items():
                    torch.testing.assert_close(value, reference[name], rtol=0, atol=0)
            check_behavior(model, images, ids, attention)
            check_backward(model, images, ids, attention, loss_mask)
            model.eval()
            logits = model(images, ids, attention)
            logits.retain_grad()
            causal_cross_entropy(logits, ids, loss_mask, attention_mask=attention).backward()
            self.assertGreater(logits.grad[:, 1].abs().sum().item(), 0)
            self.assertEqual(logits.grad[:, 0].abs().sum().item(), 0)
            self.assertEqual(logits.grad[:, -1].abs().sum().item(), 0)
            self.assertEqual(counts['model']['total'], sum(counts[n]['total'] for n in ('vision', 'connector', 'decoder')))

    @unittest.skipUnless(HAS_TOKENIZERS, 'tokenizers required')
    def test_data_task_pixels_checkpoint_and_identity(self):
        from PIL import Image
        from src.tokenization.student import train, digest
        from src.preprocessing.images import ImageConfig, ImagePreprocessor
        from src.tasks.builder import TaskBuilder, collate
        from src.data.task_dataset import TaskDataset
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root/'data'
            data.mkdir()
            records = []
            for i, color in enumerate(('red', 'blue')):
                Image.new('RGB', (17+i, 23), color).save(root/f'{i}.png')
                records.append({'image_id': i, 'image_path': f'{i}.png', 'split': 'train',
                                'short_desc': 'A small cat sits on a blue chair beside the window.',
                                'long_desc': 'A small cat sits on a blue chair beside the window. '*4})
            raw = ('\n'.join(json.dumps(r) for r in records)+'\n').encode()
            (data/'train.jsonl').write_bytes(raw)
            (data/'manifest.json').write_text(json.dumps({'schema_version': 1,
                'files': {'train.jsonl': digest(raw)}, 'counts': {'train': 2}}))
            tokenizer, _ = train(data, root/'tokenizer')
            config = replace(self.config, vocabulary_size=tokenizer.vocabulary_size)
            pixel_config = ImageConfig(image_size=32)
            builder = TaskBuilder(tokenizer, root, bos_token_id=1, eos_token_id=2,
                                  image_loader=ImagePreprocessor(pixel_config))
            for strategy in ('global', 'patch'):
                paired = replace(config, convolution_strategy=strategy)
                model = NanoVLM.from_artifacts(paired, tokenizer, pixel_config)
                for variant in ('short', 'long'):
                    dataset = TaskDataset(data, 'train', variant, builder)
                    batch = collate([dataset[0], dataset[1]], 0)
                    images = torch.stack(batch['images'])
                    ids, attention, mask = (torch.tensor(batch[k]) for k in ('input_ids', 'attention_mask', 'loss_mask'))
                    check_behavior(model, images, ids, attention)
                    check_backward(model, images, ids, attention, mask)
                path = root/(strategy+'.pt')
                model.save_checkpoint(path, provenance={'seed': 42, 'test_fixture': True})
                restored = NanoVLM.load_checkpoint(path, tokenizer=tokenizer,
                            preprocessing=pixel_config, expected_config=paired)
                model.eval()
                with torch.no_grad():
                    torch.testing.assert_close(model(images, ids, attention), restored(images, ids, attention), rtol=0, atol=0)
                for wrong in (replace(paired, convolution_strategy='patch' if strategy=='global' else 'global'),
                              replace(paired, max_text_tokens=129)):
                    with self.assertRaises(ValueError):
                        NanoVLM.load_checkpoint(path, tokenizer=tokenizer, preprocessing=pixel_config, expected_config=wrong)
                with self.assertRaises(ValueError):
                    NanoVLM.load_checkpoint(path, tokenizer=tokenizer,
                        preprocessing=replace(pixel_config, mean=(0.,0.,0.)), expected_config=paired)
                saved_hash = tokenizer.artifact_sha256
                tokenizer.artifact_sha256 = '0'*64
                with self.assertRaises(ValueError):
                    NanoVLM.load_checkpoint(path, tokenizer=tokenizer, preprocessing=pixel_config, expected_config=paired)
                tokenizer.artifact_sha256 = saved_hash
            with self.assertRaises(ValueError):
                NanoVLM.from_artifacts(replace(config, vocabulary_size=3), tokenizer, pixel_config)
            with self.assertRaises(ValueError):
                NanoVLM.from_artifacts(config, tokenizer, ImageConfig(image_size=224))

    def test_input_and_unbound_checkpoint_rejection(self):
        with self.assertRaises(ValueError):
            NanoVLM(ModelConfig())
        model = NanoVLM(self.config)
        with self.assertRaises(ValueError):
            model(torch.zeros(1,3,32,32), torch.ones(2,3,dtype=torch.long), torch.ones(2,3))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                model.save_checkpoint(Path(tmp)/'unbound.pt', provenance={})

    @unittest.skipUnless(HAS_TORCH and torch.cuda.is_available(), 'CUDA required')
    def test_both_strategies_cuda_bf16(self):
        for strategy in ('global', 'patch'):
            torch.manual_seed(42)
            model = NanoVLM(replace(self.config, convolution_strategy=strategy)).cuda()
            ids = torch.tensor([[1,4,5,2]], device='cuda')
            check_backward(model, torch.randn(1,3,32,32,device='cuda'), ids,
                           torch.ones_like(ids), torch.tensor([[0,0,1,1]],device='cuda'), True)
