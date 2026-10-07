import importlib.util
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import tempfile
import unittest
from src.learning.config import LearningConfig, gate

HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch


class LearningConfigTests(unittest.TestCase):
    def test_gate_requires_both_absolute_and_relative(self):
        config = LearningConfig()
        self.assertEqual(gate({'short': 8., 'long': 8.}, {'short': .8, 'long': 1.2}, config),
                         {'short': True, 'long': False})
        self.assertFalse(gate({'short': 1., 'long': 1.}, {'short': .5, 'long': float('nan')}, config)['short'])
        self.assertFalse(gate({'short': 1., 'long': 1.}, {'short': .1, 'long': float('nan')}, config)['long'])

    def test_invalid_and_full_training_refused(self):
        for values in ({'seed': 404}, {'train_images': 25200}, {'epochs': 0},
                       {'learning_rate': float('nan')}, {'scheduler': 'cosine'},
                       {'required_relative_reduction': 1}, {'gradient_clip': 0}):
            with self.assertRaises(ValueError):
                LearningConfig(**values)


@unittest.skipUnless(HAS_TORCH, 'PyTorch required on rama')
class LearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        cls.deterministic = torch.are_deterministic_algorithms_enabled()
        cls.backends = (torch.backends.cuda.flash_sdp_enabled(), torch.backends.cuda.mem_efficient_sdp_enabled(), torch.backends.cuda.math_sdp_enabled())
        cls.benchmark = torch.backends.cudnn.benchmark
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)
        torch.use_deterministic_algorithms(cls.deterministic)
        torch.backends.cuda.enable_flash_sdp(cls.backends[0])
        torch.backends.cuda.enable_mem_efficient_sdp(cls.backends[1])
        torch.backends.cuda.enable_math_sdp(cls.backends[2])
        torch.backends.cudnn.benchmark = cls.benchmark

    def examples(self):
        from src.tasks.builder import TaskExample
        return [TaskExample(torch.full((3,16,16), float(i)/4), i, 'unused.png', 'train', 'short', '', '',
                            [1,3], [4,5,2][:2+i%2], [0,0]+[1]*(2+i%2), 6) for i in range(4)]

    def model(self):
        from src.models.config import ModelConfig
        from src.models.nanovlm import NanoVLM
        return NanoVLM(ModelConfig(image_size=16, patch_size=8, image_embedding_dimension=16,
            embedding_dimension=16, conv_channels=(4,8), decoder_layers=1,
            vocabulary_size=16, dropout=.1, max_text_tokens=16))

    def test_epoch_resume_matches_uninterrupted(self):
        from src.learning.engine import seed_all, train_epoch, save_training, restore_training, evaluate
        config = LearningConfig(batch_size=2, epochs=2)
        examples = self.examples()
        seed_all(42)
        model = self.model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0, foreach=False)
        initial = {'short': evaluate(model, examples, 2, 'cpu'), 'long': 1.}
        before = model.decoder.lm_head.weight.detach().clone()
        train_epoch(model, optimizer, examples, config, 1, 'cpu', False)
        self.assertFalse(torch.equal(before, model.decoder.lm_head.weight))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'latest.pt'
            save_training(path, model, optimizer, {'test': 1}, 1, 2, initial, [])
            expected_metrics = train_epoch(model, optimizer, examples, config, 2, 'cpu', False)
            expected = {k: v.detach().clone() for k,v in model.state_dict().items()}
            resumed = self.model()
            resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=.001, weight_decay=0, foreach=False)
            with self.assertRaises(ValueError):
                restore_training(path, resumed, resumed_optimizer, {'test': 2})
            state = restore_training(path, resumed, resumed_optimizer, {'test': 1})
            self.assertEqual(state['epoch'], 1)
            actual = train_epoch(resumed, resumed_optimizer, examples, config, 2, 'cpu', False)
            self.assertEqual(actual, expected_metrics)
            for key, value in resumed.state_dict().items():
                torch.testing.assert_close(value, expected[key], rtol=0, atol=0)

    def test_validation_is_token_weighted_and_does_not_update(self):
        from src.learning.engine import seed_all, evaluate
        seed_all(42)
        model = self.model()
        examples = self.examples()
        before = {k: v.clone() for k,v in model.state_dict().items()}
        single = evaluate(model, examples, 1, 'cpu')
        batched = evaluate(model, examples, 4, 'cpu')
        self.assertAlmostEqual(single, batched, places=5)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    @unittest.skipUnless(HAS_TORCH and torch.cuda.is_available(), 'CUDA required')
    def test_cuda_bf16_optimizer_step(self):
        import os
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        from src.learning.engine import seed_all, train_epoch
        seed_all(42)
        model = self.model().cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, foreach=False)
        result = train_epoch(model, optimizer, self.examples(), LearningConfig(batch_size=2), 1, 'cuda', True)
        self.assertEqual(result['optimizer_steps'], 2)
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
