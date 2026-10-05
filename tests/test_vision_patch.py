import io
import json
from dataclasses import replace
from pathlib import Path
import unittest

from src.models.vision.config import VisionConfig

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != 'torch':
        raise
    torch = None
if torch is not None:
    from src.models.vision.encoder import VisionEncoder


class PatchConfigTests(unittest.TestCase):
    def test_shipped_configs_only_differ_in_strategy(self):
        root = Path(__file__).resolve().parents[1] / 'configs'
        global_config = VisionConfig.from_dict(json.loads((root / 'vision.global.json').read_text()))
        patch_config = VisionConfig.from_dict(json.loads((root / 'vision.patch.json').read_text()))
        self.assertEqual(replace(global_config, conv_strategy='patch'), patch_config)


@unittest.skipIf(torch is None, 'PyTorch is required; run on rama in qwen-vl')
class PatchVisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.config = VisionConfig(conv_strategy='patch', image_size=32, embed_dim=16,
                                   num_heads=4, conv_channels=(4, 8), depth=2, dropout=0)
        torch.manual_seed(42)
        self.model = VisionEncoder(self.config).eval()

    def test_patch_extraction_precedes_shared_convolutions(self):
        model = VisionEncoder(replace(self.config, image_size=224)).eval()
        events = []
        handles = []
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Unfold, torch.nn.Conv2d)):
                handles.append(module.register_forward_pre_hook(
                    lambda m, args, name=name: events.append((name, tuple(args[0].shape)))))
        try:
            with torch.no_grad():
                result = model(torch.randn(2, 3, 224, 224))
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(tuple(result.shape), (2, 197, 16))
        self.assertEqual([shape for _, shape in events],
                         [(2, 3, 224, 224), (392, 3, 16, 16), (392, 4, 16, 16)])
        self.assertTrue(events[0][0].endswith('unfold'))
        self.assertEqual(model.label, 'Experimental Variant B: Patch-Wise Convolution')

    def test_no_convolution_leakage_across_patch_boundary(self):
        images = torch.randn(2, 3, 32, 32)
        changed = images.clone()
        changed[0, 0, 8, 16] += 10
        with torch.no_grad():
            original = self.model.patch_embedding(images)
            actual = self.model.patch_embedding(changed)
        torch.testing.assert_close(original[:, [0, 2, 3]], actual[:, [0, 2, 3]], rtol=0, atol=0)
        torch.testing.assert_close(original[1], actual[1], rtol=0, atol=0)
        self.assertGreater((original[0, 1] - actual[0, 1]).abs().max().item(), 1e-7)

    def test_shared_stem_matches_independent_patch_loop_and_gradients(self):
        embedding = self.model.patch_embedding
        images = torch.randn(2, 3, 32, 32)
        actual = embedding(images)
        # Reference uses explicit spatial slicing, independent of Unfold/reshape.
        reference = torch.stack([
            embedding.projection(embedding.stem(images[:, :, row:row+16, col:col+16]).reshape(2, -1))
            for row, col in ((0, 0), (0, 16), (16, 0), (16, 16))
        ], dim=1)
        torch.testing.assert_close(actual, reference, atol=1e-5, rtol=1e-5)
        weights = torch.randn_like(actual)
        parameters = tuple(embedding.parameters())
        batched_grad = torch.autograd.grad((actual * weights).sum(), parameters)
        loop_grad = torch.autograd.grad((reference * weights).sum(), parameters)
        for actual_grad, expected_grad in zip(batched_grad, loop_grad):
            torch.testing.assert_close(actual_grad, expected_grad, atol=1e-4, rtol=1e-4)

    def test_identical_patches_have_identical_embeddings(self):
        patch = torch.randn(1, 3, 16, 16)
        with torch.no_grad():
            tokens = self.model.patch_embedding(patch.repeat(1, 1, 2, 2))
        torch.testing.assert_close(tokens, tokens[:, :1].expand_as(tokens))

    def test_controlled_parameter_parity(self):
        torch.manual_seed(42)
        global_model = VisionEncoder(replace(self.config, conv_strategy='global')).eval()
        self.assertEqual(sum(p.numel() for p in self.model.parameters()),
                         sum(p.numel() for p in global_model.parameters()))
        self.assertEqual(self.model.state_dict().keys(), global_model.state_dict().keys())
        for name, value in self.model.state_dict().items():
            torch.testing.assert_close(value, global_model.state_dict()[name], rtol=0, atol=0)
        # With only one patch, the two placements must be equivalent.
        one_patch = replace(self.config, image_size=16)
        patch_model = VisionEncoder(one_patch).eval()
        global_model = VisionEncoder(replace(one_patch, conv_strategy='global')).eval()
        global_model.load_state_dict(patch_model.state_dict())
        image = torch.randn(2, 3, 16, 16)
        with torch.no_grad():
            torch.testing.assert_close(patch_model(image), global_model(image), atol=1e-5, rtol=1e-5)

    def test_backward_and_checkpoint_reload(self):
        images = torch.randn(2, 3, 32, 32)
        result = self.model(images)
        (result * torch.randn_like(result)).mean().backward()
        for name, parameter in self.model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.abs().sum().item(), 0)
        buffer = io.BytesIO()
        torch.save(self.model.state_dict(), buffer)
        buffer.seek(0)
        restored = VisionEncoder(self.config).eval()
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        with torch.no_grad():
            torch.testing.assert_close(result, restored(images))

    def test_reject_invalid_inputs(self):
        for image in (torch.zeros(3, 32, 32), torch.zeros(1, 3, 31, 32),
                      torch.zeros(0, 3, 32, 32), torch.zeros(1, 3, 32, 32, dtype=torch.uint8)):
            with self.assertRaises(ValueError):
                self.model(image)


if __name__ == '__main__':
    unittest.main()
