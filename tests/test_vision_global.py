import io
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


class ConfigTests(unittest.TestCase):
    def test_defaults_and_json(self):
        self.assertEqual(VisionConfig().num_patches, 196)
        self.assertEqual(VisionConfig.from_dict({'conv_channels': [16, 32]}), VisionConfig())

    def test_invalid_configuration(self):
        for values in ({'image_size': 225}, {'patch_size': 0}, {'embed_dim': 15},
                       {'conv_kernel_size': 2}, {'conv_channels': (16,)},
                       {'depth': True}, {'dropout': float('nan')}, {'conv_strategy': ''}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                VisionConfig(**values)


@unittest.skipIf(torch is None, 'PyTorch is required; run on rama in qwen-vl')
class GlobalVisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(42)
        self.config = VisionConfig(image_size=32, embed_dim=16, num_heads=4,
                                   conv_channels=(4, 8), depth=2, dropout=0)
        self.model = VisionEncoder(self.config).eval()

    def test_full_image_convolutions_before_patching(self):
        config = VisionConfig(embed_dim=16, num_heads=4, conv_channels=(4, 8), depth=1)
        model = VisionEncoder(config).eval()
        events = []
        handles = []
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Unfold)):
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
                         [(2, 3, 224, 224), (2, 4, 224, 224), (2, 8, 224, 224)])
        self.assertTrue(events[-1][0].endswith('unfold'))

    def test_convolution_crosses_future_patch_boundary(self):
        embedding = self.model.patch_embedding
        image = torch.randn(1, 3, 32, 32)
        boundary = image.clone()
        boundary[0, 0, 8, 16] += 10
        distant = image.clone()
        distant[0, 0, 8, 20] += 10
        with torch.no_grad():
            original = embedding(image)
            changed = embedding(boundary)
            far = embedding(distant)
        self.assertGreater((original[:, 0] - changed[:, 0]).abs().max().item(), 1e-7)
        torch.testing.assert_close(original[:, 2:], changed[:, 2:], rtol=0, atol=0)
        torch.testing.assert_close(original[:, 0], far[:, 0], rtol=0, atol=0)

    def test_patch_order_and_projection(self):
        embedding = self.model.patch_embedding
        image = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            features = embedding.stem(image)
            actual = embedding(image)
            for index, (row, col) in enumerate(((0, 0), (0, 16), (16, 0), (16, 16))):
                expected = embedding.projection(features[:, :, row:row+16, col:col+16].reshape(2, -1))
                torch.testing.assert_close(actual[:, index], expected)

    def test_gradients_reach_all_parameters(self):
        result = self.model(torch.randn(2, 3, 32, 32))
        (result * torch.randn_like(result)).mean().backward()
        for name, parameter in self.model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_reload_and_eval_repeatability(self):
        image = torch.randn(2, 3, 32, 32)
        buffer = io.BytesIO()
        torch.save(self.model.state_dict(), buffer)
        buffer.seek(0)
        restored = VisionEncoder(self.config).eval()
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        with torch.no_grad():
            expected = self.model(image)
            torch.testing.assert_close(expected, self.model(image))
            torch.testing.assert_close(expected, restored(image))
            torch.testing.assert_close(expected[:1], self.model(image[:1]), atol=1e-5, rtol=1e-5)
        self.assertNotEqual(self.model.blocks[0].norm1.weight.data_ptr(),
                            self.model.blocks[1].norm1.weight.data_ptr())

    def test_reject_wrong_inputs_and_conflicting_strategy(self):
        for image in (torch.zeros(3, 32, 32), torch.zeros(1, 3, 31, 32),
                      torch.zeros(0, 3, 32, 32), torch.zeros(1, 3, 32, 32, dtype=torch.uint8)):
            with self.assertRaises(ValueError):
                self.model(image)
        with self.assertRaises(ValueError):
            VisionEncoder(self.config, conv_strategy='patch')


if __name__ == '__main__':
    unittest.main()
