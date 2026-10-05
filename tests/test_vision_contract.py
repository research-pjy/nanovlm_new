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


@unittest.skipIf(torch is None, 'PyTorch is required; run on rama in qwen-vl')
class VisionOutputContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_common_consumer_for_both_strategies(self):
        # Same downstream operation, without reading strategy or encoder internals.
        def consume(encoder, images):
            sequence = encoder(images)
            cls = sequence[:, 0, :]
            visual = sequence[:, 1:, :]
            return sequence, cls, visual

        for strategy in ('global', 'patch'):
            with self.subTest(strategy=strategy):
                torch.manual_seed(42)
                encoder = VisionEncoder(VisionConfig(
                    conv_strategy=strategy, embed_dim=16, num_heads=4,
                    depth=1, conv_channels=(4, 8), dropout=0)).eval()
                images = torch.randn(2, 3, 224, 224, requires_grad=True)
                sequence, cls, visual = consume(encoder, images)
                self.assertEqual(encoder.num_visual_tokens, 196)
                self.assertEqual(encoder.output_dim, 16)
                self.assertEqual(tuple(sequence.shape), (2, 197, 16))
                self.assertEqual(tuple(cls.shape), (2, 16))
                self.assertEqual(tuple(visual.shape), (2, 196, 16))
                torch.testing.assert_close(torch.cat([cls[:, None], visual], dim=1), sequence)
                self.assertEqual(sequence.device, images.device)
                self.assertTrue(sequence.is_floating_point())
                self.assertTrue(torch.isfinite(sequence).all())
                # Both exposed representations remain differentiable.
                for representation in (cls, visual):
                    gradient, = torch.autograd.grad(
                        (representation * torch.randn_like(representation)).sum(),
                        images, retain_graph=True)
                    self.assertTrue(torch.isfinite(gradient).all())
                    self.assertGreater(gradient.abs().sum().item(), 0)

    def test_contract_tracks_configured_dimensions(self):
        for strategy in ('global', 'patch'):
            with self.subTest(strategy=strategy):
                encoder = VisionEncoder(VisionConfig(
                    conv_strategy=strategy, image_size=32, patch_size=8,
                    embed_dim=24, num_heads=4, depth=1, conv_channels=(4, 8)))
                with torch.no_grad():
                    result = encoder(torch.randn(1, 3, 32, 32))
                self.assertEqual(tuple(result.shape),
                                 (1, 1 + encoder.num_visual_tokens, encoder.output_dim))
                self.assertEqual(encoder.num_visual_tokens, 16)
                self.assertEqual(encoder.output_dim, 24)


if __name__ == '__main__':
    unittest.main()
