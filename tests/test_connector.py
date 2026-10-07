import importlib.util
import io
import unittest

HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    from src.models.connector.projector import VisualTextualConnector


@unittest.skipUnless(HAS_TORCH, 'PyTorch required; run on rama in qwen-vl')
class ConnectorTests(unittest.TestCase):
    def test_shapes_counts_initialization_and_reload(self):
        for dv, dt in ((128, 128), (32, 16), (16, 32)):
            with self.subTest(dv=dv, dt=dt):
                torch.manual_seed(42)
                model = VisualTextualConnector(dv, dt)
                self.assertEqual(sum(p.numel() for p in model.parameters()), dv*dt+dt)
                self.assertEqual(model.projection.bias.abs().sum().item(), 0)
                image = torch.randn(2, 197, dv)
                result = model(image)
                self.assertEqual(tuple(result.shape), (2, 1, dt))
                buffer = io.BytesIO()
                torch.save(model.state_dict(), buffer)
                buffer.seek(0)
                restored = VisualTextualConnector(dv, dt)
                restored.load_state_dict(torch.load(buffer, weights_only=True))
                torch.testing.assert_close(result, restored(image), rtol=0, atol=0)
                torch.manual_seed(42)
                identical = VisualTextualConnector(dv, dt)
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, identical.state_dict()[name], rtol=0, atol=0)

    def test_cls_only_and_gradients(self):
        model = VisualTextualConnector(16, 8)
        sequence = torch.randn(2, 197, 16, requires_grad=True)
        original = sequence.detach().clone()
        output = model(sequence)
        torch.testing.assert_close(output, torch.nn.functional.gelu(
            torch.nn.functional.linear(sequence[:, :1], model.projection.weight, model.projection.bias)))
        changed = sequence.detach().clone()
        changed[:, 1:] += 100
        torch.testing.assert_close(output, model(changed), rtol=0, atol=0)
        (output * torch.randn_like(output)).sum().backward()
        self.assertTrue(torch.isfinite(sequence.grad).all())
        self.assertGreater(sequence.grad[:, :1].abs().sum().item(), 0)
        self.assertEqual(sequence.grad[:, 1:].abs().sum().item(), 0)
        for parameter in model.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        torch.testing.assert_close(sequence.detach(), original, rtol=0, atol=0)

    def test_shared_connector_with_both_encoders(self):
        from src.models.vision.config import VisionConfig
        from src.models.vision.encoder import VisionEncoder
        old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            connector = VisualTextualConnector(16, 24)
            for strategy in ('global', 'patch'):
                encoder = VisionEncoder(VisionConfig(conv_strategy=strategy, image_size=32,
                    embed_dim=16, num_heads=4, depth=1, conv_channels=(4, 8), dropout=0))
                images = torch.randn(2, 3, 32, 32, requires_grad=True)
                features = encoder(images)
                output = connector(features)
                self.assertEqual(tuple(output.shape), (2, 1, 24))
                (output * torch.randn_like(output)).sum().backward()
                self.assertTrue(torch.isfinite(images.grad).all())
                self.assertGreater(images.grad.abs().sum().item(), 0)
                # Patches influence CLS through encoder attention.
                grad = encoder.patch_embedding.stem.layers[0].weight.grad
                self.assertTrue(torch.isfinite(grad).all())
                self.assertGreater(grad.abs().sum().item(), 0)
        finally:
            torch.set_num_threads(old_threads)

    def test_bad_inputs(self):
        for dv, dt in ((0, 16), (16, -1), (True, 16)):
            with self.assertRaises(ValueError):
                VisualTextualConnector(dv, dt)
        model = VisualTextualConnector(16, 8)
        for value in (None, torch.zeros(2, 16), torch.zeros(0, 197, 16),
                      torch.zeros(2, 1, 16), torch.zeros(2, 197, 8),
                      torch.zeros(2, 197, 16, dtype=torch.long)):
            with self.assertRaises(ValueError):
                model(value)

    @unittest.skipUnless(HAS_TORCH and torch.cuda.is_available(), 'CUDA required')
    def test_cuda_bf16(self):
        model = VisualTextualConnector(128, 96).cuda()
        features = torch.randn(2, 197, 128, device='cuda', requires_grad=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = model(features)
        self.assertEqual(tuple(output.shape), (2, 1, 96))
        output.float().square().mean().backward()
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(features.grad).all())
