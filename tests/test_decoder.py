import importlib.util
import io
import unittest
from src.models.config import ModelConfig

HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    from src.models.decoder.language import CausalLanguageDecoder


class DecoderConfigTests(unittest.TestCase):
    def test_context_limit(self):
        self.assertEqual(ModelConfig().max_text_tokens, 512)
        for value in (0, -1, True):
            with self.assertRaises(ValueError):
                ModelConfig(max_text_tokens=value)


@unittest.skipUnless(HAS_TORCH, 'PyTorch required on rama')
class DecoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(42)
        self.config = ModelConfig(vocabulary_size=32, embedding_dimension=16,
                                  decoder_layers=2, max_text_tokens=8, dropout=0)
        self.model = CausalLanguageDecoder(self.config).eval()
        self.ids = torch.tensor([[1, 4, 5, 2], [1, 6, 2, 0]])
        self.mask = self.ids != 0
        self.visual = torch.randn(2, 1, 16)

    def test_causality_and_visual_conditioning(self):
        with torch.no_grad():
            original = self.model(self.ids, self.visual, self.mask)
            changed = self.ids.clone()
            changed[:, 2] = 12
            altered = self.model(changed, self.visual, self.mask)
            torch.testing.assert_close(original[:, :2], altered[:, :2], atol=1e-6, rtol=1e-5)
            prefix_only = self.model(self.ids[:, :2], self.visual)
            torch.testing.assert_close(original[:, :2], prefix_only, atol=1e-6, rtol=1e-5)
            different_image = self.model(self.ids, -self.visual, self.mask)
            self.assertGreater((original-different_image).abs().max().item(), 1e-7)

    def test_padding_invariance(self):
        with torch.no_grad():
            original = self.model(self.ids, self.visual, self.mask)
            ids = torch.cat([self.ids, torch.zeros(2, 2, dtype=torch.long)], dim=1)
            mask = torch.cat([self.mask, torch.zeros(2, 2, dtype=torch.bool)], dim=1)
            ids[~mask] = 15  # Explicit mask, not the masked token's ID, determines exclusion.
            padded = self.model(ids, self.visual, mask)
            torch.testing.assert_close(original[self.mask], padded[:, :4][self.mask], atol=1e-6, rtol=1e-5)
            single = self.model(self.ids[1:2, :3], self.visual[1:2])
            torch.testing.assert_close(original[1:2, :3], single, atol=1e-6, rtol=1e-5)

    def test_loss_backward_and_text_alignment(self):
        from src.losses.causal import causal_cross_entropy
        visual = self.visual.clone().requires_grad_()
        logits = self.model(self.ids, visual, self.mask)
        self.assertEqual(tuple(logits.shape), (2, 4, 32))
        logits.retain_grad()
        loss_mask = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 0]])
        loss = causal_cross_entropy(logits, self.ids, loss_mask, attention_mask=self.mask)
        selected = loss_mask[:, 1:].bool()
        expected = torch.nn.functional.cross_entropy(logits[:, :-1][selected], self.ids[:, 1:][selected])
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertGreater(visual.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(visual.grad).all())
        self.assertGreater(logits.grad[:, 1].abs().sum().item(), 0)
        self.assertEqual(logits.grad[:, 0].abs().sum().item(), 0)
        for name, parameter in self.model.named_parameters():
            with self.subTest(name=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertEqual(self.model.token_embedding.weight.grad[0].abs().sum().item(), 0)

    def test_configured_depth_positions_and_reload(self):
        self.assertEqual(len(self.model.blocks), 2)
        self.assertEqual(self.model.position_embedding.num_embeddings, 9)
        self.assertNotEqual(self.model.token_embedding.weight.data_ptr(), self.model.lm_head.weight.data_ptr())
        self.assertEqual(self.model.token_embedding.weight[0].abs().sum().item(), 0)
        buffer = io.BytesIO()
        torch.save(self.model.state_dict(), buffer)
        buffer.seek(0)
        restored = CausalLanguageDecoder(self.config).eval()
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        with torch.no_grad():
            torch.testing.assert_close(self.model(self.ids, self.visual), restored(self.ids, self.visual), rtol=0, atol=0)
            self.assertEqual(tuple(restored(torch.ones(2, 8, dtype=torch.long), self.visual).shape), (2, 8, 32))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            CausalLanguageDecoder(ModelConfig())
        for ids in (self.ids.float(), torch.ones(2, 9, dtype=torch.long),
                    torch.zeros(2, 4, dtype=torch.long), torch.full((2, 4), 32),
                    torch.ones(2, 0, dtype=torch.long)):
            with self.assertRaises(ValueError):
                self.model(ids, self.visual)
        with self.assertRaises(ValueError):
            self.model(self.ids, self.visual.expand(-1, 2, -1))
        for mask in (torch.tensor([[1, 0, 1, 1]]*2), torch.ones(2, 4)*.5):
            with self.assertRaises(ValueError):
                self.model(self.ids, self.visual, mask)

    @unittest.skipUnless(HAS_TORCH and torch.cuda.is_available(), 'CUDA required')
    def test_cuda_bf16_backward(self):
        from src.losses.causal import causal_cross_entropy
        model = self.model.cuda()
        visual = self.visual.cuda().to(torch.bfloat16).requires_grad_()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(self.ids.cuda(), visual, self.mask.cuda())
            loss = causal_cross_entropy(logits, self.ids.cuda(),
                                        torch.tensor([[0, 0, 1, 1], [0, 0, 1, 0]], device='cuda'))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(visual.grad).all())
        for parameter in model.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())
