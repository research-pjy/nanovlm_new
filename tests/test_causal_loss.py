import math
import unittest

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != 'torch':
        raise
    torch = None
if torch is not None:
    from src.losses.causal import causal_cross_entropy


@unittest.skipIf(torch is None, 'PyTorch required; run on rama in qwen-vl')
class CausalLossTests(unittest.TestCase):
    def test_shift_first_target_and_gradient_mask(self):
        logits = torch.zeros(1, 5, 4, requires_grad=True)
        ids = torch.tensor([[0, 1, 2, 3, 0]])
        mask = torch.tensor([[0, 0, 1, 1, 0]])
        loss = causal_cross_entropy(logits, ids, mask)
        self.assertAlmostEqual(loss.item(), math.log(4), places=6)
        loss.backward()
        # Final prompt position predicts the first target; final target predicts
        # padding and is excluded. Prefix-label predictions are also excluded.
        self.assertGreater(logits.grad[0, 1].abs().sum().item(), 0)
        self.assertGreater(logits.grad[0, 2].abs().sum().item(), 0)
        self.assertEqual(logits.grad[0, [0, 3, 4]].abs().sum().item(), 0)
        self.assertLess(logits.grad[0, 1, 2].item(), 0)
        self.assertLess(logits.grad[0, 2, 3].item(), 0)

    def test_mean_is_token_weighted_and_sum_matches(self):
        logits = torch.tensor([[[0., 2.], [2., 0.], [0., 0.]],
                               [[1., 0.], [0., 0.], [0., 0.]]])
        ids = torch.tensor([[0, 1, 0], [0, 1, 0]])
        mask = torch.tensor([[0, 1, 1], [0, 1, 0]])
        expected = (2 * math.log1p(math.exp(-2)) + math.log1p(math.exp(1))) / 3
        mean = causal_cross_entropy(logits, ids, mask)
        self.assertAlmostEqual(mean.item(), expected, places=6)
        torch.testing.assert_close(causal_cross_entropy(logits, ids, mask, reduction='sum'), mean * 3)

    def test_masked_values_do_not_affect_loss(self):
        logits = torch.zeros(1, 4, 3)
        ids = torch.tensor([[0, -100, 2, -100]])
        mask = torch.tensor([[0, 0, 1, 0]])
        expected = causal_cross_entropy(logits, ids, mask)
        logits[:, [0, 2, 3]] = float('nan')
        torch.testing.assert_close(causal_cross_entropy(logits, ids, mask), expected)

    def test_task_collation_eos_equal_to_pad(self):
        from src.tasks.builder import TaskExample, collate
        def example(prompt, target):
            return TaskExample(None, 1, 'unused.jpg', 'train', 'short', '', '',
                               prompt, target, [0]*len(prompt)+[1]*len(target), 6)
        batch = collate([example([1, 2], [3, 0]), example([1], [0])], pad_token_id=0)
        logits = torch.zeros(2, 4, 4, requires_grad=True)
        loss = causal_cross_entropy(logits, torch.tensor(batch['input_ids']),
                                    torch.tensor(batch['loss_mask']),
                                    attention_mask=torch.tensor(batch['attention_mask']))
        loss.backward()
        self.assertGreater(logits.grad[0, 2].abs().sum().item(), 0)  # genuine EOS
        self.assertGreater(logits.grad[1, 0].abs().sum().item(), 0)  # genuine EOS
        self.assertEqual(logits.grad[1, 1:].abs().sum().item(), 0)  # padding

    def test_invalid_inputs_fail(self):
        logits = torch.zeros(1, 3, 4)
        ids = torch.tensor([[0, 1, 2]])
        for mask in (torch.zeros(1, 3), torch.tensor([[1, 1, 1]]),
                     torch.tensor([[0., .5, 1.]]), torch.zeros(1, 2)):
            with self.assertRaises(ValueError):
                causal_cross_entropy(logits, ids, mask)
        mask = torch.tensor([[0, 0, 1]])
        for wrong_ids in (ids.float(), torch.tensor([[0, 1, 4]])):
            with self.assertRaises(ValueError):
                causal_cross_entropy(logits, wrong_ids, mask)
        with self.assertRaises(ValueError):
            causal_cross_entropy(logits, ids, mask, attention_mask=torch.tensor([[1, 1, 0]]))
        with self.assertRaises(ValueError):
            causal_cross_entropy(logits, ids, mask, attention_mask=torch.tensor([[1, 0, 1]]))
        with self.assertRaises(ValueError):
            causal_cross_entropy(logits[:, :1], ids[:, :1], mask[:, :1])
        with self.assertRaises(ValueError):
            causal_cross_entropy(logits, ids, mask, reduction='none')

    def test_low_precision_uses_float32_loss(self):
        logits = torch.zeros(1, 2, 4, dtype=torch.bfloat16, requires_grad=True)
        loss = causal_cross_entropy(logits, torch.tensor([[0, 1]]), torch.tensor([[0, 1]]))
        self.assertEqual(loss.dtype, torch.float32)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), 'CUDA required')
    def test_cuda_bf16_backward(self):
        logits = torch.randn(2, 5, 16, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        ids = torch.randint(16, (2, 5), device='cuda')
        mask = torch.tensor([[0, 0, 1, 1, 1]] * 2, device='cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = causal_cross_entropy(logits, ids, mask)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == '__main__':
    unittest.main()
