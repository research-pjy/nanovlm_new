from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from src.models.config import ModelConfig

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != 'torch':
        raise
    torch = None
if torch is not None:
    from src.models.vision.encoder import VisionEncoder


class ModelConfigTests(unittest.TestCase):
    def test_debug_preset_round_trip(self):
        path = Path(__file__).resolve().parents[1] / 'configs/model.debug.json'
        config = ModelConfig.load(path)
        self.assertEqual(config, ModelConfig())
        self.assertIsNone(config.vocabulary_size)
        with tempfile.TemporaryDirectory() as directory:
            saved = Path(directory) / 'config.json'
            saved.write_text(json.dumps(config.to_dict()))
            self.assertEqual(ModelConfig.load(saved), config)

    def test_vision_mapping_and_strategy_override(self):
        config = ModelConfig(image_size=32, patch_size=8, vision_blocks=2,
                             attention_heads=8, image_embedding_dimension=64,
                             embedding_dimension=128, dropout=0.2)
        vision = config.to_vision_config()
        self.assertEqual((vision.image_size, vision.patch_size, vision.depth,
                          vision.num_heads, vision.embed_dim, vision.dropout),
                         (32, 8, 2, 8, 64, 0.2))
        self.assertEqual(replace(config, convolution_strategy='patch').to_vision_config(),
                         replace(vision, conv_strategy='patch'))

    def test_invalid_dimensions_and_settings(self):
        for values in ({'decoder_layers': 0}, {'decoder_layers': True},
                       {'embedding_dimension': 127}, {'image_embedding_dimension': 127},
                       {'attention_heads': 0}, {'vision_blocks': -1},
                       {'image_size': 225}, {'patch_size': 0}, {'dropout': 1},
                       {'dropout': float('nan')}, {'convolution_strategy': 'other'},
                       {'vocabulary_size': 0}, {'vocabulary_size': True},
                       {'conv_channels': None}, {'unknown_setting': 1}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                ModelConfig.from_dict(values)

    def test_reject_nonobject_config(self):
        for value in ([], None, 4, 'debug'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ModelConfig.from_dict(value)

    def test_decoder_dimensions_preserved_without_constructing_decoder(self):
        config = ModelConfig(decoder_layers=3, embedding_dimension=256, vocabulary_size=1000)
        restored = ModelConfig.from_dict(config.to_dict())
        self.assertEqual(restored.decoder_layers, 3)
        self.assertEqual(restored.embedding_dimension, 256)
        self.assertEqual(restored.vocabulary_size, 1000)
        self.assertEqual(restored.to_vision_config().embed_dim, 128)


@unittest.skipIf(torch is None, 'PyTorch is required; run on rama in qwen-vl')
class DebugModelVisionTests(unittest.TestCase):
    def test_debug_encoders_share_parameters_and_output_contract(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            config = ModelConfig()
            images = torch.randn(1, 3, 224, 224)
            reference = None
            counts = []
            for strategy in ('global', 'patch'):
                torch.manual_seed(42)
                model = VisionEncoder(replace(config, convolution_strategy=strategy).to_vision_config()).eval()
                counts.append(sum(p.numel() for p in model.parameters()))
                if reference is None:
                    reference = model.state_dict()
                else:
                    for name, tensor in model.state_dict().items():
                        torch.testing.assert_close(tensor, reference[name], rtol=0, atol=0)
                with torch.no_grad():
                    output = model(images)
                self.assertEqual(tuple(output.shape), (1, 197, 128))
                self.assertTrue(torch.isfinite(output).all())
            self.assertEqual(counts[0], counts[1])
            self.assertLess(counts[0], 13760576)
        finally:
            torch.set_num_threads(old_threads)


if __name__ == '__main__':
    unittest.main()
