import importlib.util
import tempfile
from pathlib import Path
import unittest
from src.preprocessing.images import ImageConfig, ImagePreprocessor


def oriented_fixture(path):
    from PIL import Image
    image = Image.new('RGB', (2, 2))
    image.putdata([(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255)])
    image.getexif()[274] = 6  # 90 degrees clockwise
    # Updating getexif() does not automatically persist metadata when saving PNG.
    image.save(path, exif=image.getexif())
    return image


class ImageConfigTests(unittest.TestCase):
    def test_exif_fixture_is_persisted(self):
        from PIL import Image, ImageOps
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'oriented.png'
            oriented_fixture(path)
            with Image.open(path) as saved:
                saved.load()
                self.assertEqual(saved.getexif().get(274), 6)
                self.assertEqual(ImageOps.exif_transpose(saved).getpixel((0, 0)), (0, 0, 255))

    def test_validation_and_reload(self):
        self.assertEqual(ImageConfig.from_dict(ImageConfig().to_dict()), ImageConfig())
        for values in ({'image_size': 0}, {'std': (0, 1, 1)}, {'mean': (float('nan'), 0, 0)},
                       {'resize_policy': 'crop'}, {'apply_exif_orientation': False}):
            with self.assertRaises(ValueError):
                ImageConfig(**values)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch required on rama')
class ImageTests(unittest.TestCase):
    def test_modes_and_known_values(self):
        import torch
        from PIL import Image
        transform = ImagePreprocessor(ImageConfig(image_size=8))
        for mode, color, expected in [('RGB', (0, 255, 0), [-1, 1, -1]),
                                      ('L', 255, [1, 1, 1]),
                                      ('RGBA', (255, 0, 0, 0), [1, -1, -1])]:
            image = Image.new(mode, (3, 7), color)
            before = image.tobytes()
            tensor = transform(image)
            self.assertEqual(tuple(tensor.shape), (3, 8, 8))
            self.assertEqual(tensor.dtype, torch.float32)
            torch.testing.assert_close(tensor[:, 0, 0], torch.tensor(expected, dtype=torch.float32))
            self.assertEqual(image.tobytes(), before)
            torch.testing.assert_close(tensor, transform(image), rtol=0, atol=0)

    def test_exif_orientation_and_no_source_changes(self):
        import torch
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'oriented.png'
            image = oriented_fixture(path)
            with Image.open(path) as saved:
                saved.load()
                self.assertEqual(saved.getexif().get(274), 6)
            raw = path.read_bytes()
            transform = ImagePreprocessor(ImageConfig(image_size=2))
            result = transform(path)
            torch.testing.assert_close(result[:, 0, 0], torch.tensor([-1., -1., 1.]))
            original_pixels = image.tobytes()
            torch.testing.assert_close(transform(image), result, rtol=0, atol=0)
            self.assertEqual(image.tobytes(), original_pixels)
            self.assertEqual(path.read_bytes(), raw)
            self.assertEqual(image.getexif()[274], 6)

    def test_bicubic_resize_and_batch(self):
        import torch
        from PIL import Image
        image = Image.new('RGB', (3, 2))
        image.putdata([(i*40, i*20, i*10) for i in range(6)])
        transform = ImagePreprocessor(ImageConfig(image_size=7))
        reference = image.resize((7, 7), Image.Resampling.BICUBIC)
        expected = torch.tensor([reference.getpixel((x, y)) for y in range(7) for x in range(7)],
                                dtype=torch.float32).reshape(7, 7, 3)
        expected = (expected.permute(2, 0, 1)/255 - .5)/.5
        result = transform.batch([image, image])
        torch.testing.assert_close(result[0], expected)
        torch.testing.assert_close(result[0], result[1], rtol=0, atol=0)
        self.assertTrue(torch.isfinite(result).all())
        self.assertEqual(tuple(result.shape), (2, 3, 7, 7))

    def test_invalid_and_corrupt_inputs(self):
        transform = ImagePreprocessor()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'bad.jpg'
            path.write_bytes(b'not an image')
            for source in (path, Path(tmp)/'missing.jpg', None):
                with self.assertRaises(ValueError):
                    transform(source)
        with self.assertRaises(ValueError):
            transform.batch([])
