import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
from src.data.verify_existing import verify


class ExistingDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'images/train2017').mkdir(parents=True)
        (self.root / 'annotations').mkdir()
        images, annotations, entries = [], [], []
        for i in range(3):
            filename = f'{i:012d}.jpg'
            Image.new('RGB', (8, 8), color=(i * 50, 0, 0)).save(self.root / 'images/train2017' / filename)
            images.append(dict(id=i, file_name=filename, width=8, height=8))
            captions = [f'Caption {i} {j}' for j in range(5)]
            annotations.extend(dict(image_id=i, caption=c) for c in captions)
            entries.append(dict(image_id=i, file_name=filename, split='train2017', coco_captions=captions))
        self.manifest = dict(seed=42, num_images=2, num_held_out=1, splits=['train2017'], train=entries[:1], val=entries[1:2], held_out=entries[2:])
        (self.root / 'annotations/captions_train2017.json').write_text(json.dumps(dict(images=images, annotations=annotations)))
        self.save()
        self.config = dict(data_root=str(self.root), number_of_images=2, random_seed=42)

    def save(self):
        (self.root / 'image_selection.json').write_text(json.dumps(self.manifest))

    def check_data(self):
        return verify(self.config, progress=lambda *args, **kwargs: None)

    def test_valid_and_read_only(self):
        before = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        self.assertTrue(self.check_data()['ok'])
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()})

    def test_wrong_seed_refused(self):
        self.config['random_seed'] = 404
        with self.assertRaisesRegex(ValueError, 'seed'):
            self.check_data()

    def test_leakage_reported(self):
        self.manifest['val'] = self.manifest['train']
        self.save()
        self.assertIn('Repeated image ID', self.check_data()['errors'][0])

    def test_corrupt_and_missing_images_both_reported(self):
        (self.root / 'images/train2017/000000000000.jpg').write_bytes(b'broken')
        (self.root / 'images/train2017/000000000001.jpg').unlink()
        report = self.check_data()
        self.assertFalse(report['ok'])
        self.assertEqual(len(report['errors']), 2)

    def test_caption_mismatch_reported(self):
        self.manifest['train'][0]['coco_captions'][0] = 'Changed caption'
        self.save()
        self.assertIn('captions disagree', self.check_data()['errors'][0])


if __name__ == '__main__':
    unittest.main()
