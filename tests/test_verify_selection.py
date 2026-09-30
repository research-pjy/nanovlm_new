import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from src.data.verify_selection import verify


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'annotations').mkdir()
        # Fixed seed-42 reference, independent of the implementation under test.
        self.ids = {'held_out': [7, 3], 'train': [2, 8, 5, 6, 9, 4], 'val': [0, 1]}
        self.config = dict(data_root=str(self.root), number_of_images=8,
                           random_seed=42, number_of_held_out=2, train_fraction=0.75)
        self.annotation = {
            'images': [dict(id=i, file_name=f'{i}.jpg') for i in reversed(range(11))],
            'annotations': [dict(id=i*10+j, image_id=i, caption=f'Image {i}, caption {j}')
                            for i in range(11) for j in range(4 if i == 10 else 5)],
        }
        self.manifest = dict(seed=42, num_images=8, num_held_out=2,
                             train_fraction=0.75, splits=['train2017'])
        self.manifest.update({g: [dict(image_id=i, split='train2017', file_name=f'{i}.jpg')
                                 for i in ids] for g, ids in self.ids.items()})
        self.save()

    def save(self):
        (self.root / 'annotations/captions_train2017.json').write_text(json.dumps(self.annotation))
        (self.root / 'image_selection.json').write_text(json.dumps(self.manifest))
        (self.root / 'config.json').write_text(json.dumps(self.config))

    def test_golden_selection_and_no_writes(self):
        before = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        report = verify(self.config)
        self.assertTrue(report['ok'])
        self.assertEqual(report['candidate_count'], 10)
        self.assertEqual(report['excluded_fewer_than_five_captions'], 1)
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()})

    def test_reordered_ids_fail(self):
        self.manifest['train'].reverse()
        self.save()
        self.assertIn('ordered ID mismatches', verify(self.config)['errors'][0])

    def test_cross_split_swap_fails_without_duplicates(self):
        a, b = self.manifest['train'], self.manifest['val']
        a[0], b[0] = b[0], a[0]
        self.save()
        self.assertFalse(verify(self.config)['ok'])

    def test_duplicate_and_missing_ids_fail(self):
        self.manifest['val'] = [self.manifest['train'][0]]
        self.save()
        errors = verify(self.config)['errors']
        self.assertTrue(any('Repeated image ID' in e for e in errors))
        self.assertTrue(any('expected 2 IDs' in e for e in errors))

    def test_parameter_mismatches_fail(self):
        for field, value in [('random_seed', 43), ('number_of_images', 7),
                             ('number_of_held_out', 3), ('train_fraction', 0.9)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                verify({**self.config, field: value})

    def test_source_metadata_mismatch(self):
        self.manifest['train'][0]['split'] = 'val2017'
        self.save()
        self.assertIn('source metadata mismatch', verify(self.config)['errors'][0])

    def test_annotation_order_does_not_change_selection(self):
        self.annotation['images'].reverse()
        self.annotation['annotations'].reverse()
        self.save()
        self.assertTrue(verify(self.config)['ok'])

    def test_invalid_caption_is_not_silently_excluded(self):
        self.annotation['annotations'][0]['caption'] = ''
        self.save()
        with self.assertRaisesRegex(ValueError, 'Malformed caption'):
            verify(self.config)

    def test_duplicate_annotation_id_fails(self):
        self.annotation['images'].append(self.annotation['images'][0])
        self.save()
        with self.assertRaisesRegex(ValueError, 'Duplicate annotation image ID'):
            verify(self.config)

    def test_missing_manifest_cli_fails_without_creating_it(self):
        path = self.root / 'image_selection.json'
        path.unlink()
        result = subprocess.run([sys.executable, '-m', 'src.data.verify_selection',
                                 '--config', str(self.root / 'config.json')],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn('Cannot verify selection', result.stderr)
        self.assertFalse(path.exists())

    def test_mismatch_cli_exits_nonzero(self):
        self.manifest['train'].reverse()
        self.save()
        result = subprocess.run([sys.executable, '-m', 'src.data.verify_selection',
                                 '--config', str(self.root / 'config.json')],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertFalse(json.loads(result.stdout)['ok'])


if __name__ == '__main__':
    unittest.main()
