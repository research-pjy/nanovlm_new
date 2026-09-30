import json
from pathlib import Path
import shutil
import tempfile
import unittest

from src.data.export_metadata import export


class MetadataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = self.base / 'coco'
        (self.root / 'annotations').mkdir(parents=True)
        (self.root / 'images/train2017').mkdir(parents=True)
        self.config = dict(data_root=str(self.root), number_of_images=2,
                           random_seed=42, number_of_held_out=1, train_fraction=0.5)
        self.annotation = {'images': [], 'annotations': []}
        self.manifest = dict(seed=42, num_images=2, num_held_out=1,
                             train_fraction=0.5, splits=['train2017'])
        for group, i in [('held_out', 1), ('train', 0), ('val', 2)]:
            texts = [f' Original café caption {i}:{j}.\n' for j in range(6 if i == 0 else 5)]
            self.annotation['images'].append(dict(id=i, file_name=f'{i}.jpg'))
            self.annotation['annotations'].extend(
                dict(id=i*10+j, image_id=i, caption=c) for j, c in enumerate(texts))
            self.manifest[group] = [dict(image_id=i, file_name=f'{i}.jpg',
                                         split='train2017', coco_captions=texts[:5])]
            (self.root / f'images/train2017/{i}.jpg').write_bytes(b'Image decoding belongs to Phase 1A')
        self.output = self.base / 'output/metadata.json'
        self.save()

    def save(self):
        (self.root / 'annotations/captions_train2017.json').write_text(json.dumps(self.annotation))
        (self.root / 'image_selection.json').write_text(json.dumps(self.manifest))

    def test_all_captions_exactly_retained_and_source_unchanged(self):
        before = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        report = export(self.config, self.output)
        self.assertEqual(report['captions'], 16)
        data = json.loads(self.output.read_text())
        self.assertEqual([r['assignment'] for r in data['records']], ['train', 'val', 'held_out'])
        for record in data['records']:
            original = [a for a in self.annotation['annotations'] if a['image_id'] == record['image_id']]
            self.assertEqual(record['captions'], [a['caption'] for a in original])
            self.assertEqual(record['caption_ids'], [a['id'] for a in original])
            self.assertFalse(Path(record['image_path']).is_absolute())
        self.assertNotIn(str(self.root), self.output.read_text())
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()})

    def test_idempotent_without_rewriting(self):
        export(self.config, self.output)
        before = self.output.stat().st_mtime_ns
        self.assertEqual(export(self.config, self.output)['status'], 'unchanged')
        self.assertEqual(self.output.stat().st_mtime_ns, before)

    def test_differing_export_is_preserved(self):
        self.output.parent.mkdir()
        self.output.write_bytes(b'existing artifact')
        with self.assertRaisesRegex(ValueError, 'Existing export differs'):
            export(self.config, self.output)
        self.assertEqual(self.output.read_bytes(), b'existing artifact')
        self.assertEqual(list(self.output.parent.glob('.metadata-*')), [])

    def test_relocation_produces_identical_bytes(self):
        export(self.config, self.output)
        relocated = self.base / 'another-machine/coco'
        shutil.copytree(self.root, relocated)
        second = self.base / 'second.json'
        export({**self.config, 'data_root': str(relocated)}, second)
        self.assertEqual(self.output.read_bytes(), second.read_bytes())

    def test_missing_images_are_all_reported_without_output(self):
        for i in (0, 2):
            (self.root / f'images/train2017/{i}.jpg').unlink()
        with self.assertRaises(ValueError) as error:
            export(self.config, self.output)
        self.assertIn('image_id=0', str(error.exception))
        self.assertIn('image_id=2', str(error.exception))
        self.assertFalse(self.output.exists())

    def test_bad_caption_fails_without_output(self):
        self.annotation['annotations'][0]['caption'] = '  '
        self.save()
        with self.assertRaisesRegex(ValueError, 'Malformed caption'):
            export(self.config, self.output)
        self.assertFalse(self.output.exists())

    def test_first_five_mismatch_fails(self):
        self.manifest['train'][0]['coco_captions'][0] = 'Changed'
        self.save()
        with self.assertRaisesRegex(ValueError, 'saved first five captions'):
            export(self.config, self.output)

    def test_unsafe_path_fails(self):
        self.annotation['images'][0]['file_name'] = '../escape.jpg'
        self.manifest['held_out'][0]['file_name'] = '../escape.jpg'
        self.save()
        with self.assertRaisesRegex(ValueError, 'unsafe source filename'):
            export(self.config, self.output)

    def test_source_output_refused(self):
        with self.assertRaisesRegex(ValueError, 'outside the source'):
            export(self.config, self.root / 'metadata.json')


if __name__ == '__main__':
    unittest.main()
