import json
import shutil
import subprocess
import sys
import unittest

import test_export_metadata as fixtures
from src.data.export_metadata import export as export_metadata
from src.data.export_splits import export


class SplitTests(unittest.TestCase):
    save = fixtures.MetadataTests.save

    def setUp(self):
        fixtures.MetadataTests.setUp(self)
        export_metadata(self.config, self.output)
        self.metadata_path = self.output
        self.split_path = self.base / 'splits.json'

    def update_metadata(self, change):
        metadata = json.loads(self.metadata_path.read_text())
        change(metadata)
        self.metadata_path.write_text(json.dumps(metadata))

    def test_exact_ids_preserved_and_rerun_unchanged(self):
        before = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        report = export(self.config, self.metadata_path, self.split_path)
        self.assertEqual(report['counts'], dict(train=1, val=1, held_out=1))
        self.assertEqual(set(report['pairwise_overlap'].values()), {0})
        document = json.loads(self.split_path.read_text())
        self.assertEqual(document['image_ids'], dict(train=[0], val=[2], held_out=[1]))
        modified = self.split_path.stat().st_mtime_ns
        self.assertEqual(export(self.config, self.metadata_path, self.split_path)['status'], 'unchanged')
        self.assertEqual(self.split_path.stat().st_mtime_ns, modified)
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()})

    def test_leakage_rejected(self):
        self.update_metadata(lambda m: m['records'][1].update(image_id=0))
        with self.assertRaisesRegex(ValueError, 'Duplicate image ID'):
            export(self.config, self.metadata_path, self.split_path)
        self.assertFalse(self.split_path.exists())

    def test_changed_assignment_rejected(self):
        self.update_metadata(lambda m: m['records'][0].update(assignment='val'))
        with self.assertRaisesRegex(ValueError, 'disagree with verified selection'):
            export(self.config, self.metadata_path, self.split_path)

    def test_missing_record_rejected(self):
        self.update_metadata(lambda m: m['records'].pop())
        with self.assertRaisesRegex(ValueError, 'disagree with verified selection'):
            export(self.config, self.metadata_path, self.split_path)

    def test_changed_caption_rejected(self):
        self.update_metadata(lambda m: m['records'][0]['captions'].append('Not original'))
        with self.assertRaisesRegex(ValueError, 'differs from verified source'):
            export(self.config, self.metadata_path, self.split_path)

    def test_stale_provenance_rejected(self):
        self.update_metadata(lambda m: m.update(source_manifest_sha256='stale'))
        with self.assertRaisesRegex(ValueError, 'differs from verified source'):
            export(self.config, self.metadata_path, self.split_path)

    def test_existing_output_not_overwritten(self):
        self.split_path.write_bytes(b'old split artifact')
        with self.assertRaisesRegex(ValueError, 'Existing export differs'):
            export(self.config, self.metadata_path, self.split_path)
        self.assertEqual(self.split_path.read_bytes(), b'old split artifact')

    def test_portable_across_roots(self):
        export(self.config, self.metadata_path, self.split_path)
        relocated = self.base / 'rama/coco'
        shutil.copytree(self.root, relocated)
        second = self.base / 'second.json'
        export({**self.config, 'data_root': str(relocated)}, self.metadata_path, second)
        self.assertEqual(self.split_path.read_bytes(), second.read_bytes())

    def test_input_cannot_be_output(self):
        with self.assertRaisesRegex(ValueError, 'must not be the metadata'):
            export(self.config, self.metadata_path, self.metadata_path)

    def test_missing_metadata_cli_nonzero(self):
        config_path = self.base / 'config.json'
        config_path.write_text(json.dumps(self.config))
        result = subprocess.run([sys.executable, '-m', 'src.data.export_splits',
                                 '--config', str(config_path), '--metadata', str(self.base / 'missing.json'),
                                 '--output', str(self.split_path)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn('Cannot export splits', result.stderr)
        self.assertFalse(self.split_path.exists())


if __name__ == '__main__':
    unittest.main()
