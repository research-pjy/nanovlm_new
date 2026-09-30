import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from src.data.teachers.qwen import resolve_model_path, validate_weights


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_partial_snapshot_without_readme_is_usable(self):
        snapshot = self.root / 'snapshots/commit'
        snapshot.mkdir(parents=True)
        blob = self.root / 'config-blob'
        blob.write_text('{}')
        (snapshot / 'config.json').symlink_to(blob)
        fetch = Mock(return_value=str(snapshot / 'config.json'))
        hub = types.SimpleNamespace(hf_hub_download=fetch)
        with patch.dict('sys.modules', {'huggingface_hub': hub}):
            actual = resolve_model_path({'model': 'Qwen/Qwen3-VL-8B-Instruct', 'revision': 'main'})
        self.assertEqual(actual, snapshot)
        self.assertFalse((snapshot / 'README.md').exists())
        fetch.assert_called_once_with('Qwen/Qwen3-VL-8B-Instruct', filename='config.json',
                                      revision='main', local_files_only=True)

    def test_local_directory_needs_no_hub_import(self):
        with patch.dict('sys.modules', {'huggingface_hub': None}):
            self.assertEqual(resolve_model_path({'model': str(self.root), 'revision': 'main'}), self.root)

    def test_cache_miss_propagates_without_network_retry(self):
        fetch = Mock(side_effect=OSError('not cached'))
        with patch.dict('sys.modules', {'huggingface_hub': types.SimpleNamespace(hf_hub_download=fetch)}):
            with self.assertRaisesRegex(OSError, 'not cached'):
                resolve_model_path({'model': 'Qwen/example', 'revision': 'commit123'})
        self.assertEqual(fetch.call_count, 1)
        self.assertTrue(fetch.call_args.kwargs['local_files_only'])
        self.assertEqual(fetch.call_args.kwargs['revision'], 'commit123')

    def test_missing_shard_rejected(self):
        (self.root / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'a': 'shard.safetensors'}}))
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            validate_weights(self.root)
        (self.root / 'shard.safetensors').write_bytes(b'fixture')
        validate_weights(self.root)

    def test_shard_without_index_rejected(self):
        (self.root / 'model-00001-of-00002.safetensors').write_bytes(b'fixture')
        with self.assertRaisesRegex(ValueError, 'shard index'):
            validate_weights(self.root)

    def test_unsharded_checkpoint(self):
        (self.root / 'model.safetensors').write_bytes(b'fixture')
        validate_weights(self.root)


if __name__ == '__main__':
    unittest.main()
