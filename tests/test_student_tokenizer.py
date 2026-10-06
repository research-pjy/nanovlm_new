import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from src.tokenization.student import digest, training_corpus, validate_text


def dataset(root):
    records = [{'image_id': i, 'split': 'train', 'image_path': f'{i}.jpg',
                'short_desc': 'A small cat sits beside the big blue chair today.',
                'long_desc': 'A small cat sits beside the big blue chair today. '*5}
               for i in (2, 1)]
    raw = ('\n'.join(json.dumps(r) for r in records)+'\n').encode()
    (root/'train.jsonl').write_bytes(raw)
    (root/'manifest.json').write_text(json.dumps({'schema_version': 1,
        'files': {'train.jsonl': digest(raw)}, 'counts': {'train': 2}}))
    return records


class CorpusTests(unittest.TestCase):
    def test_train_only_and_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = dataset(root)
            texts, provenance = training_corpus(root)
            self.assertEqual(len(texts), 4)
            self.assertEqual(texts[0], records[1]['short_desc'])
            (root/'val.jsonl').write_text('not read by fitting')
            self.assertEqual(training_corpus(root), (texts, provenance))
            (root/'train.jsonl').write_text('changed')
            with self.assertRaises(ValueError):
                training_corpus(root)

    def test_reserved_strings_rejected(self):
        for text in ('hello <pad>', '<bos>', '<eos>'):
            with self.assertRaises(ValueError):
                validate_text(text)


@unittest.skipUnless(importlib.util.find_spec('tokenizers'), 'tokenizers package required')
class StudentTests(unittest.TestCase):
    def test_roundtrip_task_masks_reload_and_reproducibility(self):
        from src.tokenization.student import train, StudentTokenizer
        from src.tasks.builder import TaskBuilder, collate
        from src.models.config import ModelConfig
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'data'
            source.mkdir()
            records = dataset(source)
            first, status = train(source, root/'first')
            self.assertEqual(status, 'created')
            second, _ = train(source, root/'second')
            self.assertEqual(first.artifact_sha256, second.artifact_sha256)
            self.assertEqual(train(source, root/'first')[1], 'unchanged')
            loaded = StudentTokenizer(root/'first')
            for text in (' leading  spaces\n\tnew line', 'café 日本語 🐱', '', 'unknown𐍈'):
                ids = loaded.encode(text)
                self.assertEqual(loaded.decode(ids), text)
                self.assertTrue(all(3 <= i < loaded.vocabulary_size for i in ids))
            with self.assertRaises(ValueError):
                loaded.encode('<eos>')
            with self.assertRaises(ValueError):
                loaded.encode('text', add_special_tokens=True)
            (source/'1.jpg').touch()
            builder = TaskBuilder(loaded, source, bos_token_id=loaded.bos_token_id,
                                  eos_token_id=loaded.eos_token_id, image_loader=lambda p: None)
            example = builder.build(records[1], 'short')
            self.assertEqual(example.tokenized_prompt[0], 1)
            self.assertEqual(example.tokenized_target[-1], 2)
            self.assertEqual(loaded.decode(example.input_ids), records[1]['short_desc'])
            self.assertEqual(example.loss_mask, [0]*len(example.tokenized_prompt)+[1]*len(example.tokenized_target))
            batch = collate([example, builder.build(records[1], 'long')], pad_token_id=0)
            self.assertEqual(batch['loss_mask'][0][-1], 0)
            self.assertEqual(loaded.resolve_model_config(ModelConfig()).vocabulary_size, loaded.vocabulary_size)
            with self.assertRaises(ValueError):
                loaded.resolve_model_config(ModelConfig(vocabulary_size=1))
            path = root/'first/tokenizer.json'
            path.write_bytes(path.read_bytes()+b' ')
            with self.assertRaises(ValueError):
                StudentTokenizer(root/'first')

    def test_incomplete_artifacts_do_not_refit(self):
        from src.tokenization.student import train
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset(root)
            (root/'broken').mkdir()
            with self.assertRaises(FileNotFoundError):
                train(root, root/'broken')
