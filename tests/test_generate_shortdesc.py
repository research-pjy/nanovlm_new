import contextlib
import io
import fcntl
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from src.data.generate_shortdesc import load_inputs, run, validation
from src.data.teachers.base import TeacherOutput

VALID = ' '.join(f'word{i}' for i in range(22))


class FakeTeacher:
    identity = {'backend': 'test', 'teacher_model': 'test-only', 'runtime': {'version': 1}}

    def __init__(self, fail_call=None, invalid=False, first_invalid=False):
        self.calls = []
        self.fail_call = fail_call
        self.invalid = invalid
        self.first_invalid = first_invalid

    def generate_batch(self, prompts, seed):
        self.calls.append((prompts, seed))
        if len(self.calls) == self.fail_call:
            raise RuntimeError('simulated interruption')
        text = 'Too short.' if self.invalid or (self.first_invalid and len(self.calls) == 1) else VALID
        return [TeacherOutput(text) for _ in prompts]

    def memory_stats(self):
        return {}


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.out = self.root / 'generated'
        self.config = json.loads(Path('configs/shortdesc.qwen.json').read_text())
        self.records = [dict(image_id=i, assignment='train', image_path=f'images/train2017/{i}.jpg',
                             captions=[f'Original {j}' for j in range(6)], caption_ids=list(range(6)))
                        for i in range(35)]
        self.provenance = {'metadata_sha256': 'fixture', 'splits_sha256': 'fixture'}

    def execute(self, teacher=None, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return run(self.records, self.provenance, self.config,
                       teacher or FakeTeacher(), self.out, **kwargs)

    def output(self):
        return [json.loads(line) for line in (self.out / 'shortdesc.jsonl').read_text().splitlines()]

    def test_batches_and_resume_no_regeneration(self):
        first = FakeTeacher()
        self.assertFalse(self.execute(first, max_batches=1)['complete'])
        self.assertEqual([len(p) for p, _ in first.calls], [16])
        second = FakeTeacher()
        self.assertTrue(self.execute(second)['complete'])
        self.assertEqual([len(p) for p, _ in second.calls], [16, 3])
        self.assertEqual([seed for _, seed in second.calls], [45, 48])
        self.assertEqual([r['image_id'] for r in self.output()], list(range(35)))
        third = FakeTeacher()
        self.execute(third)
        self.assertEqual(third.calls, [])

    def test_failure_preserves_committed_batch(self):
        with self.assertRaisesRegex(RuntimeError, 'interruption'):
            self.execute(FakeTeacher(fail_call=2))
        self.assertEqual(len(self.output()), 16)
        self.assertTrue(self.execute()['complete'])

    def test_resume_matches_uninterrupted_text_and_seeds(self):
        self.execute(max_batches=1)
        self.execute()
        resumed = [(r['short_desc'], r['generation_seed']) for r in self.output()]
        self.out = self.root / 'fresh'
        self.execute()
        self.assertEqual(resumed, [(r['short_desc'], r['generation_seed']) for r in self.output()])

    def test_retry_and_provenance(self):
        self.execute(FakeTeacher(first_invalid=True), max_batches=1)
        record = self.output()[0]
        self.assertEqual(len(record['attempts']), 2)
        self.assertEqual(record['validation']['short_desc']['status'], 'valid')
        self.assertEqual(record['source_captions'], self.records[0]['captions'])
        self.assertEqual(record['generation_seed'], 43)
        self.assertIsNone(record['long_desc'])
        self.assertEqual(record['validation']['long_desc']['status'], 'not_generated')
        self.assertIn('+00:00', record['generation_timestamp'])

    def test_invalid_retained_and_not_retried_forever(self):
        result = self.execute(FakeTeacher(invalid=True), max_batches=1)
        self.assertEqual(result['validation_counts'], {'invalid': 16})
        self.assertEqual(len(self.output()[0]['attempts']), 3)
        teacher = FakeTeacher()
        self.execute(teacher, max_batches=1)
        self.assertEqual(len(teacher.calls), 1)
        self.assertEqual(self.output()[0]['validation']['short_desc']['status'], 'invalid')

    def test_config_change_rejected(self):
        self.execute(max_batches=1)
        self.config['batch_size'] = 8
        with self.assertRaisesRegex(ValueError, 'changed'):
            self.execute()
        self.assertEqual(len(self.output()), 16)

    def test_teacher_change_rejected(self):
        self.execute(max_batches=1)
        teacher = FakeTeacher()
        teacher.identity = {**teacher.identity, 'teacher_model': 'different'}
        with self.assertRaisesRegex(ValueError, 'changed'):
            self.execute(teacher)

    def test_input_fingerprint_change_rejected(self):
        self.execute(max_batches=1)
        self.provenance['metadata_sha256'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'changed'):
            self.execute()

    def test_incomplete_checkpoint_rejected(self):
        self.execute(max_batches=1)
        with contextlib.closing(sqlite3.connect(self.out / 'checkpoint.sqlite3')) as db:
            with db:
                db.execute('DELETE FROM records WHERE position=15')
        with self.assertRaisesRegex(ValueError, 'incomplete batch'):
            self.execute()

    def test_wrong_teacher_response_count_no_partial_save(self):
        teacher = FakeTeacher()
        teacher.generate_batch = lambda prompts, seed: []
        with self.assertRaisesRegex(ValueError, 'wrong number'):
            self.execute(teacher)
        self.assertEqual(self.output(), [])

    def test_export_without_teacher(self):
        self.execute(max_batches=1)
        (self.out / 'shortdesc.jsonl').unlink()
        result = self.execute(export_only=True)
        self.assertEqual(result['saved'], 16)
        self.assertEqual(len(self.output()), 16)

    def test_concurrent_writer_refused(self):
        self.out.mkdir()
        with (self.out / '.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(ValueError, 'Another process'):
                self.execute()

    def test_transaction_rolls_back_whole_batch_on_write_failure(self):
        self.execute(max_batches=1)
        with contextlib.closing(sqlite3.connect(self.out / 'checkpoint.sqlite3')) as db:
            with db:
                db.execute("CREATE TRIGGER fail_insert BEFORE INSERT ON records "
                           "WHEN NEW.position=18 BEGIN SELECT RAISE(ABORT, 'test failure'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            self.execute()
        self.assertEqual(len(self.output()), 16)
        with contextlib.closing(sqlite3.connect(self.out / 'checkpoint.sqlite3')) as db:
            with db:
                db.execute('DROP TRIGGER fail_insert')
        self.assertTrue(self.execute()['complete'])

    def test_validation(self):
        self.assertEqual(validation(VALID)['status'], 'valid')
        self.assertEqual(validation(VALID, True)['status'], 'invalid')
        self.assertEqual(validation('')['status'], 'invalid')
        self.assertEqual(validation('<think> '+VALID)['status'], 'invalid')
        self.assertEqual(validation('Description: '+VALID)['status'], 'invalid')

    def test_inputs_require_matching_metadata_fingerprint(self):
        metadata = self.root / 'metadata.json'
        splits = self.root / 'splits.json'
        metadata.write_text(json.dumps({'schema_version': 1}))
        splits.write_text(json.dumps({'schema_version': 1, 'metadata_sha256': 'wrong'}))
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            load_inputs(metadata, splits, self.config)


if __name__ == '__main__':
    unittest.main()
