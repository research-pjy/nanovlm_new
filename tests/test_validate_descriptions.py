import json
from pathlib import Path
import tempfile
import unittest
from src.data.validate_descriptions import audit, validate


class DescriptionValidationTests(unittest.TestCase):
    def setUp(self):
        self.policy = dict(min_words=20, max_words=27, exclude_flags=[])

    def raw(self, texts):
        return b''.join((json.dumps(dict(image_id=i, assignment='train' if i == 0 else 'val',
                                        short_desc=text))+'\n').encode() for i, text in enumerate(texts))

    def test_default_preserves_all_and_detects_normalized_duplicates(self):
        raw = self.raw(['Hello world.', ' HELLO\n world. ', '', 'Description: bad'])
        report, retained = audit(raw, 'short_desc', self.policy)
        self.assertEqual(retained, raw)
        self.assertEqual(report['summary']['duplicate_output_groups'], 1)
        self.assertEqual(report['summary']['cross_assignment_duplicate_groups'], 1)
        self.assertEqual(report['summary']['flag_counts']['duplicate_output'], 2)
        self.assertEqual(report['summary']['flag_counts']['empty_output'], 1)
        self.assertIn('malformed_output', report['records'][3]['flags'])

    def test_malformed_lines_and_values_are_reported(self):
        raw = b'not json\n\n[]\n'+self.raw([None, 42, '!!!'])
        report, retained = audit(raw, 'short_desc', self.policy)
        self.assertEqual(report['summary']['records'], 6)
        self.assertEqual(report['summary']['flag_counts']['malformed_json'], 2)
        self.assertEqual(retained, raw)

    def test_filter_removes_only_explicit_flags(self):
        raw = self.raw(['Same.', 'same.', 'different'])
        report, filtered = audit(raw, 'short_desc', {**self.policy, 'exclude_flags': ['duplicate_output']})
        self.assertEqual(report['summary']['excluded_by_policy'], 2)
        self.assertEqual(json.loads(filtered)['image_id'], 2)

    def test_repeated_ids_flagged_separately_from_text(self):
        raw = b'{"image_id":1,"assignment":"train","short_desc":"a"}\n'+b'{"image_id":1,"assignment":"val","short_desc":"b"}\n'
        report, _ = audit(raw, 'short_desc', self.policy)
        self.assertEqual(report['summary']['flag_counts']['duplicate_image_id'], 2)
        self.assertEqual(report['summary']['duplicate_output_groups'], 0)

    def test_multiline_prose_not_malformed(self):
        report, _ = audit(self.raw(['A scene.\nAnother sentence.']), 'short_desc', self.policy)
        self.assertNotIn('malformed_output', report['records'][0]['flags'])

    def test_long_field_and_truncation_evidence(self):
        row = dict(image_id=1, assignment='held_out', long_desc='text',
                   attempts=[dict(text='text', truncated=True)])
        report, _ = audit(json.dumps(row).encode(), 'long_desc', dict(min_words=60,max_words=70,exclude_flags=[]))
        self.assertIn('truncated_output', report['records'][0]['flags'])

    def test_unknown_filter_rejected(self):
        with self.assertRaises(ValueError):
            audit(b'', 'short_desc', {**self.policy, 'exclude_flags': ['made_up']})

    def test_artifacts_do_not_modify_source_and_are_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, config, report, subset = [root / n for n in ('input.jsonl','config.json','report.json','subset.jsonl')]
            raw = self.raw(['', 'Retain this.'])
            source.write_bytes(raw)
            config.write_text(json.dumps({'short_desc': {**self.policy, 'exclude_flags': ['empty_output']}}))
            validate(source, config, 'short_desc', report, subset)
            self.assertEqual(source.read_bytes(), raw)
            self.assertEqual(json.loads(subset.read_bytes())['image_id'], 1)
            before = report.stat().st_mtime_ns
            validate(source, config, 'short_desc', report, subset)
            self.assertEqual(report.stat().st_mtime_ns, before)
            with self.assertRaisesRegex(ValueError, 'distinct'):
                validate(source, config, 'short_desc', source)
            subset.write_bytes(b'existing artifact')
            with self.assertRaisesRegex(ValueError, 'differs'):
                validate(source, config, 'short_desc', report, subset)
            self.assertEqual(subset.read_bytes(), b'existing artifact')


if __name__ == '__main__':
    unittest.main()
