import json
from pathlib import Path
import tempfile
import unittest
from src.data.report_descriptions import report


class ReportTests(unittest.TestCase):
    def test_reasons_separated_without_changing_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'short.jsonl'
            rows = []
            for i, (count, errors) in enumerate([(24, []), (27, ['word_count_outside_20_25']),
                                                (28, ['word_count_outside_20_25']),
                                                (26, ['word_count_outside_20_25', 'token_limit_without_eos'])]):
                rows.append({'image_id': i, 'short_desc': 'example', 'validation': {'short_desc': {
                    'word_count': count, 'errors': errors, 'status': 'invalid' if errors else 'valid'}}})
            path.write_text('\n'.join(json.dumps(row) for row in rows)+'\n')
            before = path.read_bytes()
            result = report(path, 'short_desc')
            self.assertEqual(result['length_only_flagged'], 2)
            self.assertEqual(result['short_20_27_without_other_errors'], 2)
            self.assertEqual(result['error_counts']['token_limit_without_eos'], 1)
            self.assertEqual(path.read_bytes(), before)
            with path.open('a') as stream:
                stream.write(json.dumps(rows[0])+'\n')
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                report(path, 'short_desc')
