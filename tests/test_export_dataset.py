import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from src.data.export_dataset import encode, export


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.meta = self.root / 'metadata.json'
        self.splits = self.root / 'splits.json'
        self.short = self.root / 'short'
        self.long = self.root / 'long'
        self.output = self.root / 'final'
        self.records = [dict(image_id=i, assignment=g, image_path=f'images/train2017/{i}.jpg',
                             source_split='train2017', captions=['original']*5, caption_ids=list(range(5)))
                        for i,g in enumerate(('train','val','held_out'))]
        common = dict(schema_version=1, random_seed=42, source_manifest_sha256='a', source_annotation_sha256={})
        counts = dict(train=1,val=1,held_out=1)
        self.meta.write_text(json.dumps({**common,'records': self.records,'assignments':counts}))
        self.splits.write_text(json.dumps({**common,'metadata_sha256':self.digest(self.meta),'counts':counts,
                                           'image_ids': dict(train=[0],val=[1],held_out=[2])}))
        for directory, phase, field in ((self.short,'short','short_desc'),(self.long,'long','long_desc')):
            directory.mkdir()
            contract = dict(sources=dict(metadata_sha256=self.digest(self.meta),splits_sha256=self.digest(self.splits)),
                            phase=phase, configuration={},teacher={'teacher_model':'test'}, total_records=3)
            (directory / 'run.json').write_text(json.dumps(contract))
            signature = hashlib.sha256(encode(contract).encode()).hexdigest()
            rows = []
            for r in reversed(self.records):
                rows.append(dict(image_id=r['image_id'],assignment=r['assignment'],image_path=r['image_path'],
                                 source_captions=r['captions'],source_caption_ids=r['caption_ids'],
                                 **{field: f'{phase} description {r["image_id"]}'},run_signature=signature,
                                 generation_configuration={},teacher_model='test',generation_seed=42,
                                 generation_timestamp='2026-10-01T00:00:00+00:00',
                                 validation={field:{'status':'invalid','errors':['word_count_outside_range']}}))
            (directory / f'{phase}desc.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))

    def digest(self,path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def execute(self):
        return export(self.meta,self.splits,self.short,self.long,self.output)

    def mutate(self, change):
        path=self.long/'longdesc.jsonl'
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        change(rows)
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))

    def test_join_by_id_retains_flags_and_maps_test(self):
        before=self.digest(self.long/'longdesc.jsonl')
        self.assertEqual(self.execute()['counts'],dict(train=1,val=1,test=1))
        for i, split in enumerate(('train','val','test')):
            row=json.loads((self.output/f'{split}.jsonl').read_text())
            self.assertEqual(row['short_desc'],f'short description {i}')
            self.assertEqual(row['long_desc'],f'long description {i}')
            self.assertEqual(row['split'],split)
            self.assertEqual(row['validation']['short_desc']['status'],'invalid')
        self.assertEqual(self.digest(self.long/'longdesc.jsonl'),before)
        self.assertEqual(self.execute()['status'],'unchanged')

    def test_missing_id_refused(self):
        self.mutate(lambda rows:rows.pop())
        with self.assertRaisesRegex(ValueError,'missing 1 IDs'): self.execute()
        self.assertFalse(self.output.exists())

    def test_duplicate_id_refused(self):
        self.mutate(lambda rows:rows.append(rows[0]))
        with self.assertRaisesRegex(ValueError,'duplicate'): self.execute()

    def test_assignment_change_refused(self):
        self.mutate(lambda rows:rows[0].update(assignment='train'))
        with self.assertRaisesRegex(ValueError,'assignment mismatch'): self.execute()

    def test_caption_change_refused(self):
        self.mutate(lambda rows:rows[0].update(source_captions=['changed']))
        with self.assertRaisesRegex(ValueError,'source_captions mismatch'): self.execute()

    def test_mixed_run_refused(self):
        self.mutate(lambda rows:rows[0].update(run_signature='other'))
        with self.assertRaisesRegex(ValueError,'signature'): self.execute()

    def test_different_existing_dataset_preserved(self):
        self.execute()
        path=self.output/'train.jsonl'; path.write_bytes(b'previous')
        with self.assertRaisesRegex(ValueError,'Existing dataset differs'): self.execute()
        self.assertEqual(path.read_bytes(),b'previous')

    def test_output_cannot_overlap_input(self):
        self.output=self.short/'export'
        with self.assertRaisesRegex(ValueError,'separate'): self.execute()


if __name__=='__main__': unittest.main()
