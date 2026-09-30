import contextlib
import hashlib
import io
import json
from pathlib import Path
import unittest

import test_export_dataset as fixtures
from src.data.check_dataset import check, reload_dataset
from src.data.prepare_small import subset
from src.data.generate_shortdesc import run as run_short
from src.data.generate_longdesc import run as run_long
from src.data.export_dataset import export
from src.data.teachers.base import TeacherOutput
from src.data.validate_descriptions import audit


class AcceptanceTests(unittest.TestCase):
    digest=fixtures.DatasetTests.digest

    def setUp(self):
        fixtures.DatasetTests.setUp(self)
        self.images=self.root/'images_root'
        for r in self.records:
            path=self.images/r['image_path']; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(b'fixture')
        export(self.meta,self.splits,self.short,self.long,self.output)

    def execute(self):
        return check(self.output,self.images,self.meta,self.splits)

    def change(self, split, callback):
        path=self.output/(split+'.jsonl')
        rows=[json.loads(s) for s in path.read_text().splitlines()]; callback(rows)
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        manifest=json.loads((self.output/'manifest.json').read_text())
        manifest['files'][split+'.jsonl']=self.digest(path)
        manifest['counts'][split]=len(rows)
        (self.output/'manifest.json').write_text(json.dumps(manifest))

    def test_reload_and_expected_counts(self):
        result=self.execute()
        self.assertTrue(result['ok'])
        self.assertEqual(result['selected_images'],3)
        self.assertTrue(result['reload_verified'])

    def test_missing_file(self):
        (self.images/self.records[0]['image_path']).unlink()
        self.assertTrue(any('missing image' in e for e in self.execute()['errors']))

    def test_missing_captions(self):
        self.change('train',lambda rows:rows[0].update(source_captions=[]))
        self.assertFalse(self.execute()['ok'])

    def test_duplicate_and_leakage(self):
        self.change('val',lambda rows:rows[0].update(image_id=0))
        self.assertTrue(any('leakage' in e for e in self.execute()['errors']))

    def test_missing_description(self):
        for field in ('short_desc','long_desc'):
            with self.subTest(field=field):
                self.change('train',lambda rows:rows[0].update({field:None}))
                self.assertTrue(any('unavailable '+field in e for e in self.execute()['errors']))

    def test_length_deviation_is_advisory(self):
        result=self.execute()
        self.assertTrue(result['ok'])
        self.assertEqual(result['length_warnings']['short_outside_20_27'],3)
        self.assertEqual(result['length_warnings']['long_outside_60_70'],3)

    def test_word_counts_recomputed_from_saved_text(self):
        self.change('train',lambda rows:rows[0].update(short_desc=' '.join(['word']*27),long_desc=' '.join(['word']*70)))
        result=self.execute()
        self.assertEqual(result['word_count_histograms']['short_desc'][27],1)
        self.assertEqual(result['word_count_histograms']['long_desc'][70],1)
        self.assertEqual(result['length_warnings']['short_outside_20_27'],2)
        self.change('train',lambda rows:rows[0].update(short_desc=' '.join(['word']*28),long_desc=' '.join(['word']*71)))
        result=self.execute()
        self.assertTrue(result['ok'])
        self.assertEqual(result['length_warnings']['short_outside_20_27'],3)
        self.assertEqual(result['length_warnings']['long_outside_60_70'],3)

    def test_missing_selected_record(self):
        self.change('train',lambda rows:rows.clear())
        self.assertFalse(self.execute()['ok'])

    def test_checksum_failure(self):
        (self.output/'train.jsonl').write_text('changed')
        with self.assertRaisesRegex(ValueError,'checksum'): reload_dataset(self.output)

    def test_100_image_complete_pipeline_with_test_teachers(self):
        metadata=json.loads(self.meta.read_text())
        records=[]
        for i in range(281):
            group='train' if i<252 else 'val' if i<280 else 'held_out'
            records.append({**self.records[0],'image_id':i,'assignment':group,'image_path':f'images/train2017/{i}.jpg'})
            path=self.images/records[-1]['image_path'];path.write_bytes(b'fixture')
        metadata.update(records=records,assignments=dict(train=252,val=28,held_out=1))
        small=subset(metadata,100)
        self.assertEqual(small,subset(metadata,100))
        self.assertEqual(small['assignments'],dict(train=89,val=10,held_out=1))
        self.assertEqual(len({r['image_id'] for r in small['records']}),100)
        self.assertEqual({r['image_id']:r['assignment'] for r in small['records']},
                         {r['image_id']:records[r['image_id']]['assignment'] for r in small['records']})
        with self.assertRaises(ValueError): subset(metadata,100,404)
        with self.assertRaises(ValueError): subset(metadata,282)
        meta=self.root/'small_metadata.json';meta.write_text(json.dumps(small))
        split=self.root/'small_splits.json'
        spec=json.loads(self.splits.read_text());spec.update(metadata_sha256=self.digest(meta),counts=small['assignments'],
             image_ids={g:[r['image_id'] for r in small['records'] if r['assignment']==g] for g in small['assignments']})
        split.write_text(json.dumps(spec))
        provenance=dict(metadata_sha256=self.digest(meta),splits_sha256=self.digest(split))
        class TestTeacher:
            identity={'teacher_model':'test-only'}
            def __init__(self,n): self.n=n
            def generate_batch(self,prompts,seed): return [TeacherOutput(' '.join(['word']*self.n)) for p in prompts]
            def memory_stats(self): return {}
        with contextlib.redirect_stdout(io.StringIO()):
            for phase,runner,n in [('short',run_short,22),('long',run_long,65)]:
                config=json.loads(Path(f'configs/{phase}desc.qwen.json').read_text())
                runner(small['records'],provenance,config,TestTeacher(n),self.root/('smoke_'+phase))
                raw=(self.root/('smoke_'+phase)/(phase+'desc.jsonl')).read_bytes()
                report,_=audit(raw,phase+'_desc',dict(min_words=20 if phase=='short' else 60,max_words=27 if phase=='short' else 70,exclude_flags=[]))
                self.assertEqual(report['summary']['records'],100)
        final=self.root/'small_final'
        export(meta,split,self.root/'smoke_short',self.root/'smoke_long',final)
        self.assertTrue(check(final,self.images,meta,split)['ok'])


if __name__=='__main__': unittest.main()
