import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
from src.tasks.builder import TaskBuilder, TaskConfig, build_text, collate
from src.tasks.check_tasks import DiagnosticByteTokenizer, check
from src.data.task_dataset import TaskDataset


class TaskTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.images = self.root / 'coco'
        (self.images / 'images/train2017').mkdir(parents=True)
        Image.new('L', (13, 17), 128).save(self.images / 'images/train2017/1.png')
        self.record = {'image_id': 1, 'image_path': 'images/train2017/1.png', 'split': 'train',
                       'short_desc': '  A small café has red chairs beside a big window.\nTwo people sit at the table.  ',
                       'long_desc': ' '.join(f'word{i}' for i in range(65))}
        self.builder = TaskBuilder(DiagnosticByteTokenizer(), self.images)

    def build(self, variant='short', **kwargs):
        return self.builder.build({**self.record, **kwargs}, variant)

    def make_dataset(self, rows=None):
        directory = self.root / 'final'
        directory.mkdir(exist_ok=True)
        rows = rows or [self.record]
        manifest = {'schema_version': 1, 'files': {}, 'counts': {}}
        for split in ('train', 'val', 'test'):
            raw = ''.join(json.dumps({**r, 'split': split, 'image_id': r['image_id'] + (100 if split=='val' else 200 if split=='test' else 0)})+'\n' for r in rows).encode()
            filename = split + '.jsonl'
            (directory / filename).write_bytes(raw)
            manifest['files'][filename] = hashlib.sha256(raw).hexdigest()
            manifest['counts'][split] = len(rows)
        (directory / 'manifest.json').write_text(json.dumps(manifest))
        return directory

    def test_all_prefix_lengths_reachable_and_order_independent(self):
        for variant, allowed in [('short', {6, 7}), ('long', {18, 19, 20})]:
            forward = {i:build_text({**self.record, 'image_id': i},variant) for i in range(100)}
            reverse = {i:build_text({**self.record, 'image_id': i},variant) for i in reversed(range(100))}
            self.assertEqual(forward, reverse)
            self.assertEqual({t.prefix_word_count for t in forward.values()}, allowed)

    def test_unicode_newlines_and_whitespace_reconstruct_exactly(self):
        for variant in ('short', 'long'):
            e = self.build(variant)
            self.addCleanup(e.image.close)
            self.assertEqual(e.prompt_text + e.target_text, self.record[variant+'_desc'])
            self.assertEqual(bytes(e.input_ids).decode('utf-8'), self.record[variant+'_desc'])
            self.assertTrue(e.target_text[0].isspace())
            self.assertTrue(e.target_text.strip())

    def test_source_record_unchanged(self):
        original = copy.deepcopy(self.record)
        e = self.build(); e.image.close()
        self.assertEqual(self.record, original)

    def test_image_loaded_rgb_at_original_size(self):
        e = self.build()
        self.addCleanup(e.image.close)
        self.assertEqual(e.image.mode, 'RGB')
        self.assertEqual(e.image.size, (13,17))

    def test_tokenizer_is_called_separately_without_special_tokens(self):
        calls=[]
        class Spy:
            def encode(self,text,*,add_special_tokens=False):
                calls.append((text,add_special_tokens))
                return list(text.encode())
        e = TaskBuilder(Spy(),self.images).build(self.record,'short')
        e.image.close()
        self.assertEqual(calls,[(e.prompt_text,False),(e.target_text,False)])

    def test_only_target_and_explicit_eos_are_supervised(self):
        e = TaskBuilder(DiagnosticByteTokenizer(),self.images,bos_token_id=256,eos_token_id=257).build(self.record,'short')
        self.addCleanup(e.image.close)
        self.assertEqual(e.tokenized_prompt[0],256)
        self.assertEqual(e.tokenized_target[-1],257)
        self.assertEqual(e.loss_mask,[0]*len(e.tokenized_prompt)+[1]*len(e.tokenized_target))
        # Downstream causal loss shifts labels/mask once: final prompt position predicts first target.
        shifted=e.loss_mask[1:]
        self.assertEqual(shifted[len(e.tokenized_prompt)-1],1)
        self.assertTrue(all(x==0 for x in shifted[:len(e.tokenized_prompt)-1]))
        self.assertEqual(shifted[-1],1)

    def test_padding_is_not_supervised_even_when_pad_equals_eos(self):
        builder = TaskBuilder(DiagnosticByteTokenizer(), self.images, eos_token_id=256)
        a = builder.build(self.record,'short'); b = builder.build(self.record,'long')
        self.addCleanup(a.image.close);self.addCleanup(b.image.close)
        batch=collate([a,b],256)
        n=len(a.input_ids)
        self.assertEqual(batch['input_ids'][0][n-1],256)
        self.assertEqual(batch['loss_mask'][0][n-1],1)
        self.assertTrue(all(v==0 for v in batch['loss_mask'][0][n:]))
        self.assertTrue(all(v==0 for v in batch['attention_mask'][0][n:]))
        self.assertTrue(all(v==1 for v in batch['attention_mask'][0][:n]))

    def test_invalid_mask_rejected(self):
        e=self.build();self.addCleanup(e.image.close)
        with self.assertRaises(ValueError): collate([replace(e,loss_mask=[1]*len(e.input_ids))],256)
        with self.assertRaises(ValueError): collate([],256)

    def test_no_silent_truncation_or_empty_target(self):
        with self.assertRaisesRegex(ValueError,'needs more'):
            self.build(short_desc='one two three')
        with self.assertRaisesRegex(ValueError,'missing'):
            self.build(long_desc='',variant='long')

    def test_invalid_variant_or_config(self):
        with self.assertRaises(ValueError): self.build('medium')
        with self.assertRaises(ValueError): TaskConfig(short_prefix_words=(7,6))
        with self.assertRaises(ValueError): TaskConfig(random_seed=True)
        with self.assertRaises(ValueError): TaskConfig.from_dict({'unknown':42})

    def test_bad_tokenizer_ids_rejected(self):
        class Broken:
            def encode(self,text,*,add_special_tokens=False): return []
        with self.assertRaisesRegex(ValueError,'Tokenizer'):
            TaskBuilder(Broken(),self.images).build(self.record,'short')

    def test_missing_and_escaping_images_fail(self):
        for path in ('missing.jpg','../outside.jpg','/tmp/outside.jpg','images\\wrong.jpg'):
            with self.subTest(path=path),self.assertRaises(ValueError): self.build(image_path=path)
        outside=self.root/'outside.jpg';outside.write_bytes(b'fixture')
        (self.images/'escape.jpg').symlink_to(outside)
        with self.assertRaises(ValueError): self.build(image_path='escape.jpg')

    def test_injected_image_loader(self):
        sentinel=object()
        e=TaskBuilder(DiagnosticByteTokenizer(),self.images,image_loader=lambda path:sentinel).build(self.record,'short')
        self.assertIs(e.image,sentinel)

    def test_punctuation_only_tokens_not_counted_as_words(self):
        r={**self.record,'short_desc':'one — two three four five six seven eight nine ten'}
        text=build_text(r,'short',TaskConfig(short_prefix_words=(6,6)))
        self.assertEqual(text.prompt_text,'one — two three four five six')

    def test_dataset_is_lazy_and_validates_checksum(self):
        directory=self.make_dataset()
        calls=[]
        builder=TaskBuilder(DiagnosticByteTokenizer(),self.images,image_loader=lambda path:calls.append(path))
        dataset=TaskDataset(directory,'train','short',builder)
        self.assertEqual(calls,[])
        self.assertEqual(len(dataset),1)
        dataset[0];self.assertEqual(len(calls),1)
        (directory/'train.jsonl').write_text('modified')
        with self.assertRaisesRegex(ValueError,'checksum'):TaskDataset(directory,'train','short',builder)

    def test_dataset_duplicate_ids_rejected(self):
        directory=self.make_dataset([self.record,self.record])
        with self.assertRaisesRegex(ValueError,'duplicate'):
            TaskDataset(directory,'train','short',self.builder)

    def test_cpu_integration_checks_both_variants_and_all_splits(self):
        report=check(self.make_dataset(),self.images,TaskConfig())
        self.assertTrue(report['ok'])
        self.assertEqual(len(report['counts']),6)
        self.assertEqual(sum(c['passed'] for c in report['counts'].values()),6)
        self.assertIn('not for training',report['tokenizer'])


if __name__=='__main__': unittest.main()
