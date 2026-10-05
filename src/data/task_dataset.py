"""Lazy adapter from final DATA files to architecture-independent TASK examples."""
import hashlib
import json
from pathlib import Path

from src.tasks.builder import TaskBuilder


class TaskDataset:
    def __init__(self, dataset_dir: str | Path, split: str, variant: str, builder: TaskBuilder):
        if split not in ('train', 'val', 'test') or variant not in ('short', 'long'):
            raise ValueError('Expected train/val/test split and short/long variant')
        directory = Path(dataset_dir)
        manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
        if manifest['schema_version'] != 1:
            raise ValueError('Unsupported dataset schema')
        filename = split + '.jsonl'
        raw = (directory / filename).read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest['files'][filename]:
            raise ValueError(f'{filename}: checksum mismatch')
        self.records = [json.loads(line) for line in raw.splitlines()]
        if len(self.records) != manifest['counts'][split]:
            raise ValueError(f'{filename}: count mismatch')
        seen = set()
        for record in self.records:
            image_id = record['image_id']
            if type(image_id) is not int or image_id < 0 or image_id in seen or record['split'] != split:
                raise ValueError(f'{filename}: invalid ID, duplicate ID, or incorrect split')
            seen.add(image_id)
        self.variant, self.builder = variant, builder

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index: int):
        if type(index) is not int:
            raise TypeError('TaskDataset indices must be integers')
        return self.builder.build(self.records[index], self.variant)
