"""Architecture-independent image + text-prefix continuation tasks."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Callable, Protocol


class Tokenizer(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        """Return text token IDs without automatic BOS/EOS or truncation."""
        ...


@dataclass(frozen=True)
class TaskConfig:
    random_seed: int = 42
    short_prefix_words: tuple[int, int] = (6, 7)
    long_prefix_words: tuple[int, int] = (18, 20)

    def __post_init__(self):
        if type(self.random_seed) is not int or self.random_seed < 0:
            raise ValueError('random_seed must be a nonnegative integer')
        for name in ('short_prefix_words', 'long_prefix_words'):
            bounds = getattr(self, name)
            if len(bounds) != 2 or any(type(x) is not int for x in bounds) or not 1 <= bounds[0] <= bounds[1]:
                raise ValueError(f'{name} requires two increasing positive integer bounds')

    @classmethod
    def from_dict(cls, config):
        if set(config) != {'random_seed', 'prefix_words'} or set(config['prefix_words']) != {'short', 'long'}:
            raise ValueError('Task config requires random_seed and short/long prefix_words')
        return cls(config['random_seed'], tuple(config['prefix_words']['short']), tuple(config['prefix_words']['long']))


@dataclass(frozen=True)
class TaskText:
    image_id: int
    image_path: str
    split: str
    variant: str
    prompt_text: str
    target_text: str
    prefix_word_count: int


@dataclass(frozen=True)
class TaskExample:
    image: Any
    image_id: int
    image_path: str
    split: str
    variant: str
    prompt_text: str
    target_text: str
    tokenized_prompt: list[int]
    tokenized_target: list[int]
    loss_mask: list[int]
    prefix_word_count: int

    @property
    def input_ids(self) -> list[int]:
        return self.tokenized_prompt + self.tokenized_target


def build_text(record: dict, variant: str, config: TaskConfig = TaskConfig()) -> TaskText:
    """Choose a stable per-image prefix and preserve exact description characters."""
    if variant not in ('short', 'long'):
        raise ValueError('variant must be short or long')
    image_id = record['image_id']
    if type(image_id) is not int or image_id < 0:
        raise ValueError('image_id must be a nonnegative integer')
    split = record['split']
    if split not in ('train', 'val', 'test'):
        raise ValueError(f'{image_id}: invalid split')
    description = record[variant + '_desc']
    if not isinstance(description, str) or not description.strip():
        raise ValueError(f'{image_id}: missing {variant} description')
    # Same word convention as DATA: whitespace tokens containing a word character.
    words = [match for match in re.finditer(r'\S+', description) if re.search(r'\w', match.group())]
    low, high = getattr(config, variant + '_prefix_words')
    key = json.dumps([config.random_seed, image_id, variant], separators=(',', ':')).encode()
    prefix_length = low + int.from_bytes(hashlib.sha256(key).digest(), 'big') % (high - low + 1)
    if len(words) <= prefix_length:
        raise ValueError(f'{image_id}: {variant} description has {len(words)} words; '
                         f'needs more than chosen prefix length {prefix_length}')
    boundary = words[prefix_length - 1].end()
    return TaskText(image_id, record['image_path'], split, variant,
                    description[:boundary], description[boundary:], prefix_length)


def resolve_image(data_root: Path, image_path: str) -> Path:
    if (not isinstance(image_path, str) or not image_path or
            PurePosixPath(image_path).is_absolute() or '..' in PurePosixPath(image_path).parts or
            '\\' in image_path):
        raise ValueError('image_path must be a safe relative POSIX path')
    root = Path(data_root).expanduser().resolve()
    path = (root / image_path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f'Missing image or image path outside data_root: {image_path}')
    return path


def load_rgb(path: Path):
    """Load original-resolution RGB pixels; resizing/patches belong elsewhere."""
    from PIL import Image
    with Image.open(path) as image:
        image.load()
        return image.convert('RGB')


def token_ids(tokenizer: Tokenizer, text: str) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not isinstance(ids, (list, tuple)) or not ids or any(type(i) is not int or i < 0 for i in ids):
        raise ValueError('Tokenizer must return a nonempty sequence of nonnegative integer IDs')
    return list(ids)


class TaskBuilder:
    def __init__(self, tokenizer: Tokenizer, data_root: str | Path,
                 config: TaskConfig = TaskConfig(), *, bos_token_id: int | None = None,
                 eos_token_id: int | None = None, image_loader: Callable = load_rgb):
        self.tokenizer, self.data_root, self.config = tokenizer, Path(data_root), config
        self.bos_token_id, self.eos_token_id = bos_token_id, eos_token_id
        self.image_loader = image_loader
        for value in (bos_token_id, eos_token_id):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError('Special token IDs must be nonnegative integers or None')

    def build(self, record: dict, variant: str) -> TaskExample:
        text = build_text(record, variant, self.config)
        # Encode separately so prefix tokenization never sees the continuation.
        prompt_ids = token_ids(self.tokenizer, text.prompt_text)
        target_ids = token_ids(self.tokenizer, text.target_text)
        if self.bos_token_id is not None:
            prompt_ids = [self.bos_token_id] + prompt_ids
        if self.eos_token_id is not None:
            target_ids = target_ids + [self.eos_token_id]
        image = self.image_loader(resolve_image(self.data_root, text.image_path))
        return TaskExample(image, text.image_id, text.image_path, text.split, variant,
                           text.prompt_text, text.target_text, prompt_ids, target_ids,
                           [0] * len(prompt_ids) + [1] * len(target_ids), text.prefix_word_count)


def collate(examples: list[TaskExample], pad_token_id: int) -> dict:
    """Right-pad text lists only. No tensor framework or image architecture required."""
    if not examples:
        raise ValueError('Cannot collate an empty batch')
    if type(pad_token_id) is not int or pad_token_id < 0:
        raise ValueError('pad_token_id must be a nonnegative integer')
    width = max(len(e.input_ids) for e in examples)
    batch = {'images': [e.image for e in examples], 'input_ids': [], 'loss_mask': [], 'attention_mask': []}
    for example in examples:
        ids = example.input_ids
        if not example.tokenized_prompt or not example.tokenized_target:
            raise ValueError('Prompt and target token sequences must both be nonempty')
        expected_mask = [0]*len(example.tokenized_prompt) + [1]*len(example.tokenized_target)
        if example.loss_mask != expected_mask:
            raise ValueError('Example loss mask is not aligned with prompt and target')
        padding = width - len(ids)
        batch['input_ids'].append(ids + [pad_token_id]*padding)
        batch['loss_mask'].append(example.loss_mask + [0]*padding)
        batch['attention_mask'].append([1]*len(ids) + [0]*padding)
    return batch
