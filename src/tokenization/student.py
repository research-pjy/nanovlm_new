"""Frozen train-only student BPE; no network or teacher dependencies."""
import hashlib
import importlib.metadata
import json
from pathlib import Path

SPECIALS = ['<pad>', '<bos>', '<eos>']
SETTINGS = {'algorithm': 'byte_level_bpe', 'target_vocabulary_size': 4096,
            'min_frequency': 2, 'random_seed': 42, 'add_prefix_space': False,
            'normalization': None, 'bpe_dropout': None,
            'special_tokens': SPECIALS, 'fields': ['short_desc', 'long_desc'],
            'order': 'image_id_ascending_then_short_long'}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()


def validate_text(text):
    if not isinstance(text, str):
        raise ValueError('Expected text string')
    if any(token in text for token in SPECIALS):
        raise ValueError('Text contains a reserved special-token spelling')
    text.encode('utf-8')  # Reject unpaired surrogate characters.


def training_corpus(directory):
    """Read only manifest + train.jsonl; validation/test never enter fitting."""
    directory = Path(directory)
    manifest_raw = (directory / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_raw)
    raw = (directory / 'train.jsonl').read_bytes()
    if manifest['schema_version'] != 1 or digest(raw) != manifest['files']['train.jsonl']:
        raise ValueError('Training dataset schema/checksum mismatch')
    records = [json.loads(line) for line in raw.splitlines()]
    if not records or len(records) != manifest['counts']['train']:
        raise ValueError('Training record count mismatch or empty corpus')
    seen = set()
    for record in records:
        identity = record['image_id']
        if type(identity) is not int or identity < 0 or identity in seen or record['split'] != 'train':
            raise ValueError('Invalid training ID, duplicate, or non-train record')
        seen.add(identity)
        for field in SETTINGS['fields']:
            validate_text(record[field])
            if not record[field].strip():
                raise ValueError(f'{identity}: empty {field}')
    texts = [r[field] for r in sorted(records, key=lambda r: r['image_id'])
             for field in SETTINGS['fields']]
    provenance = {'dataset_manifest_sha256': digest(manifest_raw),
                  'train_file_sha256': digest(raw), 'corpus_sha256': digest(json_bytes(texts)),
                  'train_images': len(records), 'training_texts': len(texts),
                  'train_ids_sha256': digest(json_bytes(sorted(seen)))}
    return texts, provenance


def fit(texts):
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    tokenizer = Tokenizer(models.BPE(unk_token=None, dropout=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=4096, min_frequency=2,
                                  special_tokens=SPECIALS, show_progress=True,
                                  initial_alphabet=sorted(pre_tokenizers.ByteLevel.alphabet()))
    tokenizer.train_from_iterator(texts, trainer=trainer, length=len(texts))
    return tokenizer


class StudentTokenizer:
    pad_token_id, bos_token_id, eos_token_id = 0, 1, 2

    def __init__(self, directory):
        from tokenizers import Tokenizer, pre_tokenizers
        directory = Path(directory)
        self.manifest = json.loads((directory / 'manifest.json').read_bytes())
        raw = (directory / 'tokenizer.json').read_bytes()
        if (self.manifest.get('schema_version') != 1 or self.manifest.get('settings') != SETTINGS
                or self.manifest.get('special_token_ids') != {'pad': 0, 'bos': 1, 'eos': 2}):
            raise ValueError('Unsupported student tokenizer manifest/settings')
        if digest(raw) != self.manifest['tokenizer_sha256']:
            raise ValueError('Tokenizer checksum mismatch')
        installed = importlib.metadata.version('tokenizers')
        if self.manifest['tokenizers_version'] != installed:
            raise ValueError('Tokenizer library version differs from artifact; use its recorded version')
        self._tokenizer = Tokenizer.from_str(raw.decode())
        config = json.loads(raw)
        if (config['model']['type'] != 'BPE' or config['model'].get('dropout') is not None
                or config['model'].get('unk_token') is not None
                or config.get('normalizer') is not None or config.get('post_processor') is not None
                or config.get('padding') is not None or config.get('truncation') is not None
                or config['pre_tokenizer']['type'] != 'ByteLevel'
                or config['pre_tokenizer']['add_prefix_space'] is not False
                or config['decoder']['type'] != 'ByteLevel'):
            raise ValueError('Tokenizer operations disagree with approved settings')
        vocab = self._tokenizer.get_vocab()
        self.vocabulary_size = len(vocab)
        if (self.vocabulary_size != self.manifest['vocabulary_size']
                or set(vocab.values()) != set(range(self.vocabulary_size))
                or any(vocab.get(t) != i for i, t in enumerate(SPECIALS))
                or any(c not in vocab for c in pre_tokenizers.ByteLevel.alphabet())):
            raise ValueError('Invalid vocabulary, byte coverage or special IDs')
        self.artifact_sha256 = digest(raw)

    def encode(self, text, *, add_special_tokens=False):
        if add_special_tokens:
            raise ValueError('TASK must insert BOS/EOS explicitly')
        validate_text(text)
        ids = self._tokenizer.encode(text, add_special_tokens=False).ids
        if any(i < 3 or i >= self.vocabulary_size for i in ids):
            raise ValueError('Ordinary text produced a reserved/out-of-range token ID')
        return ids

    def decode(self, ids, *, skip_special_tokens=True):
        if any(type(i) is not int or not 0 <= i < self.vocabulary_size for i in ids):
            raise ValueError('Invalid token ID')
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def resolve_model_config(self, config):
        from dataclasses import replace
        if config.vocabulary_size not in (None, self.vocabulary_size):
            raise ValueError('Model vocabulary size differs from saved tokenizer')
        return replace(config, vocabulary_size=self.vocabulary_size)


def train(dataset_dir, output_dir):
    from src.data.artifacts import publish
    texts, provenance = training_corpus(dataset_dir)
    output = Path(output_dir)
    if output.exists():
        saved = StudentTokenizer(output)  # Incomplete artifacts fail, never refit silently.
        if saved.manifest['provenance'] != provenance:
            raise ValueError('Existing tokenizer belongs to different training data')
        return saved, 'unchanged'
    tokenizer = fit(texts)
    # Check exact recovery before publishing any artifact.
    for text in texts:
        if tokenizer.decode(tokenizer.encode(text).ids) != text:
            raise ValueError('Training text failed exact tokenizer round-trip')
    raw = tokenizer.to_str(pretty=True).encode()
    manifest = {'schema_version': 1, 'identity': 'nanovlm-student-byte-bpe-v1',
                'settings': SETTINGS, 'provenance': provenance,
                'tokenizers_version': importlib.metadata.version('tokenizers'),
                'tokenizer_sha256': digest(raw), 'vocabulary_size': tokenizer.get_vocab_size(),
                'special_token_ids': {'pad': 0, 'bos': 1, 'eos': 2}}
    output.mkdir(parents=True, exist_ok=False)
    publish(output / 'tokenizer.json', raw)
    publish(output / 'manifest.json', json_bytes(manifest))  # completion marker last
    return StudentTokenizer(output), 'created'
