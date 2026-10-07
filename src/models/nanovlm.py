"""Complete small NanoVLM; teacher-free, text-aligned forward contract."""
import copy
import hashlib
import io
import json
import torch
from torch import nn

from src.models.config import ModelConfig
from src.models.vision.encoder import VisionEncoder
from src.models.connector.projector import VisualTextualConnector
from src.models.decoder.language import CausalLanguageDecoder
from src.preprocessing.images import ImageConfig


def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True))


def artifact_identity(config, tokenizer, preprocessing):
    if config.vocabulary_size != tokenizer.vocabulary_size:
        raise ValueError('Model vocabulary must match the resolved student tokenizer')
    if (tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id) != (0, 1, 2):
        raise ValueError('Expected approved PAD/BOS/EOS IDs 0/1/2')
    if config.image_size != preprocessing.image_size or config.in_channels != 3:
        raise ValueError('Model and RGB preprocessing dimensions disagree')
    import PIL
    import importlib.metadata
    pixel_config = canonical(preprocessing.to_dict())
    return {'tokenizer_sha256': tokenizer.artifact_sha256,
            'tokenizer_manifest': canonical(tokenizer.manifest),
            'preprocessing': pixel_config,
            'preprocessing_sha256': hashlib.sha256(json.dumps(pixel_config, sort_keys=True).encode()).hexdigest(),
            'pillow_version': PIL.__version__,
            'tokenizers_version': importlib.metadata.version('tokenizers')}


class NanoVLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.vocabulary_size is None:
            raise ValueError('Resolve vocabulary with the student tokenizer first')
        self.config = config
        self.vision = VisionEncoder(config.to_vision_config())
        self.connector = VisualTextualConnector(config.image_embedding_dimension, config.embedding_dimension)
        self.decoder = CausalLanguageDecoder(config)
        self._artifact_identity = None

    @classmethod
    def from_artifacts(cls, config, tokenizer, preprocessing):
        identity = artifact_identity(config, tokenizer, preprocessing)
        model = cls(config)
        model._artifact_identity = identity
        return model

    def forward(self, images, input_ids, attention_mask):
        if (not isinstance(images, torch.Tensor) or not isinstance(input_ids, torch.Tensor)
                or images.ndim != 4 or input_ids.ndim != 2
                or images.shape[0] != input_ids.shape[0]):
            raise ValueError('Images [B,C,H,W] and text IDs [B,T] require matching batches')
        if attention_mask is None:
            raise ValueError('Supply the TASK text attention mask explicitly')
        visual_prefix = self.connector(self.vision(images))
        return self.decoder(input_ids, visual_prefix, attention_mask)

    def parameter_counts(self):
        def count(module):
            return {'total': sum(p.numel() for p in module.parameters()),
                    'trainable': sum(p.numel() for p in module.parameters() if p.requires_grad)}
        return {**{name: count(getattr(self, name)) for name in ('vision', 'connector', 'decoder')},
                'model': count(self)}

    def save_checkpoint(self, path, *, provenance):
        """Model-only artifact; training/resume state belongs to Phase 5."""
        if self._artifact_identity is None:
            raise ValueError('Bind real tokenizer/preprocessing artifacts before checkpoint export')
        from src.data.artifacts import publish
        state = {name: value.detach().cpu() for name, value in self.state_dict().items()}
        payload = {'schema_version': 1, 'model_config': self.config.to_dict(),
                   'identity': copy.deepcopy(self._artifact_identity),
                   'provenance': canonical(provenance), 'state_dict': state,
                   'parameter_counts': self.parameter_counts()}
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        return publish(path, buffer.getvalue())

    @classmethod
    def load_checkpoint(cls, path, *, tokenizer, preprocessing, expected_config):
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if not isinstance(payload, dict) or payload.get('schema_version') != 1:
            raise ValueError('Unsupported model checkpoint schema')
        config = ModelConfig.from_dict(payload['model_config'])
        if config != expected_config:
            raise ValueError('Checkpoint configuration differs, including possible convolution strategy mismatch')
        identity = artifact_identity(config, tokenizer, preprocessing)
        if payload['identity'] != identity:
            raise ValueError('Checkpoint tokenizer/preprocessing identity or runtime version mismatch')
        model = cls.from_artifacts(config, tokenizer, preprocessing)
        model.load_state_dict(payload['state_dict'], strict=True)
        if model.parameter_counts() != payload['parameter_counts']:
            raise ValueError('Checkpoint parameter counts mismatch')
        return model.eval()
