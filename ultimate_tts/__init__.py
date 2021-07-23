from .tts import models
from .tts.layers import losses
from . import callbacks
from . import runners
from .utils import text, audio, features, forced_aligner
from .dataset import text_mel_collate_fn
from catalyst.registry import REGISTRY


# -- Register text preprocessing functions --
REGISTRY.add_from_module(text.processor)
REGISTRY.add_from_module(text.cleaners)
REGISTRY.add_from_module(text.g2p)
REGISTRY.add_from_module(text.normalizers)

# -- Register audio preprocessing functions --
REGISTRY.add_from_module(audio.processor)
REGISTRY.add_from_module(audio.transforms)

# -- Register feature extraction functions --
REGISTRY.add_from_module(features.processor)
REGISTRY.add_from_module(features.mel)
REGISTRY.add_from_module(features.xvectors)
REGISTRY.add_from_module(features.prosody)

# -- Register forced alignment extraction functions --
REGISTRY.add_from_module(forced_aligner.processor)

# -- Register data functions --
REGISTRY.add(text_mel_collate_fn)

# -- Register TTS training utils --
REGISTRY.add_from_module(runners)
REGISTRY.add_from_module(callbacks)

# -- Register models and losses --
REGISTRY.add_from_module(models)
REGISTRY.add_from_module(losses)
