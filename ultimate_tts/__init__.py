from .tts import models
from .tts.layers import losses
from . import callbacks
from . import runners
from .utils.text import cleaners, tokenizers

from catalyst.registry import REGISTRY

# -- Register text preprocessing functions --
REGISTRY.add_from_module(cleaners)
REGISTRY.add_from_module(tokenizers)

# -- Register TTS training utils --
REGISTRY.add_from_module(runners)
REGISTRY.add_from_module(callbacks)

# -- Register models and losses --
REGISTRY.add_from_module(models)
REGISTRY.add_from_module(losses)