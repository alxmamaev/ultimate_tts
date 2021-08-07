from .tts import models
from .tts.layers import losses
from . import callbacks
from . import runners
from . import processors
from . import transforms
from .data.dataset import text_mel_collate_fn
from catalyst.registry import REGISTRY


# -- Register processors --
for processor_name in processors.__all__:
    processor_module = processors.__dict__[processor_name]
    REGISTRY.add_from_module(processor_module, prefix=f"processors.{processor_name}.")


# -- Register transforms --
for transform_name in transforms.__all__:
    transform_module = transforms.__dict__[transform_name]
    REGISTRY.add_from_module(transform_module, prefix=f"transforms.{transform_name}.")

# -- Register data modules --
REGISTRY.add(text_mel_collate_fn)

# -- Register TTS training modules --
REGISTRY.add_from_module(runners)
REGISTRY.add_from_module(callbacks)

# -- Register models and losses --
REGISTRY.add_from_module(models)
REGISTRY.add_from_module(losses)
