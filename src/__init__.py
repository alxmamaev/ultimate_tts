from .tts.models import Tacotron2
from .tts.layers.losses import Tacotron2Loss
from .callbacks import TacotronOutputLogger
from .runner import TTSRunner

from catalyst.registry import Registry


Registry(TTSRunner)
Registry(Tacotron2)
Registry(Tacotron2Loss)
Registry(TacotronOutputLogger)