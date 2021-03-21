from tacotron2.model import Tacotron2
from tacotron2.loss import Tacotron2Loss
from dataset import TextMelDataset
from catalyst.registry import Registry, REGISTRY
from runner import TTSRunner
from callbacks.tacotron_output_logger import TacotronOutputLogger


Registry(TTSRunner)
Registry(Tacotron2)
Registry(Tacotron2Loss)
Registry(TacotronOutputLogger)