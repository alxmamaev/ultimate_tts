from collections import OrderedDict
from functools import partial

from catalyst import utils
from catalyst.dl import IRunner, SupervisedConfigRunner
from dataset import TextMelDataset, text_mel_collate_fn
from text.tokenizers import CharTokenizer
from text.utils import TextPreprocesser
from text.cleaners import LowerCaseCleaner, VocabCleaner
from catalyst.utils.data import get_loaders_from_params


class TTSRunner(SupervisedConfigRunner):
    def get_datasets(self, stage: str):
        datasets = OrderedDict()

        vocab = self._config["shared"]["vocab"]

        tokenizer = CharTokenizer(vocab)
        cleaners = [LowerCaseCleaner(), VocabCleaner(vocab)]
        text_preprocessor = TextPreprocesser(tokenizer, cleaners=cleaners)

        datasets["train"] = TextMelDataset(self._config["shared"]["train_metadata"],
                                           self._config["shared"]["datapath"],
                                           text_preprocessor)


        datasets["valid"] = TextMelDataset(self._config["shared"]["valid_metadata"],
                                           self._config["shared"]["datapath"],
                                           text_preprocessor)

        return datasets

    def get_loaders(self, stage: str):
        loader_params = dict(self._stage_config[stage]["loaders"])
        loader_params["collate_fn"] = text_mel_collate_fn

        loaders_params = {"valid":loader_params, "train":loader_params}

        loaders = get_loaders_from_params(
            datasets_fn=partial(self.get_datasets, stage=stage),
            initial_seed=self.seed,
            stage=stage,
            loaders_params=loaders_params,
        )
        return loaders