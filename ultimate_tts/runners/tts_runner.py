from collections import OrderedDict
from functools import partial

from ..dataset import TextMelDataset, text_mel_collate_fn
# from ..utils.text.preprocessor import TextPreprocessor

from catalyst import utils
from catalyst.dl import IRunner, SupervisedConfigRunner
from catalyst.utils.data import get_loaders_from_params
from copy import copy
from catalyst.registry import REGISTRY


class TTSRunner(SupervisedConfigRunner):
    def get_collate_fn(self):
        data_params = self._config["data_params"]
        collate_fn = REGISTRY.get(data_params["collate_fn"])

        return collate_fn

    def get_text_preprocessor(self):
        data_params = self._config["data_params"]

        tokenizer = REGISTRY.get_from_params(**data_params["tokenizer"])
        cleaners = [REGISTRY.get_from_params(**cleaner_factory_params) for cleaner_factory_params in data_params["cleaners"]]

        text_preprocessor = TextPreprocesser(tokenizer, cleaners=cleaners)

        return text_preprocessor

    def get_datasets(self, stage: str):
        datasets = OrderedDict()
        data_params = self._config["data_params"]
        
        text_preprocessor = self.get_text_preprocessor()
        datasets["train"] = TextMelDataset(text_preprocessor, 
                                           data_params["train_metadata"],
                                           data_params["mels_datapath"],
                                           durations_datapath=data_params.get("durations_datapath"))

        datasets["valid"] = TextMelDataset(text_preprocessor,
                                           data_params["valid_metadata"],
                                           data_params["mels_datapath"],
                                           durations_datapath=data_params.get("durations_datapath"))

        return datasets


    def get_loaders(self, stage: str):
        loader_params = dict(self._stage_config[stage]["loaders"])
        loader_params["collate_fn"] = self.get_collate_fn()

        loaders_params = {"valid": copy(loader_params), 
                          "train": copy(loader_params)}

        loaders = get_loaders_from_params(
            datasets_fn=partial(self.get_datasets, stage=stage),
            initial_seed=self.seed,
            stage=stage,
            loaders_params=loaders_params,
        )
        return loaders


__all__ = ["TTSRunner"]