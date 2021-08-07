import numpy as np
from catalyst.core.callback import Callback, CallbackNode, CallbackOrder
from catalyst.metrics._confusion_matrix import ConfusionMatrixMetric
from matplotlib import pyplot as plt
from catalyst.contrib.utils.visualization import render_figure_to_array
from typing import List


class TTSOutputsLogger(Callback):
    def __init__(self, outputs_keys: List[str]):
        super().__init__(CallbackOrder.metric, CallbackNode.all)
        self.outputs_keys = outputs_keys

    def on_loader_end(self, runner):
        item_index = 0

        for name in self.outputs_keys:
            output = runner.batch[name].detach().cpu()[item_index]
            output = np.transpose(output)

            fig = plt.figure(figsize=(12, 6))
            plt.imshow(output)
            plt.ion()

            image = render_figure_to_array(fig)
            runner.log_image(tag=name, image=image, scope="tacotron_output")


__all__ = ["TTSOutputsLogger"]
