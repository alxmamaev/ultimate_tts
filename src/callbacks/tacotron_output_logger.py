import numpy as np
from catalyst.core.callback import Callback, CallbackNode, CallbackOrder
from catalyst.metrics._confusion_matrix import ConfusionMatrixMetric
from matplotlib import pyplot as plt
from catalyst.contrib.utils.visualization import render_figure_to_array


class TacotronOutputLogger(Callback):
    def __init__(
        self,
        input_key: str
    ):
        """Callback initialisation."""
        super().__init__(CallbackOrder.metric, CallbackNode.all)
        self.input_key = input_key

        self.model_output = [None, None, None]
        self.model_output_names = ["decoder_output", "postnet_output", "alignment"]

    def on_batch_end(self, runner):
        for i in range(3):
            output = runner.batch[self.input_key][i].detach().cpu()[0]
            output = np.transpose(output)

            self.model_output[i] = output

    def on_loader_end(self, runner):
        for (name, output) in zip(self.model_output_names, self.model_output):
            fig = plt.figure(figsize=(12, 12))
            plt.imshow(output, interpolation="nearest")
            plt.colorbar()
            plt.ion()

            image = render_figure_to_array(fig)
            runner.log_image(tag=name, image=image, scope="loader")