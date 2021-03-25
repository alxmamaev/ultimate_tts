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

    def on_loader_end(self, runner):
        item_index = 0

        for i, name in enumerate(["decoder_output", "postnet_output", "alignment"]):
            output = runner.batch[self.input_key][i].detach().cpu()[item_index]
            output = np.transpose(output)

            if name == "alignment":
                output = output[:runner.batch["text_lenghts"][item_index], 
                                :runner.batch["mel_lenghts"][item_index]]

            fig = plt.figure(figsize=(12, 12))
            plt.imshow(output)
            plt.ion()

            image = render_figure_to_array(fig)
            runner.log_image(tag=name, image=image, scope="tacotron_output")
            