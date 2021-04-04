from .runners import TTSRunner
import torch
import yaml
from .utils.audio import denormalize_mel
import torchaudio as taudio


class TTSInference:
    def __init__(self, config_path, checkpoint_path, device="cpu"):
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.Loader)

        if isinstance(device, str):
            device = torch.device(device)

        self.config = config
        self.runner = TTSRunner(config)
        self.model = self.runner.get_model("infer").to(device)
        self.model.eval()
        self.text_preprocessor = self.runner.get_text_preprocessor()
        self.collate_fn = self.runner.get_collate_fn()
        self.output_keys = config["inference"]["output_key"]
        self.mel_key = config["inference"]["mel_key"]

        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint["model_state_dict"]

        self.model.load_state_dict(state_dict)

    def __call__(self, texts):
        batch = []

        for text in texts:
            text = self.text_preprocessor(text)
            text = torch.LongTensor(text)
            batch.append({"text": text})

        batch = self.collate_fn(batch)
        model_output = self.model.inference(**batch)
        model_output = dict(zip(self.output_keys, model_output))
        model_output[self.mel_key] = denormalize_mel(model_output[self.mel_key])

        return model_output


class GriffinlimVocoderInference:
    def __init__(self, config_path):
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.Loader)

        self.config = config

        feature_extractor_params = config["feature_extractor_params"]
        self.invert_mel = taudio.transforms.InverseMelScale(
                                                            n_stft=feature_extractor_params["n_fft"] // 2 + 1,
                                                            n_mels=feature_extractor_params["n_mels"],
                                                            sample_rate=feature_extractor_params["sample_rate"],
                                                            f_min=feature_extractor_params["f_min"],
                                                            f_max=feature_extractor_params["f_max"]
                                                            )
        
        self.griffin_lim = taudio.transforms.GriffinLim(
                                                        n_fft=feature_extractor_params["n_fft"],
                                                        win_length=feature_extractor_params["win_length"],
                                                        hop_length=feature_extractor_params["hop_length"],
                                                        power=feature_extractor_params["power"]
                                                        )

    def __call__(self, mels):
        inverted_mels = self.invert_mel(mels.transpose(1, 2))
        wave_forms = self.griffin_lim(inverted_mels)

        return wave_forms


__all__ = ["TTSInference", "GriffinlimVocoderInference"]