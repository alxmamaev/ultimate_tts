import torch
from torch.nn.utils.rnn import pad_sequence
import torchaudio as taudio
from ...utils.functional import normalize_mel


class MelExtractor:
    """Mel Extractor generate mel spectrograms for given audios"""

    def __init__(
        self,
        sample_rate,
        wav_max_value,
        n_fft,
        n_mels,
        win_length,
        hop_length,
        f_min,
        f_max,
        power,
    ):
        self.mel_extract = taudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            power=power,
        )

        if torch.cuda.is_available():
            self.mel_extract = self.mel_extract.cuda()
        self.__hop_length = hop_length
        self.wav_max_value = wav_max_value

    @torch.no_grad()
    def __call__(self, corpus, audios_batch):
        mels_lenghts = [
            audio.shape[0] // self.__hop_length + 1 for audio in audios_batch
        ]

        padded_audios = pad_sequence(audios_batch, batch_first=True)

        if torch.cuda.is_available():
            padded_audios = padded_audios.cuda()

        padded_mels = self.mel_extract(padded_audios.float() / self.wav_max_value)
        padded_mels = normalize_mel(padded_mels).cpu()

        mels = [
            padded_mels[i, :, : mels_lenghts[i]].transpose(0, 1)
            for i in range(padded_mels.shape[0])
        ]

        output_features = {
            "mels": dict([(corpus[i].filename, mels[i]) for i in range(len(corpus))])
        }

        return output_features
