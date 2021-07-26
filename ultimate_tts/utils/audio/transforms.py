import torch
import torchaudio as taudio


class AudioTrimmer:
    def __init__(self, sample_rate, trim_front_params=None, trim_back_params=None):
        self.vad_front = None 
        self.vad_back = None

        if trim_front_params is not None:
            self.vad_front = taudio.transforms.Vad(sample_rate=sample_rate, **trim_front_params)
            if torch.cuda.is_available():
                self.vad_front = self.vad_front.cuda()

        if trim_back_params is not None:
            self.vad_front = taudio.transforms.Vad(sample_rate=sample_rate, **trim_back_params)
            if torch.cuda.is_available():
                self.vad_back = self.vad_front.cuda()


    @torch.no_grad()
    def __call__(self, audio_batch):
        processed_batch = []

        for audio in audio_batch:
            if torch.cuda.is_available():
                audio = audio.cuda()

            if self.vad_front is not None:
                audio = self.vad_front(audio)

            if self.vad_back is not None:
                audio = torch.flip(audio, [0])
                audio = self.vad_back(audio)
                audio = torch.flip(audio, [0])
            
            audio = audio.cpu()
            processed_batch.append(audio)

        return processed_batch


class AudioPadForMel:
    def __init__(self, hop_length, win_length):
        self.hop_length = hop_length
        self.win_length = win_length

    @torch.no_grad()
    def __call__(self, audio_batch):
        processed_batch = []

        for audio in audio_batch:
            n_windows = 1
            while (n_windows - 1) * self.hop_length + self.win_length < audio.shape[0]:
                n_windows += 1

            pad_size = (n_windows - 1) * self.hop_length + self.win_length - audio.shape[0]

            audio = torch.cat([audio, torch.zeros(pad_size, dtype=audio.dtype)])
            processed_batch.append(audio)

        return processed_batch

        
class AudioLengthRegulator:
    def __init__(self, sample_rate, max_length, min_length):
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.min_length = min_length

    def __call__(self, audio_batch):
        processed_batch = []

        for audio in audio_batch:
            audio_length = audio.shape[0] / self.sample_rate
            if audio_length < self.min_length and audio_length > self.max_length:
                processed_batch.append(None)
            else:
                processed_batch.append(audio)

        return processed_batch
