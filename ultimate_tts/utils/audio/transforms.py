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