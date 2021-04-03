from tts.models.fastspeech import FastSpeech
from tts.models.tacotron2 import Tacotron2
import torch

if __name__ == "__main__":
    model = Tacotron2()#FastSpeech()

    dummy_input = torch.randint(0, 15, (10, 5))
    encoder_mask = torch.BoolTensor([[False, False, False, False, False]]*10)
    decoder_mask = torch.BoolTensor([[False] * 16])
    dummy_durations = torch.LongTensor([[3, 1, 1, 1, 10]])
    
    # output_mels, durations, alignments = model(dummy_input, dummy_durations, encoder_mask, decoder_mask) 

    # print(output)

    model.eval()
    output = model.inference(dummy_input, encoder_mask)
    print(output[-1].shape)