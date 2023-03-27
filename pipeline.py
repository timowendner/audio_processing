import numpy as np
import torch
import torchaudio
from audio_preprocessing import AudioPreprocessor
from audio2mel import AudioToMelSpectrogram
from mel2audio import MelSpectrogramToAudio
import matplotlib.pyplot as plt

def main():
    # Load the input wavefile
    filename = "audio/test_guns.wav"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    waveform, sample_rate = torchaudio.load(filename)
    
    # # apply the preprocessor
    # preprocessor = AudioPreprocessor()
    # waveform = preprocessor(waveform, sample_rate)
    
    # shorten the audio
    # waveform = waveform[:,:1*sample_rate]
    
    # create the mel-spectrogram
    waveform.to(device)
    audio2mel = AudioToMelSpectrogram(sample_rate, device)
    audio2mel.to(device)
    # mel_spec = audio2mel(waveform, device)

    transform = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=256, power=2).to(device)
    mel_spec = transform(waveform)


    # plot the mel-spectrogram
    plt.imshow(mel_spec.squeeze().numpy(), cmap='magma')
    plt.xlabel('Time')
    plt.ylabel('Mel-frequency')
    plt.title('Mel-spectrogram')
    plt.show()


    # # get the original audio back
    # reconstructed = audio2mel.reverse(mel_spec)
    

    # # Play the preprocessed audio
    # torchaudio.save('output/guns.wav', reconstructed, sample_rate)
    # # print(waveform.max(), waveform.min())
    
if __name__ == '__main__':
    main()