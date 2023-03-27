import numpy as np
import torch
import torchaudio
from audio_preprocessing import AudioPreprocessor
from audio2mel import AudioToMelSpectrogram
from mel2audio import MelSpectrogramToAudio
import sounddevice as sd
import matplotlib.pyplot as plt

def main():
    # Load the input wavefile
    filename = "audio/test_guns.wav"
    waveform, sample_rate = torchaudio.load(filename)
    
    # # apply the preprocessor
    # preprocessor = AudioPreprocessor()
    # waveform = preprocessor(waveform, sample_rate)
    
    # shorten the audio
    # waveform = waveform[:,:1*sample_rate]
    
    # create the mel-spectrogram
    audio2mel = AudioToMelSpectrogram(sample_rate)
    mel_spec = audio2mel(waveform)


    # plot the mel-spectrogram
    plt.imshow(mel_spec.squeeze().numpy(), cmap='magma')
    plt.xlabel('Time')
    plt.ylabel('Mel-frequency')
    plt.title('Mel-spectrogram')
    plt.show()


    # # get the original audio back
    reconstructed = audio2mel.reverse(mel_spec)
    

    # # Play the preprocessed audio
    torchaudio.save('output/guns.wav', reconstructed, sample_rate)
    # # print(waveform.max(), waveform.min())
    
if __name__ == '__main__':
    main()