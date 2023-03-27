import torch
import torchaudio
import torchaudio.functional as F
import numpy as np

class AudioPreprocessor(torch.nn.Module):
    def __init__(self, eq_chance=1, gain_chance=1):
        super().__init__()
        self.eq_chance = eq_chance
        self.gain_chance = gain_chance
        self.gain = torchaudio.transforms.Vol(1)

    def forward(self, waveform, sr):
        # Apply EQ with the chance specified
        if np.random.random() < self.eq_chance:
            waveform = F.equalizer_biquad(
                waveform,
                sr,
                center_freq=np.random.choice([20, 60, 100, 250, 400, 630, 900, 1200, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000]),
                gain=np.random.uniform(-6,6),
                Q=0.707
            )
        
        if np.random.random() < self.gain_chance:
            gain = np.random.uniform(-6, 6)
            self.gain.gain = gain
            waveform = self.gain(waveform)

        return waveform