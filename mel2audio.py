import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import numpy as np

class MelSpectrogramToAudio(torch.nn.Module):
    def __init__(self, sr, n_fft=1024, win_length=1024, hop_length=320, n_mels=128):
        super().__init__()
        self.inverse_mel_scale = T.InverseMelScale(n_mels=128, sample_rate=sr, n_stft=n_fft // 2 + 1)
        self.griffin_lim = T.GriffinLim(n_fft=1024, win_length=800, hop_length=320, power=2)
        # self.mel_to_stft = T.MelToSTFT(n_fft=n_fft, n_mels=n_mels, sample_rate=sr, f_min=0, f_max=sr//2, htk=False, norm='slaney')

    def forward(self, mel_spec_db):
        # Convert back from decibels
        # mel_spec = F.DB_to_amplitude(mel_spec_db, 10, 1e-10)
        
        # Inverse mel scale
        spec = self.inverse_mel_scale(mel_spec_db)
        
        # Mel to STFT
        # stft = self.mel_to_stft(mel_spec)

        # Apply Griffin-Lim algorithm to recover waveform
        waveform = self.griffin_lim(spec)

        return waveform