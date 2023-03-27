import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import numpy as np

class AudioToMelSpectrogram(torch.nn.Module):
    def __init__(self, sr, n_fft=1024, win_length=1024, hop_length=256, n_mels=120):
        super().__init__()
        # initialize the Mel-Spectrogram
        self.spectrogram = T.Spectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length, power=2)
        self.mel = T.MelScale(n_mels=n_mels, sample_rate=sr, n_stft=n_fft // 2 + 1, f_min=20)
        
        # initialize the inverse Mel-Spectrogram
        self.inverse_mel_scale = T.InverseMelScale(n_mels=n_mels, sample_rate=sr, n_stft=n_fft // 2 + 1, f_min=20)
        self.griffin_lim = T.GriffinLim(n_fft=n_fft, win_length=win_length, hop_length=hop_length, power=2)

    def forward(self, waveform):
        """Convert the waveform into a human readable Mel-Spectrogram

        Args:
            waveform (torch.tensor): Tensor of the waveform

        Returns:
            torch.tensor: Mel-Spectrogram
        """
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Apply mel spectrogram
        
        
        
        spec = self.spectrogram(waveform)
        spec = self.mel(spec)

        # # Convert to decibels
        # spec = F.amplitude_to_DB(spec, 20, 1e-10, np.log10(max(spec.max(), 1e-10)))
        spec = torch.log2(torch.clamp(spec, min=1e-10))
        
        # ATDB = T.AmplitudeToDB(stype="amplitude", top_db=80)
        # spec = ATDB(spec)
        # mel_spec_db = F.amplitude_to_DB(spec, 10, 1e-10, 1)
        
        return spec
    
    def reverse(self, spectrogram):
        """reverse of the Audio to Mel-Spectrogram function. Gets the orginal waveform given a Mel-Spectrogram

        Args:
            spectrogram (torch.tensor): The Spectrogram calculated by AudioToMelSpectrogram

        Returns:
            torch.tensor: The original Waveform
        """
        spectrogram = torch.pow(spectrogram, 2)
        spectrogram = self.inverse_mel_scale(spectrogram)
        waveform = self.griffin_lim(spectrogram)
        return waveform
