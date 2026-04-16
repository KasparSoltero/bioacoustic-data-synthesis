# spectrogram.py
# STFT, log scaling, PCEN, band filtering, dB normalization, PIL/numpy conversion
import matplotlib.pyplot as plt
import torch
import torchaudio
from torchaudio.functional import spectrogram

class Spectrogram:
    def __init__(self, waveform, sample_rate, n_fft=2048, hop_length=None, win_length=None, window=None, power=2.0, normalised=False):
        self.waveform = waveform
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length if hop_length is not None else n_fft // 4
        self.win_length = win_length if win_length is not None else n_fft
        self.window = window
        self.power = power
        self.normalised = normalised
        
        self.is_logscale = False
        self._values = None # Lazy evaluation: computed only when requested

    @property
    def values(self):
        """Computes the spectrogram lazily. Only runs STFT if necessary."""
        if self._values is None:
            self._values = spectrogram(
                self.waveform,
                pad=0,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                power=self.power,
                normalized=self.normalised
            )
            if self.is_logscale:
                self._apply_logscale()
        return self._values

    @property
    def is_complex(self):
        # Determine theoretically if not computed, or practically if computed
        if self._values is not None:
            return self._values.is_complex()
        return self.power is None

    def to_complex(self):
        """Forces complex representation. Recomputes only if currently holding real values."""
        if not self.is_complex:
            self.power = None
            self._values = None # Invalidate cache to force STFT recomputation
        return self

    def to_real(self, power=2.0):
        """Converts to real. Derives mathematically if already complex to save STFT overhead."""
        if self.power == power:
            return self
            
        if self.is_complex and self._values is not None:
            # We already have complex values; compute magnitude/power directly
            if power == 1.0:
                self._values = self._values.abs()
            elif power == 2.0:
                # abs().pow(2) is mathematically identical to power=2.0
                self._values = self._values.abs().pow(2)
            else:
                self._values = None # Fallback for edge-case power values
        else:
            # We don't have complex values, so we must recompute STFT
            self._values = None 

        self.power = power
        return self

    def to_logscale(self):
        """Applies log scaling in-place if computed."""
        if not self.is_logscale:
            self.is_logscale = True
            if self._values is not None:
                self._apply_logscale()
        return self
        
    def _apply_logscale(self):
        original_height = self._values.shape[-2]
        log_scale = torch.logspace(0, 1, steps=original_height, base=10.0, device=self._values.device) - 1
        log_scale_indices = torch.clamp(log_scale * (original_height - 1) / (10 - 1), 0, original_height - 1).long()
        self._values = self._values[..., log_scale_indices, :]

    def to_linear(self):
        """Reverts to linear scale. Drops the tensor to save memory; recomputes lazily next access."""
        if self.is_logscale:
            self.is_logscale = False
            self._values = None # Dropping cache is much safer for ML memory limits
        return self

    def to_melspec(self, n_mels=128):
        mel_spec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window_fn=self.window,
            n_mels=n_mels,
            power=self.power,
            normalized=self.normalised
        ).to(self.waveform.device) # Ensure transform lives on the same device
        
        self._values = mel_spec_transform(self.values)
        return self._values

if __name__ == "__main__":
    random_waveform = torch.randn(1, 48000 * 5)
    sample_rate = 48000

    test_waveform_path = '/Users/kaspar/Downloads/09/20240917_034600_from_20240917_034600.WAV'
    test_waveform, sample_rate = torchaudio.load(test_waveform_path)
    
    # Init lazily (no computation done yet)
    spec = Spectrogram(test_waveform, sample_rate, power=None) # Start complex
    
    print(f'is complex (before compute): {spec.is_complex}')
    
    # Computation triggers here
    print(f'mean (complex abs): {spec.values.abs().mean()}')
    
    # Mathematically derives real without running STFT again!
    spec.to_real(power=2.0)
    print(f'is complex after to_real: {spec.is_complex}')
    print(f'mean (real power=2): {spec.values.mean()}')

    # spec.to_logscale() 
    # print(f'mean (logscale): {spec.values.mean()}')

    figure, ax = plt.subplots(1, 1, figsize=(7, 7))
    plot_values = spec.values[0].abs().numpy() if spec.is_complex else spec.values[0].numpy()
    ax.imshow(plot_values, aspect='auto', origin='lower', vmin=0, vmax=1) 
    ax.set_xlabel('Time Frames')
    ax.set_ylabel('Frequency Bins')
    plt.show()