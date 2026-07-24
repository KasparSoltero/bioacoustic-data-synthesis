# Waveform produces Spectrogram; Spectrogram can reconstruct a Waveform.

import torch
import torchaudio
import random
import numpy as np

class Waveform:
    """Encapsulates a 1D/2D PyTorch tensor representing audio, providing core synthesis transformations."""
    
    def __init__(self, tensor: torch.Tensor, sample_rate: int):
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        self.tensor = tensor
        self.sample_rate = sample_rate
        self.original_nyquist: int = sample_rate // 2  # overwritten by load() when resampled up

    @classmethod
    def load(cls, path: str, target_sr: int = None):
        tensor, sr = torchaudio.load(str(path))
        
        if tensor.shape[0] > 1:
            print('    waveform: converting to mono')
            tensor = tensor.mean(dim=0, keepdim=True)  # Convert to Mono
        
        # Remove DC offset early to prevent thumps during fading/editing
        tensor = tensor - tensor.mean(dim=-1, keepdim=True)

        original_nyquist = sr // 2

        if target_sr and sr != target_sr:
            print(f'    waveform: resample: {sr} -> {target_sr}')
            tensor = torchaudio.transforms.Resample(sr, target_sr, dtype=torch.float32).to(tensor.device)(tensor)
            sr = target_sr

            if target_sr > original_nyquist * 2:
                print(f'    waveform: anti-alias smooth LP at {original_nyquist} Hz')
                n = tensor.shape[-1]
                fft_data = torch.fft.rfft(tensor, dim=-1)
                freqs = torch.fft.rfftfreq(n, d=1.0 / sr).to(tensor.device)
                
                fade_hz = min(1000, original_nyquist // 4)
                fade_start_hz = original_nyquist - fade_hz
                
                mask = torch.ones_like(freqs)
                mask[freqs > original_nyquist] = 0.0
                
                fade_mask = (freqs > fade_start_hz) & (freqs <= original_nyquist)
                fade_bins = fade_mask.sum().item()
                if fade_bins > 0:
                    t = torch.linspace(torch.pi, 0, steps=fade_bins, device=tensor.device)
                    mask[fade_mask] = 0.5 * (1.0 - torch.cos(t))
                    
                fft_data = fft_data * mask
                tensor = torch.fft.irfft(fft_data, n=n, dim=-1)

        instance = cls(tensor, sr)
        instance.original_nyquist = original_nyquist
        return instance

    def crop(self, seconds: float, random_start: bool = True):
        """
        Crops the waveform to exactly `seconds` length.
        Modifies the tensor in-place and returns self for chaining.
        Raises ValueError if the waveform is shorter than the requested length.
        """
        target_samples = int(seconds * self.sample_rate)
        current_samples = self.tensor.shape[1]
        
        if current_samples < target_samples:
            raise ValueError(
                f"Waveform too short for crop. Required {target_samples} samples "
                f"({seconds}s), but only has {current_samples}."
            )
        elif current_samples == target_samples:
            return self
            
        if random_start:
            start = random.randint(0, current_samples - target_samples)
        else:
            start = 0
            
        self.tensor = self.tensor[:, start:start + target_samples]
        return self
    
    def fade(self, fade_in_len: int, fade_out_len: int = None):
        """
        Applies a linear fade in and fade out.
        Modifies the tensor in-place and returns self for chaining.
        """
        if fade_out_len is None:
            fade_out_len = fade_in_len
            
        length = self.tensor.shape[1]
        fade_in_len = min(fade_in_len, length // 2)
        fade_out_len = min(fade_out_len, length - fade_in_len)
        
        if fade_in_len > 0:
            fade_in_curve = torch.linspace(0, 1, steps=fade_in_len, device=self.tensor.device)
            self.tensor[:, :fade_in_len] *= fade_in_curve
        
        if fade_out_len > 0:
            fade_out_curve = torch.linspace(1, 0, steps=fade_out_len, device=self.tensor.device)
            self.tensor[:, -fade_out_len:] *= fade_out_curve
            
        return self

    def sine_fade(self, fade_in_len: int, fade_out_len: int = None):
        """
        Applies a fixed-width sinusoidal (raised cosine) fade. 
        Unlike standard fade, this does not cap the fade length to the waveform length.
        Very short transients will remain at low amplitude and become virtually inaudible.
        """
        if fade_out_len is None:
            fade_out_len = fade_in_len
            
        length = self.tensor.shape[1]
        
        if fade_in_len > 0:
            t_in = torch.linspace(0, torch.pi, steps=fade_in_len, device=self.tensor.device)
            curve_in = 0.5 * (1.0 - torch.cos(t_in))
            apply_in = min(length, fade_in_len)
            self.tensor[:, :apply_in] *= curve_in[:apply_in]
            
        if fade_out_len > 0:
            t_out = torch.linspace(torch.pi, 0, steps=fade_out_len, device=self.tensor.device)
            curve_out = 0.5 * (1.0 - torch.cos(t_out))
            apply_out = min(length, fade_out_len)
            self.tensor[:, -apply_out:] *= curve_out[-apply_out:]
            
        return self

    def trim_to_mix(self, start_offset: int, mix_samples: int) -> int:
        """
        Trims the waveform so it strictly fits within the boundaries of a mix track 
        of length `mix_samples` given a `start_offset` (which can be negative).
        Returns the new corrected start sample (which will be >= 0).
        """
        overlay_samples = self.tensor.shape[1]
        
        # Left bound trimming (if start_offset is negative, chop off the beginning)
        trim_start = max(0, -start_offset)
        new_start = max(0, start_offset)
        
        # Right bound trimming (if it extends past the mix, chop off the end)
        available_space = mix_samples - new_start
        trim_end = min(overlay_samples, trim_start + available_space)
        
        self.tensor = self.tensor[:, trim_start:trim_end]
        
        return new_start

    def _get_rms(self) -> float:
        return torch.sqrt(torch.mean(self.tensor ** 2)).item()

    def set_db(self, target_db: float):
        """
        Scales the waveform to a target dBFS level (amplitude RMS).
        Modifies the tensor in-place and returns self for chaining.
        """
        current_rms = self._get_rms()
        if current_rms == 0:
            return self
            
        target_rms = 10 ** (target_db / 20.0)
        self.tensor = self.tensor * (target_rms / current_rms)
        return self

    def add_noise(self, noise_type: str, target_rms: float):
        """
        Injects white, pink, or brown noise into the waveform at a specific RMS.
        Modifies the tensor in-place and returns self for chaining.
        """
        if noise_type == 'white':
            noise = torch.randn_like(self.tensor)
        elif noise_type == 'brown':
            noise = torch.cumsum(torch.randn_like(self.tensor), dim=-1)
            noise = noise - noise.mean()
        elif noise_type == 'pink':
            white_noise = torch.randn_like(self.tensor)
            fft = torch.fft.rfft(white_noise, dim=-1)
            frequencies = torch.fft.rfftfreq(self.tensor.shape[-1], d=1.0)
            fft[:, 1:] /= torch.sqrt(frequencies[1:])
            noise = torch.fft.irfft(fft, n=self.tensor.shape[-1], dim=-1)
        else:
            return self
            
        noise_rms = torch.sqrt(torch.mean(noise ** 2))
        if noise_rms > 0:
            noise = noise * (target_rms / noise_rms)
            
        self.tensor += noise
        return self

    def bandpass(self, highpass_hz: int = None, lowpass_hz: int = None):
        """
        Applies a brickwall highpass and/or lowpass filter using FFT.
        Modifies the tensor in-place and returns self for chaining.
        """
        if not highpass_hz and not lowpass_hz:
            return self
            
        n = self.tensor.shape[-1]
        fft_data = torch.fft.rfft(self.tensor, dim=-1)
        freqs = torch.fft.rfftfreq(n, d=1.0 / self.sample_rate).to(self.tensor.device)
        
        if highpass_hz:
            fft_data[..., freqs < highpass_hz] = 0.0
        if lowpass_hz:
            fft_data[..., freqs > lowpass_hz] = 0.0
            
        self.tensor = torch.fft.irfft(fft_data, n=n, dim=-1)
        return self

    def mix_with(self, other: 'Waveform', start_sample: int = 0):
        """
        Mixes another Waveform into this one at a specified offset.
        Modifies the tensor in-place and returns self for chaining.
        """
        if self.sample_rate != other.sample_rate:
            raise ValueError("Sample rates must match to mix waveforms.")
            
        length = other.tensor.shape[1]
        max_end = min(self.tensor.shape[1], start_sample + length)
        mix_length = max_end - start_sample
        
        if mix_length > 0:
            self.tensor[:, start_sample:max_end] += other.tensor[:, :mix_length]
        return self

    def play(self):
        """
        Plays the audio directly using pydub. 
        Useful for quick debugging without saving files.
        """
        from pydub import AudioSegment
        from pydub.playback import play as pydub_play

        arr = self.tensor.detach().cpu().numpy()
        
        # pydub expects interleaved data (samples, channels)
        if arr.ndim == 2:
            arr = arr.T
            
        # Prevent clipping distortion and convert to 16-bit PCM
        arr = np.clip(arr, -1.0, 1.0)
        arr = (arr * 32767).astype(np.int16)
        
        channels = 1 if arr.ndim == 1 else arr.shape[1]
        
        audio_segment = AudioSegment(
            arr.tobytes(),
            frame_rate=self.sample_rate,
            sample_width=arr.dtype.itemsize,
            channels=channels
        )
        pydub_play(audio_segment)
        return self