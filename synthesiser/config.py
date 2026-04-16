# config loading, defaults
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

@dataclass
class PathsConfig:
    vocalisations: str
    negative: str
    noise: str
    output: str

@dataclass
class SynthesisConfig:
    n_soundscapes: int
    length_seconds: int
    sample_rate: int
    positive_overlay_range: Tuple[int, int]
    negative_overlay_range: Tuple[int, int]
    repetitions: Tuple[int, int]
    snr_range: Tuple[float, float]

@dataclass
class ProportionsConfig:
    species: Dict[str, float] = field(default_factory=dict)
    noise: Dict[str, float] = field(default_factory=dict)

@dataclass
class OutputConfig:
    include_audio: bool
    include_spectrogram: bool
    include_masks: bool  # Implicitly UNet++ per project docs
    include_boxes: bool
    val_ratio: float
    overwrite: bool
    color_mode: str

@dataclass
class SpectrogramConfig:
    n_fft: int
    win_length: int
    hop_length: int

@dataclass
class Config:
    paths: PathsConfig
    synthesis: SynthesisConfig
    proportions: ProportionsConfig
    output: OutputConfig
    spectrogram: SpectrogramConfig

    def validate(self):
        """Performs logical validation on the configuration state."""
        
        # Validate proportions
        species_sum = sum(self.proportions.species.values())
        if species_sum > 1.0:
            raise ValueError(f"Species proportions sum to {species_sum}, which is > 1.0")
            
        noise_sum = sum(self.proportions.noise.values())
        if noise_sum > 1.0:
            raise ValueError(f"Noise proportions sum to {noise_sum}, which is > 1.0")

        # Validate ranges
        if self.synthesis.positive_overlay_range[0] > self.synthesis.positive_overlay_range[1]:
            raise ValueError("positive_overlay_range min cannot be greater than max.")

        if self.synthesis.snr_range[0] > self.synthesis.snr_range[1]:
            raise ValueError("snr_range min cannot be greater than max.")

def load_config(yaml_path: str = "config.yaml") -> Config:
    """Loads, parses, and validates the YAML configuration into a Config object."""
    with open(yaml_path, 'r') as f:
        raw = yaml.safe_load(f)

    # Convert lists to tuples for range parameters
    synth_raw = raw.get('synthesis', {})
    synth_raw['positive_overlay_range'] = tuple(synth_raw.get('positive_overlay_range', [1, 1]))
    synth_raw['negative_overlay_range'] = tuple(synth_raw.get('negative_overlay_range', [0, 0]))
    synth_raw['repetitions'] = tuple(synth_raw.get('repetitions', [1, 1]))
    synth_raw['snr_range'] = tuple(synth_raw.get('snr_range', [0.1, 1.0]))

    # Handle potentially null/empty proportion dictionaries
    prop_raw = raw.get('proportions', {})
    species_props = prop_raw.get('species') or {}
    noise_props = prop_raw.get('noise') or {}

    config = Config(
        paths=PathsConfig(**raw.get('paths', {})),
        synthesis=SynthesisConfig(**synth_raw),
        proportions=ProportionsConfig(species=species_props, noise=noise_props),
        output=OutputConfig(**raw.get('output', {})),
        spectrogram=SpectrogramConfig(**raw.get('spectrogram', {}))
    )

    config.validate()
    return config

if __name__ == "__main__":
    # Test execution
    config = load_config('config.yaml')
    print(f"Loaded config successfully. Planned soundscapes: {config.synthesis.n_soundscapes}")