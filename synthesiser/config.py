import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

@dataclass
class PathsConfig:
    vocalisations: List[str]
    negative: List[str]
    noise: List[str]
    output: str
    vocalisations_raw: List[str] = field(default_factory=list)

@dataclass
class SyntheticNoiseConfig:
    white: bool = True
    pink: bool = True
    brown: bool = True
    probability: float = 0.5
    db_range: Tuple[float, float] = (-46.0, -30.0)

@dataclass
class SynthesisConfig:
    n_soundscapes: int
    length_seconds: int
    sample_rate: int
    positive_overlay_range: Tuple[int, int]
    negative_overlay_range: Tuple[int, int]
    negative_snr_range: Tuple[float, float]
    repetitions: Tuple[int, int]
    repetitions_spacing_s: Tuple[float, float]
    snr_range: Tuple[float, float]
    minimum_mask_area_px: int = 50
    mask_threshold_db: float = 5.0
    edge_fade_ms: int = 20
    synthetic_noise: SyntheticNoiseConfig = field(default_factory=SyntheticNoiseConfig)

@dataclass
class ProportionsConfig:
    species: Dict[str, float] = field(default_factory=dict)
    noise: Dict[str, float] = field(default_factory=dict)

@dataclass
class OutputConfig:
    include_audio: bool
    include_spectrogram: bool
    include_masks: bool
    include_boxes: bool
    overwrite: bool
    color_mode: str
    include_presence: bool = False
    include_simple_labels: bool = True
    generate_raw_dataset: bool = False
    ignore_classes: List[str] = field(default_factory=list)
    target_db: float = -10.0
    val_ratio: float = 0.8

@dataclass
class SpectrogramConfig:
    n_fft: int
    win_length: int
    hop_length: int
    log_base: float = 10.0

@dataclass
class Config:
    paths: PathsConfig
    synthesis: SynthesisConfig
    proportions: ProportionsConfig
    output: OutputConfig
    spectrogram: SpectrogramConfig

    def validate(self):
        species_sum = sum(self.proportions.species.values())
        if species_sum > 1.0:
            raise ValueError(f"Species proportions sum to {species_sum}, which is > 1.0")
            
        noise_sum = sum(self.proportions.noise.values())
        if noise_sum > 1.0:
            raise ValueError(f"Noise proportions sum to {noise_sum}, which is > 1.0")

        if self.synthesis.positive_overlay_range[0] > self.synthesis.positive_overlay_range[1]:
            raise ValueError("positive_overlay_range min cannot be greater than max.")

        if self.synthesis.snr_range[0] > self.synthesis.snr_range[1]:
            raise ValueError("snr_range min cannot be greater than max.")

        if self.synthesis.negative_snr_range[0] > self.synthesis.negative_snr_range[1]:
            raise ValueError("negative_snr_range min cannot be greater than max.")

def _ensure_list(val) -> List[str]:
    if isinstance(val, str):
        return [val]
    elif isinstance(val, list):
        return [str(v) for v in val]
    return []

def load_config(yaml_path: str = "config.yaml") -> Config:
    with open(yaml_path, 'r') as f:
        raw = yaml.safe_load(f)

    synth_raw = raw.get('synthesis', {})
    synth_raw['positive_overlay_range'] = tuple(synth_raw.get('positive_overlay_range', [1, 1]))
    synth_raw['negative_overlay_range'] = tuple(synth_raw.get('negative_overlay_range', [0, 0]))
    synth_raw['repetitions'] = tuple(synth_raw.get('repetitions', [1, 1]))
    synth_raw['negative_snr_range'] = tuple(synth_raw.get('negative_snr_range', [0.1, 1.0]))
    synth_raw['repetitions_spacing_s'] = tuple(synth_raw.get('repetitions_spacing_s', [0.5, 3.0]))
    synth_raw['snr_range'] = tuple(synth_raw.get('snr_range', [0.1, 1.0]))
    
    noise_raw = synth_raw.pop('synthetic_noise', {})
    if 'db_range' in noise_raw:
        noise_raw['db_range'] = tuple(noise_raw['db_range'])
    synth_noise_config = SyntheticNoiseConfig(**noise_raw)
    synth_raw['synthetic_noise'] = synth_noise_config

    prop_raw = raw.get('proportions', {})
    species_props = prop_raw.get('species') or {}
    noise_props = prop_raw.get('noise') or {}

    raw_paths = raw.get('paths', {})
    paths_config = PathsConfig(
        vocalisations=_ensure_list(raw_paths.get('vocalisations')),
        vocalisations_raw=_ensure_list(raw_paths.get('vocalisations_raw')),
        negative=_ensure_list(raw_paths.get('negative')),
        noise=_ensure_list(raw_paths.get('noise')),
        output=raw_paths.get('output', 'output')
    )

    raw_output = raw.get('output', {})
    if 'ignore_classes' in raw_output:
        raw_output['ignore_classes'] = _ensure_list(raw_output['ignore_classes'])

    config = Config(
        paths=paths_config,
        synthesis=SynthesisConfig(**synth_raw),
        proportions=ProportionsConfig(species=species_props, noise=noise_props),
        output=OutputConfig(**raw_output),
        spectrogram=SpectrogramConfig(**raw.get('spectrogram', {}))
    )

    config.validate()
    return config