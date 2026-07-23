import csv
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple

import torch
import torchaudio
from synthesiser.config import Config

@dataclass
class AudioRecord:
    """Represents a single audio file and its metadata in the catalog."""
    path: Path
    label: str
    class_id: Optional[int] = None
    highpass_hz: int = 0
    lowpass_hz: int = 0          # 0 means "use nyquist" — resolved at synthesis time
    tags: Dict[str, str] = field(default_factory=dict)

class Catalog:
    """
    Manages the dataset paths and provides sampling mechanisms for the synthesis pipeline.
    Expects a folder-based structure where subfolders in the vocalisations directory 
    represent species/class labels.
    """
    
    ALLOWED_EXTENSIONS: Set[str] = {'.wav', '.mp3', '.flac'}

    def __init__(self, config: Config, limit_per_class: Optional[int] = None, use_raw_vocalisations: bool = False, sample_seed: Optional[int] = None):
        self.config = config
        self.limit_per_class = limit_per_class
        self.use_raw_vocalisations = use_raw_vocalisations
        self.sample_seed = sample_seed
        self.positives: List[AudioRecord] = []
        self.negatives: List[AudioRecord] = []
        self.backgrounds: List[AudioRecord] = []
        self.species_map: Dict[str, int] = {}
        self.class_counts: Dict[str, int] = {}
        self._build_catalog()
        self._print_summary()

    def _print_summary(self):
        print("\n=== Dataset Summary ===")
        
        # Positives (vocalisations)
        pos_samples = len(self.positives)
        pos_classes = len(self.species_map)
        print(f"vocalisations: {pos_samples} samples / {pos_classes} classes")
        
        # Negatives
        neg_samples = len(self.negatives)
        # Using parent directory name as a proxy for the negative class/category
        neg_classes = len(set(r.path.parent.name for r in self.negatives)) if neg_samples > 0 else 0
        print(f"negative: {neg_samples} samples / {neg_classes} classes")
        
        # Backgrounds (noise)
        bg_samples = len(self.backgrounds)
        print(f"noise: {bg_samples} samples")
        
        # Proportions
        if pos_samples > 0:
            print("\nvocalisations (proportions):")
            class_counts = {}
            for r in self.positives:
                class_counts[r.label] = class_counts.get(r.label, 0) + 1
                
            # Sort by sample count descending
            sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
            for label, count in sorted_classes:
                pct = (count / pos_samples) * 100
                print(f"- {label}: {count} samples - {pct:.1f}%")
                
        print("=======================\n")

    def _is_audio(self, path: Path) -> bool:
        return path.suffix.lower() in self.ALLOWED_EXTENSIONS and not path.name.startswith('.')
    
    def _read_tags(self, directory: Path) -> Dict[str, Dict[str, str]]:
        """Reads tags.csv if present and maps filenames to their row dictionary."""
        tags_file = directory / 'tags.csv'
        tags_data = {}
        if tags_file.exists():
            with open(tags_file, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    filename = row.get('filename')
                    if filename:
                        tags_data[filename] = row
        return tags_data

    def _build_catalog(self):
        """Scans directories and populates the catalog."""
        self._load_positives()
        self._load_negatives()
        self._load_backgrounds()

    def _load_positives(self):
        class_id_counter = 0
        voc_paths = self.config.paths.vocalisations_raw if self.use_raw_vocalisations else self.config.paths.vocalisations
        for voc_dir_str in voc_paths:
            voc_dir = Path(voc_dir_str)
            if not voc_dir.exists():
                print(f"Warning: Vocalisations directory not found: {voc_dir}")
                continue

            tags = self._read_tags(voc_dir)
            
            for subdir in voc_dir.iterdir():
                if not subdir.is_dir():
                    continue
                    
                species = subdir.name
                if species not in self.species_map:
                    self.species_map[species] = class_id_counter
                    class_id_counter += 1
                    
                class_id = self.species_map[species]
                if species not in self.class_counts:
                    self.class_counts[species] = 1

                # Sort for determinism, then apply a fixed-seed per-species shuffle.
                # Slicing an increasing limit against the same seed gives nested,
                # non-repeating growth across sweep steps (2 ⊂ 4 ⊂ 6 ⊂ 8) instead
                # of always taking the same filesystem-order prefix.
                files = sorted(f for f in subdir.rglob('*') if self._is_audio(f))
                if self.sample_seed is not None:
                    random.Random(self.sample_seed).shuffle(files)
                if self.limit_per_class:
                    if len(files) < self.limit_per_class:
                        print(f"Warning: only {len(files)} files available for {species} (requested limit {self.limit_per_class}).")
                    files = files[:self.limit_per_class]

                for file_path in files:
                    file_tags = tags.get(file_path.name, {})
                    self.positives.append(AudioRecord(path=file_path, label=species, class_id=class_id, tags=file_tags))

    def _load_negatives(self):
        for neg_dir_str in self.config.paths.negative:
            neg_dir = Path(neg_dir_str)
            if not neg_dir.exists():
                print(f"Warning: Negative directory not found: {neg_dir}. Skipping.")
                continue

            tags = self._read_tags(neg_dir)
            for file_path in neg_dir.rglob('*'):
                if self._is_audio(file_path):
                    file_tags = tags.get(file_path.name, {})
                    self.negatives.append(AudioRecord(path=file_path, label='negative', tags=file_tags))

    def _load_backgrounds(self):
        target_len = self.config.synthesis.length_seconds
        sample_rate = self.config.synthesis.sample_rate

        for bg_dir_str in self.config.paths.noise:
            bg_dir = Path(bg_dir_str)
            if not bg_dir.exists():
                print(f"Warning: Noise directory not found: {bg_dir}. Skipping.")
                continue

            for file_path in bg_dir.rglob('*'):
                if self._is_audio(file_path):
                    info = torchaudio.info(str(file_path))
                    duration_s = info.num_frames / info.sample_rate
                    
                    if duration_s < target_len:
                        print(f"   [Catalog] Reassigned {file_path.name} to negatives: too short ({duration_s:.1f}s < {target_len}s)")
                        self.negatives.append(AudioRecord(path=file_path, label='negative'))
                        continue

                    hp_hz, lp_hz = self._detect_background_bandpass(
                        file_path, sample_rate, self.config.spectrogram.n_fft,
                        self.config.spectrogram.hop_length, self.config.spectrogram.win_length,
                    )
                    print(f"   [Catalog] {file_path.name}: detected bandpass HP={hp_hz}Hz LP={lp_hz}Hz")

                    record = AudioRecord(
                        path=file_path, label='background',
                        highpass_hz=hp_hz, lowpass_hz=lp_hz,
                    )
                    self.backgrounds.append(record)

    def _detect_background_bandpass(
        self,
        file_path: Path,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        win_length: int,
    ) -> tuple[int, int]:
        """
        Loads a short clip of the background file at its *original* sample rate
        and estimates bandpass bounds. The returned lowpass_hz is capped at the
        original nyquist, so it reflects the true hardware ceiling even after
        the waveform is later resampled for synthesis.
        """
        from synthesiser.spectrogram import Spectrogram
        tensor, sr = torchaudio.load(str(file_path))
        if tensor.shape[0] > 1:
            tensor = tensor.mean(dim=0, keepdim=True)
        original_nyquist = sr // 2
        # Use at most 30 seconds at the original sample rate — no resampling
        max_samples = sample_rate * 30
        if tensor.shape[1] > max_samples:
            tensor = tensor[:, :max_samples]
        spec = Spectrogram(tensor, sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        spec.to_real(power=2.0)
        hp_hz, lp_hz = spec.detect_bandpass()
        # Cap LP at original nyquist — this is the true hardware ceiling
        lp_hz = min(lp_hz, original_nyquist)
        return hp_hz, lp_hz

    # --- Sampling Methods for Synthesis ---

    def sample_background(self) -> AudioRecord:
        """Returns a random background noise record."""
        if not self.backgrounds:
            raise ValueError("No background audio files available in the catalog.")
        return random.choice(self.backgrounds)

    def sample_positive(self, species: Optional[str] = None) -> AudioRecord:
        """
        Returns a random positive (vocalisation) record.
        Optionally filters by a specific species/class.
        """
        if not self.positives:
            raise ValueError("No positive audio files available in the catalog.")
            
        if species:
            filtered = [record for record in self.positives if record.label == species]
            if not filtered:
                raise ValueError(f"No positive audio files found for species: {species}")
            return random.choice(filtered)
            
        return random.choice(self.positives)

    def sample_negative(self) -> AudioRecord:
        if not self.negatives:
            raise ValueError("No negative audio files available in the catalog.")
        return random.choice(self.negatives)

    def sample_species(self) -> str:
        """
        Returns a species label sampled inversely proportional to play counts,
        so under-represented classes are preferred. All species start at 1,
        giving a uniform prior before any plays are recorded.
        """
        species = list(self.class_counts.keys())
        total = sum(self.class_counts.values())
        weights = [total / self.class_counts[s] for s in species]
        return random.choices(species, weights=weights, k=1)[0]

    def record_play(self, species: str):
        """Increments the play counter for a species after a successful placement."""
        if species in self.class_counts:
            self.class_counts[species] += 1

    def get_species_names(self) -> List[str]:
        sorted_items = sorted(self.species_map.items(), key=lambda item: item[1])
        return [item[0] for item in sorted_items]


if __name__ == "__main__":
    from config import load_config
    
    # Test execution
    try:
        config = load_config('config.yaml')
        catalog = Catalog(config)
        
        print(f"Loaded Catalog:")
        print(f" - Positives: {len(catalog.positives)}")
        print(f" - Negatives: {len(catalog.negatives)}")
        print(f" - Backgrounds: {len(catalog.backgrounds)}")
        print(f" - Species Map: {catalog.species_map}")
        
        if catalog.positives:
            sample = catalog.sample_positive()
            print(f"Sample positive: {sample.path.name} (Class: {sample.label}, ID: {sample.class_id})")
            
    except Exception as e:
        print(f"Catalog initialization failed (check if directories exist): {e}")