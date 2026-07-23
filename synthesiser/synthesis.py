import math
import random
import torch
import numpy as np
from typing import List, Tuple, Dict, Any
from synthesiser.config import Config
from synthesiser.catalog import AudioRecord
from synthesiser.catalog import Catalog
from synthesiser.waveform import Waveform
from synthesiser.spectrogram import Spectrogram
from synthesiser.visualisation import plot_spectrogram
from PIL import Image

class SoundscapeSynthesiser:
    """Core logic for constructing the synthetic soundscape from waveforms."""
    
    def __init__(self, config: Config):
        self.config = config

    def generate(self, bg_record: AudioRecord, negatives: List[AudioRecord], positives: List[AudioRecord], catalog: Catalog) -> Tuple[Waveform, List[Dict[str, Any]]]:
        """Constructs a soundscape mix containing a background, noise, negatives, and positives."""
        print(f"\n[Synthesis] Starting mix with background: {bg_record.path.name}")
        
        # --- Phase 1: Background & Artificial Noise ---
        
        # 1. Load and Crop Background
        wf = Waveform.load(str(bg_record.path), target_sr=self.config.synthesis.sample_rate)
        wf.crop(self.config.synthesis.length_seconds)
        
        # 2. Pre-Normalise to 0 dBFS to establish relative baseline
        wf.set_db(0.0)
        
        # 3. Add Synthetic Noise (before bandpass so noise outside the band is removed too)
        sn_config = self.config.synthesis.synthetic_noise
        added_noises = []
        for n_type, is_enabled in [('white', sn_config.white), ('pink', sn_config.pink), ('brown', sn_config.brown)]:
            if is_enabled and random.random() < sn_config.probability:
                db_target = random.uniform(sn_config.db_range[0], sn_config.db_range[1])
                target_rms = 10 ** (db_target / 20.0)
                wf.add_noise(n_type, target_rms)
                added_noises.append(f"{n_type} ({db_target:.1f} dB)")
        if added_noises:
            print(f"            Added synthetic noise: {', '.join(added_noises)}")

        # 4. Apply Bandpass
        nyquist = self.config.synthesis.sample_rate // 2
        hp_hz = bg_record.highpass_hz if bg_record.highpass_hz > 0 else None
        lp_hz = bg_record.lowpass_hz or None
        if hp_hz or lp_hz:
            wf.bandpass(highpass_hz=hp_hz, lowpass_hz=lp_hz)
            print(f"Applied detected bandpass: HP={hp_hz}Hz LP={lp_hz}Hz")

        # Precompute STFT bin indices for background bandpass region.
        # Used in Phase 3 to confine noise-power estimates and validate mask area.
        n_bins = self.config.spectrogram.n_fft // 2 + 1
        hz_per_bin = nyquist / (n_bins - 1)
        bp_hp_bin = int(math.ceil((hp_hz or 0) / hz_per_bin))
        bp_lp_bin = int(math.floor((lp_hz or nyquist) / hz_per_bin))
        bp_lp_bin = min(bp_lp_bin, n_bins - 1)

        # 5. Final Normalisation
        wf.set_db(0.0)
        print("            Normalised background mix to 0.0 dBFS")
        
        # --- Phase 2: Negative/Adversarial Overlays ---

        for neg_record in negatives:
            neg_wf = Waveform.load(str(neg_record.path), target_sr=self.config.synthesis.sample_rate)
            bg_samples = wf.tensor.shape[1]
            neg_samples = neg_wf.tensor.shape[1]

            # Require at least 1 second of overlap, or the full file if it's shorter than 1 sec
            min_overlap = min(self.config.synthesis.sample_rate, neg_samples)

            # Calculate offset boundaries ensuring min_overlap logic applies whether longer or shorter than bg
            min_offset = min_overlap - neg_samples
            max_offset = bg_samples - min_overlap
            start_offset = random.randint(min_offset, max_offset)
            
            # Trim the overlay so any overhangs are sliced off, returns corrected >= 0 start sample
            actual_start_sample = neg_wf.trim_to_mix(start_offset, bg_samples)
            
            # Apply slow 0.5s sinusoidal smoothing on-off for negatives (treating them as transients)
            sine_fade_samples = int(0.5 * self.config.synthesis.sample_rate)
            neg_wf.sine_fade(sine_fade_samples)
            
            # Fade the negative overlay to avoid harsh cut-ins
            fade_samples = int((self.config.synthesis.edge_fade_ms / 1000.0) * self.config.synthesis.sample_rate)
            neg_wf.fade(fade_samples)

            # Apply uniform SNR modification against the 0.0 dBFS background target
            snr = random.uniform(self.config.synthesis.negative_snr_range[0], self.config.synthesis.negative_snr_range[1])
            neg_db = 10 * math.log10(snr) + 0.0
            neg_wf.set_db(neg_db)
            wf.mix_with(neg_wf, start_sample=actual_start_sample)
            
            print(f"            Added negative: {neg_record.path.name} (SNR: {snr:.2f}, {neg_db:.1f} dB, Offset: {start_offset / self.config.synthesis.sample_rate:.2f}s)")

        # --- Phase 3: Positive Overlays (Vocalisations) ---

        annotations = []

        # Total successful plays to reach (across all positives and repetitions).
        total_target = random.randint(
            self.config.synthesis.positive_overlay_range[0],
            self.config.synthesis.positive_overlay_range[1],
        )
        max_attempts = total_target * 10
        successful_plays = 0
        attempt = 0

        print(f"            Adding positives: target={total_target} plays (max attempts={max_attempts})...")

        # Cache loaded waveforms within this soundscape to avoid redundant disk reads
        _wf_cache: Dict[str, Waveform] = {}
        
        current_pos_record = None
        current_reps_target = 0
        current_reps_successful = 0
        current_last_end_sample = 0

        while successful_plays < total_target and attempt < max_attempts:
            attempt += 1
            
            if current_pos_record is None or current_reps_successful >= current_reps_target:
                species = catalog.sample_species()
                current_pos_record = catalog.sample_positive(species=species) if positives else None
                if current_pos_record is None:
                    break
                current_reps_target = max(1, random.randint(
                    self.config.synthesis.repetitions[0],
                    self.config.synthesis.repetitions[1],
                ))
                current_reps_target = min(current_reps_target, total_target - successful_plays)
                current_reps_successful = 0
                current_last_end_sample = 0

            pos_record = current_pos_record

            # Load (or reuse cached) source waveform
            cache_key = str(pos_record.path)
            if cache_key not in _wf_cache:
                _wf_cache[cache_key] = Waveform.load(
                    cache_key, target_sr=self.config.synthesis.sample_rate
                )
            original_pos_wf = _wf_cache[cache_key]

            # 1. Clone and place at an offset
            pos_wf = Waveform(original_pos_wf.tensor.clone(), original_pos_wf.sample_rate)
            pos_wf.original_nyquist = original_pos_wf.original_nyquist
            bg_samples = wf.tensor.shape[1]
            pos_samples = pos_wf.tensor.shape[1]

            min_overlap = min(self.config.synthesis.sample_rate, pos_samples)

            if current_reps_successful == 0:
                min_offset = min_overlap - pos_samples
                max_offset = bg_samples - min_overlap
                start_offset = random.randint(min_offset, max_offset)
            else:
                spacing_s = random.uniform(
                    self.config.synthesis.repetitions_spacing_s[0],
                    self.config.synthesis.repetitions_spacing_s[1]
                )
                start_offset = current_last_end_sample + int(spacing_s * self.config.synthesis.sample_rate)

            # If the repetition falls outside the background (or doesn't meet min overlap),
            # abandon the remaining repetitions for this record.
            if start_offset > bg_samples - min_overlap:
                current_pos_record = None  # Force new record on next iteration
                continue

            actual_start_sample = pos_wf.trim_to_mix(start_offset, bg_samples)

            # 2. Normalise positive to 0 dBFS to use as a spectral weighting mask. Doing this before BP so to not amplify an empty background region if BP cuts out the vocalisation.
            pos_wf.set_db(0.0)

            # 3. Apply background bandpass to the positive waveform BEFORE scaling.
            # Field recordings often have massive out-of-band energy (like low wind rumble). 
            # If not removed here, the later in-band SNR scaling amplifies this invisible rumble.
            hp_hz = bg_record.highpass_hz if bg_record.highpass_hz > 0 else None
            lp_hz = bg_record.lowpass_hz or None
            if hp_hz or lp_hz:
                pos_wf.bandpass(highpass_hz=hp_hz, lowpass_hz=lp_hz)

                # Check if the bandpass removed almost all energy (i.e. vocalisation was out of band)
                bp_rms = pos_wf._get_rms()
                if bp_rms == 0 or 20 * math.log10(bp_rms) < -15.0:
                    bp_db = 20 * math.log10(bp_rms) if bp_rms > 0 else -float('inf')
                    print(f"            [Skip] {pos_record.path.name}: energy dropped to {bp_db:.1f}dB after bandpass (out-of-band) (attempt {attempt})")
                    current_pos_record = None
                    continue

            mix_spec = Spectrogram(
                wf.tensor, wf.sample_rate,
                n_fft=self.config.spectrogram.n_fft,
                hop_length=self.config.spectrogram.hop_length,
                win_length=self.config.spectrogram.win_length,
            ).to_real(power=2.0).values

            pos_spec = Spectrogram(
                pos_wf.tensor, pos_wf.sample_rate,
                n_fft=self.config.spectrogram.n_fft,
                hop_length=self.config.spectrogram.hop_length,
                win_length=self.config.spectrogram.win_length,
            ).to_real(power=2.0).values

            start_frame = actual_start_sample // self.config.spectrogram.hop_length
            mix_slice = mix_spec[..., start_frame: start_frame + pos_spec.shape[-1]]
            min_f = min(pos_spec.shape[-1], mix_slice.shape[-1])
            pos_spec_aligned = pos_spec[..., :min_f]
            mix_slice_aligned = mix_slice[..., :min_f]

            # 3. Confine noise-power estimate to background bandpass region.
            #    Bins outside [bp_hp_bin, bp_lp_bin] are near-zero in the background
            #    due to prior bandpass filtering; including them would dilute the
            #    weighted average downward.  We also cap lp at the positive file's
            #    own original nyquist to exclude upsampling aliases.
            eff_lp_bin = min(bp_lp_bin, int(math.floor(
                pos_wf.original_nyquist / hz_per_bin
            )))
            bin_mask = torch.zeros(n_bins, dtype=torch.bool, device=pos_spec_aligned.device)
            bin_mask[bp_hp_bin: eff_lp_bin + 1] = True

            pos_spec_bp = pos_spec_aligned[:, bin_mask, :]
            mix_slice_bp = mix_slice_aligned[:, bin_mask, :]

            pos_total_power = torch.sum(pos_spec_bp)
            if pos_total_power > 0:
                weighted_noise_power = torch.sum(pos_spec_bp * mix_slice_bp) / pos_total_power
            else:
                weighted_noise_power = torch.tensor(1e-10)
            local_noise_db = 10 * math.log10(weighted_noise_power.item() + 1e-10)

            # 4. Scale positive to target SNR relative to local noise floor
            snr = random.uniform(self.config.synthesis.snr_range[0], self.config.synthesis.snr_range[1])
            target_pos_db = local_noise_db + (10 * math.log10(snr))
            pos_wf.set_db(target_pos_db)

            # Fade clip edges AFTER scaling to ensure any brickwall FFT ringing 
            # from the earlier bandpass is cleanly tapered to exactly 0.0 at the boundaries.
            fade_samples = int((self.config.synthesis.edge_fade_ms / 1000.0) * self.config.synthesis.sample_rate)
            pos_wf.sine_fade(fade_samples)

            # 5. Compute mask and bounding box from the scaled and faded positive spectrogram
            final_pos_spec = Spectrogram(
                pos_wf.tensor, pos_wf.sample_rate,
                n_fft=self.config.spectrogram.n_fft,
                hop_length=self.config.spectrogram.hop_length,
                win_length=self.config.spectrogram.win_length,
            ).to_real(power=2.0).values

            threshold_db_above_noise = self.config.synthesis.mask_threshold_db
            threshold_power = weighted_noise_power.item() * (10 ** (threshold_db_above_noise / 10))

            box = None
            mask = None

            if self.config.output.include_boxes:
                freq_profile = final_pos_spec[0].max(dim=-1).values
                active_bins = torch.where(freq_profile > threshold_power)[0]
                active_bins = active_bins[(active_bins >= bp_hp_bin) & (active_bins <= eff_lp_bin)]
                if len(active_bins) > 0:
                    f_min = active_bins[0].item()
                    f_max = active_bins[-1].item()
                    box = [start_frame, start_frame + min_f, f_min, f_max]

            if self.config.output.include_masks:
                mask = (final_pos_spec[0] > threshold_power).byte().cpu().numpy()

            # 6. Bandpass-simulate the mask: zero rows outside [bp_hp_bin, eff_lp_bin],
            #    then resize to 640×640 and check surviving area against threshold.
            if mask is not None:
                bp_mask = mask.copy()
                if bp_hp_bin > 0:
                    bp_mask[:bp_hp_bin, :] = 0
                if eff_lp_bin < n_bins - 1:
                    bp_mask[eff_lp_bin + 1:, :] = 0

                # Resize to output resolution to evaluate area in pixel units
                bp_mask_img = Image.fromarray(bp_mask, mode='L').resize(
                    (640, 640), Image.Resampling.NEAREST
                )
                surviving_px = np.count_nonzero(np.array(bp_mask_img))

                if surviving_px < self.config.synthesis.minimum_mask_area_px:
                    print(
                        f"            [Skip] {pos_record.path.name}: mask area "
                        f"{surviving_px}px < {self.config.synthesis.minimum_mask_area_px}px "
                        f"after bandpass simulation (attempt {attempt})"
                    )
                    current_pos_record = None
                    continue  # discard — try again

                mask = bp_mask

            # 7. All checks passed: commit to the mix
            is_ignored = pos_record.label in self.config.output.ignore_classes
            has_label = (box is not None) or (mask is not None) or (not self.config.output.include_boxes and not self.config.output.include_masks)
            
            if not has_label and not is_ignored:
                print(f"            [Skip] {pos_record.path.name}: failed to generate valid label geometry (attempt {attempt})")
                current_pos_record = None
                continue
                
            if not is_ignored:
                annotations.append({
                    'record': pos_record,
                    'box': box,
                    'mask': mask,
                    'start_sample': actual_start_sample,
                    'end_sample': actual_start_sample + pos_wf.tensor.shape[1],
                    'start_offset_samples': start_offset,
                    'play_idx': successful_plays,
                })

            wf.mix_with(pos_wf, start_sample=actual_start_sample)
            
            padded_pos_tensor = torch.zeros_like(wf.tensor)
            end_sample = actual_start_sample + pos_wf.tensor.shape[1]
            padded_pos_tensor[:, actual_start_sample:end_sample] = pos_wf.tensor
            
            current_last_end_sample = end_sample

            successful_plays += 1
            current_reps_successful += 1
            catalog.record_play(pos_record.label)
            lbl_note = " (label ignored)" if is_ignored else ""
            print(
                f"            Added positive{lbl_note}: {pos_record.path.name} "
                f"[{successful_plays}/{total_target}] "
                f"(SNR: {snr:.2f}, Local Noise: {local_noise_db:.1f} dB, "
                f"Target: {target_pos_db:.1f} dB)"
            )
            
            fig_name = f"{pos_record.label} ({successful_plays})"
            
#            plot_spectrogram(
#                Spectrogram(
#                    padded_pos_tensor, pos_wf.sample_rate,
#                    n_fft=self.config.spectrogram.n_fft,
#                    hop_length=self.config.spectrogram.hop_length,
#                    win_length=self.config.spectrogram.win_length,
#                ),
#                show=False,
#                title=fig_name,
#            )

        if successful_plays < total_target:
            print(
                f"            [Warning] Only placed {successful_plays}/{total_target} positives "
                f"after {attempt} attempts — remaining attempts exhausted."
            )

        if hp_hz or lp_hz:
            wf.bandpass(highpass_hz=hp_hz, lowpass_hz=lp_hz)
            print(f"            Re-applied bandpass {hp_hz} - {lp_hz} to final mix for consistency")

        wf.set_db(self.config.output.target_db)
        print(f"            Final mix set to target dB: {self.config.output.target_db}")

#        plot_spectrogram(Spectrogram(wf.tensor, wf.sample_rate, n_fft=self.config.spectrogram.n_fft, hop_length=self.config.spectrogram.hop_length, win_length=self.config.spectrogram.win_length), show=True, annotations=annotations, title="Final Mix")
#        wf.play()

        return wf, annotations