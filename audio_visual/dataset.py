# dataloader_simple
# 
# This Dataset class only loads a FIXED number of audio files, and adds `n_gaps_per_audio` gaps to each audio file

import os
from pathlib import Path
import yaml
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append("..")
import utils
import librosa

def load_config(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    return cfg

class LibriSpeechDataset(Dataset):
    def __init__(self, config_path, dataset_type="train"):
        # Load YAML and extract dataset parameters
        full_cfg = load_config(config_path)
        data_cfg = full_cfg['data']

        self.root_dir       = data_cfg['root_path']
        self.n_fft          = data_cfg['spectrogram']['n_fft']
        self.hop_len        = data_cfg['spectrogram']['hop_length']
        self.win_len        = data_cfg['spectrogram']['win_length']

        self.sr             = data_cfg['sample_rate']
        self.max_len_s      = data_cfg['max_len_s']
        self.gap_len_s      = data_cfg['gap_len_s']

        self.max_files      = data_cfg['train_limit']
        self.gaps_per_audio = data_cfg['gaps_per_audio']

        # Determine dataset path
        if (dataset_type == 'train'):
            data_path_key = 'train_path'
        elif (dataset_type == 'valid'):
            data_path_key = 'valid_path'
        elif (dataset_type == 'test'):
            data_path_key = 'test_path'
        else:
            raise ValueError(f"Invalid dataset_type: {dataset_type}")

        self.root_path = Path(data_cfg['root_path'])
        self.dataset_dir = self.root_path / data_cfg[data_path_key]

        # Check if path is valid
        if (not os.path.exists(self.dataset_dir)):
            raise ValueError(f"Path {self.dataset_dir} does not exist")

        # Recursively walk through the dataset and collect all FLAC file paths
        counter = 0
        self.file_paths = []
        for subdir, _, files in os.walk(self.dataset_dir):
            for file in files:
                if (file.endswith('.flac') and counter < self.max_files):
                    self.file_paths.append(os.path.join(subdir, file))
                    counter += 1

        # Ensure consistent order of file paths
        self.file_paths.sort()

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Get `n_gaps_per_audio` spectrogram with for a single audio file, all gap intervals [s], all gap masks, and all true spectrogram (no gap)

        Returns
        -------
        spectrogram_gap          : torch.Tensor - Spectrogram (log magnitude spectrogram, WITHOUT phase info) with 0.2 s gap applied
        gap_int_s                : torch.Tensor - Gap interval (start, end) in seconds
        gap_mask                 : torch.Tensor - Binary mask indicating gap location (1 for gap, 0 for rest/true audio)
        spectrogram_target_phase : torch.Tensor - True spectrogram (magnitude spectrogram with phase info, no gap)
        """
        file_path = self.file_paths[idx]

        # Generate n_gaps_per_audio
        # Pre-allocate tensors for speed with correct dimensions from configuration
        spectrogram_gaps           = torch.zeros((self.gaps_per_audio, self.n_fft // 2 + 1, math.ceil(self.sr * self.max_len_s / self.hop_len)), dtype=torch.float32)
        spectrogram_target_phases  = torch.zeros((self.gaps_per_audio, self.n_fft // 2 + 1, math.ceil(self.sr * self.max_len_s / self.hop_len)), dtype=torch.cfloat)
        gap_masks                  = torch.zeros((self.gaps_per_audio, self.n_fft // 2 + 1, math.ceil(self.sr * self.max_len_s / self.hop_len)), dtype=torch.float32)
        gap_ints                   = torch.zeros((self.gaps_per_audio, 2), dtype=torch.float32)
        for i in range(self.gaps_per_audio):
            # Load audio data
            audio_data, sample_rate   = utils.load_audio(file_path)

            # Obtain audio data with gap
            audio_data_gap, gap_int_s = utils.add_random_gap(file_path, self.gap_len_s)

            # Extract energy spectrogram (with phase info) for true audio (allows for better reconstruction later via iSTFT)
            # However, only extract log magnitude for the gap (this is what will be passed into the model)
            spectrogram_target_phase = utils.extract_spectrogram(audio_data, n_fft=self.n_fft, hop_length=self.hop_len, win_length=self.win_len)
            spectrogram_gap          = np.abs(utils.extract_spectrogram(audio_data_gap, n_fft=self.n_fft, hop_length=self.hop_len, win_length=self.win_len))        # Model does NOT support complex numbers, so we only take the magnitude

            # NEW: Convert magnitude spectrograms to log magnitude spectrograms (suggested normalization in Audio-Visual paper)
            spectrogram_gap          = np.log10(spectrogram_gap + 1e-9)           # Add small epsilon to avoid log(0)

            # Convert target and gap spectrograms to PyTorch tensors
            n_timeframes                 = spectrogram_target_phases.shape[2]
            spectrogram_target_phases[i] = torch.from_numpy(spectrogram_target_phase[:, :n_timeframes])             # Take care of any off-by-1 errors due to rounding
            spectrogram_gaps[i]          = torch.tensor(spectrogram_gap[:, :n_timeframes], dtype=torch.float32)
            gap_ints[i]                  = torch.tensor(gap_int_s, dtype=torch.float32)

            # Create gap mask
            gap_mask      = torch.zeros_like(spectrogram_gaps[i], dtype=torch.float32)
            gap_start_idx = librosa.time_to_frames(gap_int_s[0], sr=sample_rate, hop_length=self.hop_len)
            gap_end_idx   = librosa.time_to_frames(gap_int_s[1], sr=sample_rate, hop_length=self.hop_len)
            gap_mask[:, gap_start_idx:gap_end_idx] = 1
            gap_masks[i]  = gap_mask

        return spectrogram_gaps, gap_ints, gap_masks, spectrogram_target_phases

if __name__ == "__main__":
    # Load config
    config_path = 'cnn_blstm.yaml'
    config = load_config(config_path)
    config = config['data']['spectrogram']

    dataset = LibriSpeechDataset(config_path)

    data_loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True
    )

    # Usage Example
    for batch_idx, (spectrogram_gap, gap_int_s, gap_mask, spectrogram_target_phase) in enumerate(data_loader):
        print(f"Batch {batch_idx}")
        print(f"Gap size: {gap_int_s.shape}")
        print(f"Gap interval: start={gap_int_s[0, 0, 0]:.3f} s, end={gap_int_s[0, 0, 1]:.3f} s, length={gap_int_s[0, 0, 1] - gap_int_s[0, 0, 0]:.3f} s")
        print(f"Spectrogram gap shape: {spectrogram_gap.shape}")                    # Should be (batch_size, max_length, channels)
        print(f"Spectrogram orig (w/ phase) shape: {spectrogram_target_phase.shape}")
        print(f"Gap mask shape: {gap_mask.shape}")

        print()
        print(f"min/max spectrogram gap: {spectrogram_gap.min()}, {spectrogram_gap.max()}")

        # Visualize the histograms
        import matplotlib.pyplot as plt
        fig1 = utils.visualize_spectrogram(np.abs(spectrogram_target_phase[0, 0]), 
                                           in_db=False, 
                                           n_fft=config['n_fft'], 
                                           win_length=config['win_length'], 
                                           hop_length=config['hop_length'], 
                                           title="Original Audio Spectrogram")
        fig2 = utils.visualize_spectrogram(10 ** spectrogram_gap[0, 0], 
                                           in_db=False, gap_int=tuple(gap_int_s[0, 0]), 
                                           n_fft=config['n_fft'], 
                                           win_length=config['win_length'], 
                                           hop_length=config['hop_length'], 
                                           title="Original Audio Spectrogram with Gap (White)")
        fig3 = utils.visualize_spectrogram(spectrogram_target_phase[0, 0] * gap_mask[0, 0],
                                           in_db=False, gap_int=tuple(gap_int_s[0, 0]), 
                                           n_fft=config['n_fft'], 
                                           win_length=config['win_length'], 
                                           hop_length=config['hop_length'], 
                                           title="Gap Spectrogram")

        # Plot the gap mask as a heatmap
        plt.figure(figsize=(10, 4))
        plt.title("Gap Mask Heatmap")
        plt.imshow(gap_mask[0, 0], aspect='auto', origin='lower', cmap='gray')
        plt.colorbar(label="Mask Value (0 or 1)")
        plt.xlabel("Sample Index")
        plt.ylabel("Frequency")

        # Save a dummy audio file
        spectrogram_target_phase_sample = (spectrogram_target_phase[0, 0]).detach().cpu().numpy()
        utils.save_audio(utils.spectrogram_to_audio(spectrogram_target_phase_sample, 
                                                    phase_info=True, n_fft=config['n_fft'], 
                                                    win_length=config['win_length'], 
                                                    hop_length=config['hop_length']), 
                                                    f"output/dataloader_true_audio_test_{batch_idx}.flac")

        plt.show()
        break  # Just load one batch for demo