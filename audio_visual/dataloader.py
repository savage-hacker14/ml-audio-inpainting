import os
import yaml
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append("..")
import utils
import librosa
from config import LIBRISPEECH_ROOT_PROCESSED

class LibriSpeechDataset(Dataset):
    def __init__(self, root_dir, n_fft, hop_len, win_len, transform=None):
        self.root_dir   = root_dir
        self.transform  = transform
        self.n_fft      = n_fft
        self.hop_len    = hop_len
        self.win_len    = win_len
        self.file_paths = []

        # Recursively walk through the dataset and collect all FLAC file paths
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.flac'):
                    self.file_paths.append(os.path.join(subdir, file))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Get spectrogram with hap, gap interval [s], gap mask, and true spectrogram (no gap)

        Returns
        -------
        spectrogram_gap          : torch.Tensor - Spectrogram (log magnitude spectrogram, WITHOUT phase info) with 0.2 s gap applied
        gap_int_s                : torch.Tensor - Gap interval (start, end) in seconds
        gap_mask                 : torch.Tensor - Binary mask indicating gap location (1 for gap, 0 for rest/true audio)
        spectrogram_target_phase : torch.Tensor - True spectrogram (magnitude spectrogram with phase info, no gap)
        """
        file_path = self.file_paths[idx]

        # Load audio data (magnitude only)
        audio_data, sample_rate   = utils.load_audio(file_path)
        audio_data_gap, gap_int_s = utils.add_random_gap(file_path, 0.2)
        #print(f"Gap interval: {gap_int_s}")

        # Extract energy spectrogram (with phase info) for true audio (allows for better reconstruction later via iSTFT in test.py)
        # However, only extract log magnitude for the gap (this is what will be passed into the model)
        spectrogram_target_phase = utils.extract_spectrogram(audio_data, n_fft=self.n_fft, hop_length=self.hop_len, win_length=self.win_len)
        spectrogram_gap          = np.abs(utils.extract_spectrogram(audio_data_gap, n_fft=self.n_fft, hop_length=self.hop_len, win_length=self.win_len))

        # NEW: Convert magnitude spectrograms to log magnitude spectrograms (suggested normalization in Audio-Visual paper)
        #spectrogram_target = librosa.power_to_db(spectrogram_target, ref=np.max)
        spectrogram_gap          = np.log10(spectrogram_gap + 1e-6)           # Add small epsilon to avoid log(0)

        # Convert target and gap spectrograms to PyTorch tensors
        spectrogram_target_phase = torch.from_numpy(spectrogram_target_phase)
        spectrogram_gap          = torch.tensor(spectrogram_gap, dtype=torch.float32)
        gap_int_s                = torch.tensor(gap_int_s, dtype=torch.float32)

        # Create gap mask
        gap_mask      = torch.zeros_like(spectrogram_gap, dtype=torch.float32)
        gap_start_idx = librosa.time_to_frames(gap_int_s[0], sr=sample_rate, hop_length=self.hop_len)
        gap_end_idx   = librosa.time_to_frames(gap_int_s[1], sr=sample_rate, hop_length=self.hop_len)
        gap_mask[:, gap_start_idx:gap_end_idx] = 1
        
        #spectrogram_gap    = spectrogram_target * (1 - gap_mask)
        #print(f"Spectrogram size: {spectrogram_gap.shape}, audio_data size: {audio_data.shape}")
        #print(f"Gap inserted at t = {gap_int_s[0]:.2f} s with length {(gap_int_s[1] - gap_int_s[0]):.2f} s")

        return spectrogram_gap, gap_int_s, gap_mask, spectrogram_target_phase

if __name__ == "__main__":
    # Load BLSTM config
    with open('blstm.yaml', 'r') as f:
        config = yaml.safe_load(f)


    dataset = LibriSpeechDataset(root_dir=LIBRISPEECH_ROOT_PROCESSED, 
                                 n_fft=config['n_fft'], 
                                 hop_len=config['hop_length'],
                                 win_len=config['hann_win_length'])

    data_loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        #collate_fn=collate_fn
    )

    # Usage Example
    for batch_idx, (spectrogram_gap, gap_int_s, gap_mask, spectrogram_target_phase) in enumerate(data_loader):
        print(f"Batch {batch_idx}")
        print(f"Gap size: {gap_int_s.shape}")
        print(f"Gap interval: start={gap_int_s[0, 0]:.3f} s, end={gap_int_s[0, 1]:.3f} s, length={gap_int_s[0, 1] - gap_int_s[0, 0]:.3f} s")
        print(f"Spectrogram gap shape: {spectrogram_gap.shape}")                    # Should be (batch_size, max_length, channels)
        print(f"Spectrogram orig (w/ phase) shape: {spectrogram_target_phase.shape}")
        print(f"Gap mask shape: {gap_mask.shape}")

        print()
        print(f"min/max spectrogram gap: {spectrogram_gap.min()}, {spectrogram_gap.max()}")
        #print(f"min/max spectrogram orig: {spectrogram_target_phase.min()}, {spectrogram_target_phase.max()}")         # .min() doesn't work for torch ComplexFloat

        # Visualize the histograms
        import matplotlib.pyplot as plt
        fig1 = utils.visualize_spectrogram(np.abs(spectrogram_target_phase[0]), in_db=False, power=1, title="Original Audio Spectrogram")
        fig2 = utils.visualize_spectrogram(10 ** spectrogram_gap[0], in_db=False, power=1, gap_int=tuple(gap_int_s[0]), title="Gap Audio Spectrogram")

        # Plot the gap mask as a heatmap
        plt.figure(figsize=(10, 4))
        plt.title("Gap Mask Heatmap")
        plt.imshow(gap_mask[0], aspect='auto', origin='lower', cmap='gray')
        plt.colorbar(label="Mask Value (0 or 1)")
        plt.xlabel("Sample Index")
        plt.ylabel("Frequency")

        plt.show()
        break  # Just load one batch for demo