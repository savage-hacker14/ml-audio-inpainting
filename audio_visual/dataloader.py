import os
import torch
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append("..")
import utils
from config import LIBRISPEECH_ROOT_PROCESSED

class LibriSpeechDataset(Dataset):
    def __init__(self, root_dir, n_fft=2048, hop_len=512, transform=None):
        self.root_dir   = root_dir
        self.transform  = transform
        self.n_fft      = n_fft
        self.hop_len    = hop_len
        self.file_paths = []

        # Recursively walk through the dataset and collect all FLAC file paths
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.flac'):
                    self.file_paths.append(os.path.join(subdir, file))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        audio_data, sample_rate = utils.load_audio(file_path)
        audio_data_gap, gap_int_s = utils.add_random_gap(file_path, 0.2)
        gap_int_s = torch.tensor(gap_int_s, dtype=torch.float32)
        #print(f"Gap interval: {gap_int_s}")

        spectrogram_target = utils.extract_spectrogram(audio_data, n_fft=self.n_fft) #torchaudio.load(file_path)
        spectrogram_gap    = utils.extract_spectrogram(audio_data_gap, n_fft=self.n_fft)

        # Convert target and gap spectrograms to PyTorch tensors
        spectrogram_target = torch.tensor(spectrogram_target, dtype=torch.float32)
        spectrogram_gap    = torch.tensor(spectrogram_gap, dtype=torch.float32)

        # Create gap mask
        gap_mask      = torch.zeros_like(spectrogram_target, dtype=torch.float32)
        gap_start_idx = int(gap_int_s[0] * sample_rate / self.hop_len)
        gap_end_idx   = int(gap_int_s[1] * sample_rate / self.hop_len)
        gap_mask[:, gap_start_idx:gap_end_idx] = 1

        #print(f"Spectrogram size: {spectrogram_gap.shape}, audio_data size: {audio_data.shape}")
        #print(f"Gap inserted at t = {gap_int_s[0]:.2f} s with length {(gap_int_s[1] - gap_int_s[0]):.2f} s")

        return spectrogram_gap, gap_int_s, gap_mask, spectrogram_target

if __name__ == "__main__":
    dataset = LibriSpeechDataset(root_dir=LIBRISPEECH_ROOT_PROCESSED)

    data_loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        #collate_fn=collate_fn
    )

    # Usage Example
    for batch_idx, (spectrogram_gap, gap_int_s, gap_mask, spectrogram_target) in enumerate(data_loader):
        print(f"Batch {batch_idx}")
        print(f"Gap size: {gap_int_s.shape}")
        print(f"Gap interval: start={gap_int_s[0, 0]:.3f} s, end={gap_int_s[0, 1]:.3f} s, length={gap_int_s[0, 1] - gap_int_s[0, 0]:.3f} s")
        print(f"Spectrogram gap shape: {spectrogram_gap.shape}")                    # Should be (batch_size, max_length, channels)
        print(f"Spectrogram orig shape: {spectrogram_target.shape}")
        print(f"Gap mask shape: {gap_mask.shape}")

        # Visualize the histograms
        import matplotlib.pyplot as plt
        fig1 = utils.visualize_spectrogram(spectrogram_target[0], title="Original Audio Spectrogram")
        fig2 = utils.visualize_spectrogram(spectrogram_gap[0],  gap_int=tuple(gap_int_s[0]), title="Gap Audio Spectrogram")

        # Plot the gap mask as a heatmap
        plt.figure(figsize=(10, 4))
        plt.title("Gap Mask Heatmap")
        plt.imshow(gap_mask[0], aspect='auto', origin='lower', cmap='gray')
        plt.colorbar(label="Mask Value (0 or 1)")
        plt.xlabel("Sample Index")
        plt.ylabel("Frequency")

        plt.show()
        break  # Just load one batch for demo