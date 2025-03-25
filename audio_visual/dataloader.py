import os
import torch
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append("..")
import utils
from config import LIBRISPEECH_ROOT_PROCESSED

class LibriSpeechDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
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
        audio_data_gap, gap_int = utils.add_random_gap(file_path, 0.2)

        spectrogram_target = utils.extract_mel_spectrogram(audio_data, n_fft=2048, sample_rate=sample_rate) #torchaudio.load(file_path)
        spectrogram_gap    = utils.extract_mel_spectrogram(audio_data_gap, n_fft=2048, sample_rate=sample_rate)
        #print(f"Spectrogram size: {spectrogram.shape}, audio_data size: {audio_data.shape}")
        #print(f"Gap inserted at t = {gap_int[0]} s with length {gap_int[1] - gap_int[0]} s")

        return torch.tensor(spectrogram_gap, dtype=torch.float32), torch.tensor(spectrogram_target, dtype=torch.float32)

if __name__ == "__main__":
    dataset = LibriSpeechDataset(root_dir=LIBRISPEECH_ROOT_PROCESSED)

    data_loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        #collate_fn=collate_fn
    )

    # Usage Example
    for batch_idx, (spectrogram_gap, spectrogram_target) in enumerate(data_loader):
        print(f"Batch {batch_idx}")
        print(f"Spectrogram gap shape: {spectrogram_gap.shape}")  # Should be (batch_size, max_length, channels)
        print(f"Spectrogram orig shape: {spectrogram_target.shape}")

        # Visualize the histograms
        import matplotlib.pyplot as plt
        fig1 = utils.visualize_spectrogram(spectrogram_target[0], title="Original Audio Spectrogram")
        fig2 = utils.visualize_spectrogram(spectrogram_gap[0], title="Gap Audio Spectrogram")
        plt.show()
        break  # Just load one batch for demo