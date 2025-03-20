import os
from torch.utils.data import Dataset, DataLoader
import torchaudio

import sys
sys.path.append("..")
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
        # TODO: Consider switching this to our utils functions
        file_path = self.file_paths[idx]
        waveform, sample_rate = torchaudio.load(file_path)

        if self.transform:
            waveform = self.transform(waveform)

        return waveform, sample_rate

if __name__ == "__main__":
    dataset = LibriSpeechDataset(root_dir=LIBRISPEECH_ROOT_PROCESSED)

    data_loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        #collate_fn=collate_fn
    )

    # Usage Example
    for batch_idx, (waveforms, sample_rates) in enumerate(data_loader):
        print(f"Batch {batch_idx}")
        print(f"Waveforms shape: {waveforms.shape}")  # Should be (batch_size, max_length, channels)
        print(f"Sample rates: {sample_rates}")
        break  # Just load one batch for demo