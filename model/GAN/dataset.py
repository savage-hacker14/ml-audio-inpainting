import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Dict, Any
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from utils import load_audio, create_gap_mask, audio_to_spectrogram

class SpeechInpaintingDataset(Dataset):
    """
    PyTorch Dataset for loading LibriSpeech audio, creating gaps,
    and generating spectrograms/masks for the inpainting task.
    """
    def __init__(self, cfg: Dict[str, Any], dataset_type: str = 'train'):
        """
        Initialize the dataset.

        Args:
            cfg (Dict[str, Any]): Configuration dictionary.
            dataset_type (str): 'train', 'valid', or 'test'.
        """
        self.cfg = cfg
        self.data_cfg = cfg['data']
        self.spec_cfg = self.data_cfg['spectrogram']
        self.train_cfg = cfg['training']

        self.sample_rate = self.data_cfg['sample_rate']
        self.max_len_s = self.data_cfg['max_len_s']
        self.gap_len_s = self.data_cfg['gap_len_s']
        self.max_samples = int(self.sample_rate * self.max_len_s)


        # Determine dataset path
        if dataset_type == 'train':
            data_path_key = 'train_path'
        elif dataset_type == 'valid':
            data_path_key = 'valid_path'
        elif dataset_type == 'test':
            data_path_key = 'test_path'
        else:
            raise ValueError(f"Invalid dataset_type: {dataset_type}")

        self.root_path = Path(self.data_cfg['root_path'])
        self.dataset_dir = self.root_path / self.data_cfg[data_path_key]

        if not self.dataset_dir.exists():
             raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")

        # Find all audio files (assuming .flac for LibriSpeech)
        self.file_paths = list(self.dataset_dir.rglob('*.flac'))
        if not self.file_paths:
             raise FileNotFoundError(f"No .flac files found in {self.dataset_dir}")
        self.file_paths.sort()

        print(f"Found {len(self.file_paths)} files in {self.dataset_dir}")

    def __len__(self) -> int:
        """Return the number of audio files."""
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load an audio file, create a gap, compute spectrograms and mask.

        Args:
            idx (int): Index of the file to load.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                'original_magnitude': Magnitude spectrogram of original audio.
                'impaired_magnitude': Magnitude spectrogram of audio with gap.
                'mask': Binary mask (1 for valid, 0 for hole) in spectrogram domain.
                'original_phase': Phase spectrogram of original audio (for potential recon).
        """
        file_path = self.file_paths[idx]

        # 1. Load original audio (fixed length)
        original_audio, sr = load_audio(
            file_path,
            sample_rate=self.sample_rate,
            max_len=self.max_len_s
        )
        if sr != self.sample_rate:
             raise ValueError(f"Sample rate mismatch: expected {self.sample_rate}, got {sr}")
        if len(original_audio) != self.max_samples:
            if np.all(original_audio == 0):
                print(f"Warning: Got zero audio for {file_path}, possibly due to loading error. Skipping?")
                # Option 1: Return dummy data (might skew training if frequent)
                dummy_spec_shape = (self.spec_cfg['n_fft'] // 2 + 1, int(np.ceil(self.max_samples / self.spec_cfg['hop_length'])))
                return {
                    'original_magnitude': torch.zeros(dummy_spec_shape, dtype=torch.float32),
                    'impaired_magnitude': torch.zeros(dummy_spec_shape, dtype=torch.float32),
                    'mask': torch.ones(dummy_spec_shape, dtype=torch.float32), # Mask all as valid
                    'original_phase': torch.zeros(dummy_spec_shape, dtype=torch.float32),
                }
                # Option 2: Try loading next index (complex with map-style dataset)
                # return self.__getitem__((idx + 1) % len(self))
            else:
                 raise ValueError(f"Audio length mismatch: expected {self.max_samples}, got {len(original_audio)}")


        # 2. Create impaired audio with a gap
        time_domain_mask, (gap_start_sample, gap_end_sample) = create_gap_mask(
            len(original_audio),
            self.gap_len_s,
            self.sample_rate
        )
        impaired_audio = original_audio * time_domain_mask

        # 3. Compute Spectrograms (Magnitude and Phase)
        original_magnitude, original_phase = audio_to_spectrogram(
            original_audio,
            n_fft=self.spec_cfg['n_fft'],
            hop_length=self.spec_cfg['hop_length'],
            win_length=self.spec_cfg['win_length'],
            window=self.spec_cfg['window'],
            normalize=self.spec_cfg['normalize'],
            power=self.spec_cfg['power']
        )

        impaired_magnitude, _ = audio_to_spectrogram( # Don't need impaired phase
            impaired_audio,
            n_fft=self.spec_cfg['n_fft'],
            hop_length=self.spec_cfg['hop_length'],
            win_length=self.spec_cfg['win_length'],
            window=self.spec_cfg['window'],
            normalize=self.spec_cfg['normalize'],
            power=self.spec_cfg['power']
        )

        # 4. Create Spectrogram Mask
        # Convert sample indices to frame indices
        hop_length = self.spec_cfg['hop_length']
        # Ensure integer division behaves as expected (floor for start, ceil for end)
        gap_start_frame = gap_start_sample // hop_length
        # Use ceiling to ensure the frame covering the end sample is included
        gap_end_frame = int(np.ceil(gap_end_sample / hop_length))

        num_frames = original_magnitude.shape[1]
        # Clamp frame indices to valid range
        gap_start_frame = max(0, gap_start_frame)
        gap_end_frame = min(num_frames, gap_end_frame)


        # Mask is 1 for valid regions, 0 for the gap (hole)
        spec_mask = np.ones_like(original_magnitude, dtype=np.float32)
        if gap_end_frame > gap_start_frame: # Only apply mask if gap exists in time frames
            spec_mask[:, gap_start_frame:gap_end_frame] = 0

        # 5. Convert to Tensors
        original_magnitude_t = torch.from_numpy(original_magnitude).float()
        impaired_magnitude_t = torch.from_numpy(impaired_magnitude).float()
        mask_t = torch.from_numpy(spec_mask).float()
        original_phase_t = torch.from_numpy(original_phase).float() # Keep for recon demo

        # Add channel dimension (C, H, W) -> (1, Freq, Time)
        return {
            'original_magnitude': original_magnitude_t.unsqueeze(0),
            'impaired_magnitude': impaired_magnitude_t.unsqueeze(0),
            'mask': mask_t.unsqueeze(0),
            'original_phase': original_phase_t.unsqueeze(0) # Keep original phase
        }