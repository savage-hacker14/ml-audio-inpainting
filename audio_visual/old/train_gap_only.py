# train_v3.py
#
# Support training of BLSTM gap-only models

# Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import tqdm
import configparser

import sys
sys.path.append("..")
from config import LIBRISPEECH_ROOT

from models import *
from audio_visual.dataset import LibriSpeechDataset
import librosa

# Load config file
import yaml
with open('blstm.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = StackedBLSTMModelGapOnly(config, dropout_rate=0, device=device, is_training=True)
print(f"Device: {device}")
print(model)
model.to(device)

# Create data loader
N_FILES = config['n_files']
GAPS_PER_AUDIO = config['gaps_per_audio']
dataset = LibriSpeechDataset(root_dir=LIBRISPEECH_ROOT, 
                             n_files=N_FILES,
                             n_gaps_per_audio=GAPS_PER_AUDIO,
                             n_fft=config['n_fft'], 
                             hop_len=config['hop_length'],
                             win_len=config['hann_win_length'])
train_dataset, test_dataset = random_split(dataset, [config['p_train'], config['p_test']])
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

# Define loss function and optimizer
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Define number of train epochs
num_epochs = config['max_n_epochs']
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for batch_idx, (log_spectrogram_gaps, gap_ints_s, gap_masks, spectrogram_target_phases) in enumerate(pbar):
        # Squeeze first and second dim of all the tensors
        # TODO: Fix with batch size later
        log_spectrogram_gaps      = log_spectrogram_gaps.reshape(config['batch_size'] * GAPS_PER_AUDIO, log_spectrogram_gaps.shape[2], log_spectrogram_gaps.shape[3])
        gap_ints_s                = gap_ints_s.reshape(config['batch_size'] * GAPS_PER_AUDIO, 2)
        spectrogram_target_phases = spectrogram_target_phases.reshape(config['batch_size'] * GAPS_PER_AUDIO, spectrogram_target_phases.shape[2], spectrogram_target_phases.shape[3])

        # Put all tensors on the device
        log_spectrogram_gaps      = log_spectrogram_gaps.to(device)
        spectrogram_target_phases = spectrogram_target_phases.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # TODO: Spectrogram normalization

        # Reconstruct the missing gap (only the gap) audio using the model
        reconstructed_spectrogram_gaps = model(log_spectrogram_gaps)

        # Get only the gap portion of the true audio
        gap_start_idx = librosa.time_to_frames(gap_ints_s.numpy()[:, 0], sr=DEFAULT_SAMPLE_RATE, hop_length=config['hop_length'])
        gap_len_idx   = reconstructed_spectrogram_gaps.shape[2]
        #print(f"gap_start_idx: {gap_start_idx}, gap_end_idx: {gap_end_idx}")
        true_spectrogram_gaps = torch.zeros_like(reconstructed_spectrogram_gaps).to(device)
        for i in range(log_spectrogram_gaps.shape[0]):
            true_spectrogram_gaps[i] = torch.log10(torch.abs(spectrogram_target_phases[i, :, gap_start_idx[i]:(gap_start_idx[i] + gap_len_idx)]))

        # Compute L1 loss
        # TODO: Make sure to convert true_spectrogram_gaps to log10 scale
        loss = criterion(reconstructed_spectrogram_gaps, true_spectrogram_gaps)

        # Backward pass & optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        pbar.set_description(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}], Loss: {loss.item():.4f}")

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

    # Save model every 5 epochs
    if (epoch % 3 == 0):
        torch.save(model.state_dict(), f"checkpoints/blstm_gap_only_2025_04_04_epoch_{epoch+1}.pt")

    # # Save model every epoch
    # torch.save(model.state_dict(), f"checkpoints/blstm_gap_only_2025_04_04_epoch_{epoch+1}.pt")

print("Training Complete!")