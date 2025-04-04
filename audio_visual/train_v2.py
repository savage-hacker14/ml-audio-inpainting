# train_v2.py
#
# Updated training script for dataloader_simple

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
from dataloader_simple import LibriSpeechDataset
import librosa

# Load config file
import yaml
with open('blstm.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = StackedBLSTMModel(config, dropout_rate=0, device=device, is_training=True)
#model = StackedNormBLSTMModel(config, dropout_rate=0, device=device, is_training=True)
model = StackedBLSTMModelGapOnly(config, dropout_rate=0, device=device, is_training=True)
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
criterion = nn.L1Loss(reduction='sum')
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
        gap_masks                 = gap_masks.reshape(config['batch_size'] * GAPS_PER_AUDIO, gap_masks.shape[2], gap_masks.shape[3])
        spectrogram_target_phases = spectrogram_target_phases.reshape(config['batch_size'] * GAPS_PER_AUDIO, spectrogram_target_phases.shape[2], spectrogram_target_phases.shape[3])

        # Put all tensors on the device
        log_spectrogram_gaps      = log_spectrogram_gaps.to(device)
        spectrogram_target_phases = spectrogram_target_phases.to(device)
        gap_masks                 = gap_masks.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # TODO: Spectrogram normalization

        # Reconstruct the corrupated audio using the model
        spectrograms_reconstructed = model.reconstruct_audio(log_spectrogram_gaps, gap_masks)

        # Compute L1 loss
        loss = criterion(spectrograms_reconstructed * gap_masks, torch.abs(spectrogram_target_phases) * gap_masks)        # Remove phase info from spectrogram_target_phase

        # Backward pass & optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        pbar.set_description(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}], Loss: {loss.item():.4f}")

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

    # # Save model every 5 epochs
    # if (epoch % 5 == 0):
    #     torch.save(model.state_dict(), f"checkpoints/blstm_simple_2025_04_01_epoch_{epoch+1}.pt")

    # Save model every epoch
    torch.save(model.state_dict(), f"checkpoints/blstm_gap_only_2025_04_04_epoch_{epoch+1}.pt")

print("Training Complete!")