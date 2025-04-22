# train_v2.py
#
# Updated training script for dataloader_simple

# Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import tqdm
from datetime import datetime

import sys
sys.path.append("..")
from pathlib import Path
from config import LIBRISPEECH_ROOT

from models import *
from audio_visual.dataset import LibriSpeechDataset
import librosa

# Load config file
import yaml
config_path = 'cnn_blstm.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Create the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = StackedBLSTMCNN(config_path)

# Preload model weights if available
if (config['paths']['resume_mdl_path'] is not None):
    model.load_state_dict(torch.load(config['paths']['resume_mdl_path'], weights_only=False))

# Print model details and send to device
print(model)
model.to(device)

# Create data loader
N_FILES        = config['data']['n_files']
GAPS_PER_AUDIO = config['data']['gaps_per_audio']
BATCH_SIZE     = config['training']['batch_size']

dataset        = LibriSpeechDataset(config_path, dataset_type='train')
train_loader   = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Create logging directories
paths_cfg  = config['paths']
run_name   = datetime.today().strftime("%Y_%m_%d_%H%M")
tb_dir     = Path(paths_cfg['tensorboard_dir']) / run_name
chkpt_dir  = Path(paths_cfg['checkpoint_dir']) / run_name
sample_dir = Path(paths_cfg['sample_dir']) / run_name
log_dir    = Path(paths_cfg['log_dir'])
log_file   = log_dir / f"{run_name}.log"

tb_dir.mkdir(parents=True, exist_ok=True)
chkpt_dir.mkdir(parents=True, exist_ok=True)
sample_dir.mkdir(parents=True, exist_ok=True)
log_dir.mkdir(parents=True, exist_ok=True)


# Define loss function and optimizer
criterion = nn.L1Loss(reduction='sum')
if (config['training']['optimizer_type'] == "adam"):
    optimizer = optim.Adam(model.parameters(), lr=config['training']['starter_learning_rate'])

# Define number of train epochs
num_epochs = config['training']['max_n_epochs']
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    ## TRAINING
    for batch_idx, (log_spectrogram_gaps, gap_ints_s, gap_masks, spectrogram_target_phases) in enumerate(pbar):
        # Squeeze first and second dim of all the tensors
        log_spectrogram_gaps      = log_spectrogram_gaps.reshape(BATCH_SIZE * GAPS_PER_AUDIO, log_spectrogram_gaps.shape[2], log_spectrogram_gaps.shape[3])
        gap_masks                 = gap_masks.reshape(BATCH_SIZE * GAPS_PER_AUDIO, gap_masks.shape[2], gap_masks.shape[3])
        gap_ints_s                = gap_ints_s.reshape(BATCH_SIZE * GAPS_PER_AUDIO, 2)
        spectrogram_target_phases = spectrogram_target_phases.reshape(BATCH_SIZE * GAPS_PER_AUDIO, spectrogram_target_phases.shape[2], spectrogram_target_phases.shape[3])

        # Put all tensors on the device
        log_spectrogram_gaps      = log_spectrogram_gaps.to(device)
        spectrogram_target_phases = spectrogram_target_phases.to(device)
        gap_masks                 = gap_masks.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # TODO: Spectrogram normalization

        # Reconstruct the corrupated audio using the model
        log_spectrogram_reconstructed = model(log_spectrogram_gaps.unsqueeze(1))

        # Compute L1 loss on only the gap portions of the audio
        loss = criterion((10 ** log_spectrogram_reconstructed) * gap_masks, torch.abs(spectrogram_target_phases) * gap_masks)        # Remove phase info from spectrogram_target_phase

        # Backward pass & optimization
        loss.backward()
        optimizer.step()

        # Store running loss
        running_loss += loss.item()
        
        # Update progress bar
        pbar.set_description(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}], Loss: {loss.item():.4f}")

    # Compute training average loss
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

    # Save model every `checkpoint_interval` epochs
    if ((epoch + 1) % config['logging']['checkpoint_interval'] == 0):
        mdl_path = Path(config['paths']['checkpoint_dir']) / f"blstm_cnn_epoch_{epoch+1}.pt"
        torch.save(model.state_dict(), mdl_path)

print("Training Complete!")