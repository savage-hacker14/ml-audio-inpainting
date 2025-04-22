# train_v2.py
#
# Updated training script for dataloader_simple

# Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

import sys
sys.path.append("..")
sys.path.append("../..")
from pathlib import Path

from models import *
from audio_visual.dataset import LibriSpeechDataset
import utils

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

train_dataset  = LibriSpeechDataset(config_path, dataset_type='train')
train_loader   = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataset   = LibriSpeechDataset(config_path, dataset_type='test')
test_loader    = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

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

# Create tensorboard
writer = SummaryWriter(log_dir=str(tb_dir))

# Define loss function and optimizer
criterion = nn.L1Loss(reduction='sum')
if (config['training']['optimizer_type'] == "adam"):
    optimizer = optim.Adam(model.parameters(), lr=config['training']['starter_learning_rate'])


# TRAINING LOOP
num_epochs = config['training']['max_n_epochs']
global_step = 0
for epoch in range(num_epochs):
    ## TRAINING
    model.train()
    running_loss = 0.0
    train_pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for batch_idx, (log_spectrogram_gaps, gap_ints_s, gap_masks, spectrogram_target_phases) in enumerate(train_pbar):
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
        train_pbar.set_description(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}], Loss: {loss.item():.4f}")

        # Write train loss to tensorboard every `metric_interval` batches
        if (global_step % config['logging']['metric_interval'] == 0):
            writer.add_scalar('Train_Loss', loss.item(), global_step)

        # Save sample spectrograms (original, gap, reconstructed) to tensorboard every `spectrogram_interval` batches
        if (global_step % config['logging']['spectrogram_interval'] == 0):
            orig_spectogram_sample     = spectrogram_target_phases[0].detach().cpu().numpy()
            gap_spectrogram_sample     = (10 ** log_spectrogram_gaps[0]).detach().cpu().numpy()
            reconst_spectrogram_sample = 10 ** model.reconstruct_audio(log_spectrogram_gaps, gap_masks)
            reconst_spectrogram_sample = reconst_spectrogram_sample[0].detach().cpu().numpy()
            gap_start_s                = gap_ints_s[0, 0].item()
            gap_end_s                  = gap_ints_s[0, 1].item()
            vis_kwargs = {
                'sample_rate': config['data']['sample_rate'],
                'hop_length': config['data']['spectrogram']['hop_length'],
                'in_db': False,
                'gap_int': (gap_start_s, gap_end_s)
            }
            # Create figures (function returns figure object)
            fig_orig = utils.visualize_spectrogram(orig_spectogram_sample, title="Original Spectogram", **vis_kwargs)
            fig_imp  = utils.visualize_spectrogram(gap_spectrogram_sample, title="Spectrogram w/ Gap", **vis_kwargs)
            fig_gen  = utils.visualize_spectrogram(reconst_spectrogram_sample, title="Reconstructed Spectrogram", **vis_kwargs)

            # Log figures to TensorBoard
            if fig_orig: writer.add_figure("Spectrograms/Original", fig_orig, global_step)
            if fig_imp: writer.add_figure("Spectrograms/Impaired", fig_imp, global_step)
            if fig_gen: writer.add_figure("Spectrograms/Generated", fig_gen, global_step)
            plt.close('all') # Close all generated figures

        # Increment batch step
        global_step += 1

    # Compute training average loss
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

    ## TESTING
    model.eval()
    test_pbar = tqdm.tqdm(test_loader, desc="Computing Test Loss")
    running_test_loss = 0.0
    for batch_idx, (log_spectrogram_gaps, gap_ints_s, gap_masks, spectrogram_target_phases) in enumerate(test_pbar):
        # Squeeze first and second dim of all the tensors
        log_spectrogram_gaps      = log_spectrogram_gaps.reshape(BATCH_SIZE * GAPS_PER_AUDIO, log_spectrogram_gaps.shape[2], log_spectrogram_gaps.shape[3])
        gap_masks                 = gap_masks.reshape(BATCH_SIZE * GAPS_PER_AUDIO, gap_masks.shape[2], gap_masks.shape[3])
        gap_ints_s                = gap_ints_s.reshape(BATCH_SIZE * GAPS_PER_AUDIO, 2)
        spectrogram_target_phases = spectrogram_target_phases.reshape(BATCH_SIZE * GAPS_PER_AUDIO, spectrogram_target_phases.shape[2], spectrogram_target_phases.shape[3])

        # Put all tensors on the device
        log_spectrogram_gaps      = log_spectrogram_gaps.to(device)
        spectrogram_target_phases = spectrogram_target_phases.to(device)
        gap_masks                 = gap_masks.to(device)

        with torch.no_grad():
            # Reconstruct the corrupated audio using the model
            log_spectrogram_reconstructed = model(log_spectrogram_gaps.unsqueeze(1))

            # Compute L1 loss on only the gap portions of the audio
            loss = criterion((10 ** log_spectrogram_reconstructed) * gap_masks, torch.abs(spectrogram_target_phases) * gap_masks)        # Remove phase info from spectrogram_target_phase

            running_test_loss += loss.item()

    # Write test loss to Tensorboard
    avg_test_loss = running_test_loss / len(test_loader)
    writer.add_scalar('Test_Loss', avg_test_loss, epoch + 1)

    # Save sample spectogram audio files to tensorboard

    # Save model every `checkpoint_interval` epochs
    if ((epoch + 1) % config['logging']['checkpoint_interval'] == 0):
        mdl_path = Path(config['paths']['checkpoint_dir']) / f"blstm_cnn_epoch_{epoch+1}.pt"
        torch.save(model.state_dict(), mdl_path)

print("Training Complete!")