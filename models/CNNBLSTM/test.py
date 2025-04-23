# train_v2.py
#
# Used to test blstm_simple models

# Imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import utils

from model import *
from dataset import LibriSpeechDataset

# Load config file
import yaml
config_path = 'cnn_blstm.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = StackedBLSTMCNN(config_path)
model.load_state_dict(torch.load('checkpoints/blstm_cnn_h128_2025_04_06_epoch_41.pt', weights_only=False))
print(model)
model.to(device)

# Load the dataset
spec_cfg       = config['data']['spectrogram']
N_FFT          = spec_cfg['n_fft']
HOP_LEN        = spec_cfg['hop_length']
WIN_LEN        = spec_cfg['win_length']
GAPS_PER_AUDIO = config['data']['gaps_per_audio']
BATCH_SIZE     = config['training']['batch_size']

dataset        = LibriSpeechDataset(config_path='cnn_blstm.yaml', dataset_type='test')
data_loader    = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
criterion      = nn.L1Loss()


# Evaluate the model
model.eval()
for batch_idx, (log_spectrogram_gaps, gap_ints_s, gap_masks, spectrogram_target_phases) in enumerate(data_loader):
    # Squeeze first and second dim of all the tensors
    # TODO: Fix with batch size later
    log_spectrogram_gaps      = log_spectrogram_gaps.reshape(BATCH_SIZE * GAPS_PER_AUDIO, log_spectrogram_gaps.shape[2], log_spectrogram_gaps.shape[3])
    gap_masks                 = gap_masks.reshape(BATCH_SIZE * GAPS_PER_AUDIO, gap_masks.shape[2], gap_masks.shape[3])
    gap_ints_s                = gap_ints_s.reshape(BATCH_SIZE * GAPS_PER_AUDIO, 2)
    spectrogram_target_phases = spectrogram_target_phases.reshape(BATCH_SIZE * GAPS_PER_AUDIO, spectrogram_target_phases.shape[2], spectrogram_target_phases.shape[3])

    # Put all tensors on the device
    log_spectrogram_gaps      = log_spectrogram_gaps.to(device)
    spectrogram_target_phases = spectrogram_target_phases.to(device)
    gap_masks                 = gap_masks.to(device)

    # Forward pass
    with torch.no_grad():
        spectrogram_reconstructed = model.reconstruct_spectrogram(log_spectrogram_gaps, gap_masks)
        spectrogram_full2 = model(log_spectrogram_gaps.unsqueeze(1))

    # Compute L1 loss
    temp = spectrogram_reconstructed * gap_masks
    print(f"min/max w gaps: {temp.min()}, {temp.max()}")
    loss = criterion(spectrogram_reconstructed * gap_masks, torch.abs(spectrogram_target_phases))

    print(f"Batch {batch_idx} - Loss: {loss.item()}")

    # Visualize the histograms
    spectrogram_target_sample  = spectrogram_target_phases[0].detach().cpu().numpy()
    spectrogram_out_sample     = spectrogram_reconstructed[0].detach().cpu().numpy()
    spectrogram_gap_sample     = log_spectrogram_gaps[0].detach().cpu().numpy()
    spectrogram_full2_sample   = spectrogram_full2[0].detach().cpu().numpy()

    print(f"min/max spectrogram_target: {spectrogram_target_sample.min()}, {spectrogram_target_sample.max()}")
    print(f"min/max spectrogram_full2: {spectrogram_full2_sample.min()}, {spectrogram_full2_sample.max()}")
    print(f"min/max spectrogram_reconstructed: {spectrogram_out_sample.min()}, {spectrogram_out_sample.max()}")

    # Save new audio file
    utils.save_audio(utils.spectrogram_to_audio(10 ** spectrogram_out_sample, phase_info=False, n_fft=N_FFT), f"output/reconstructed_audio_{batch_idx}.flac")
    utils.save_audio(utils.spectrogram_to_audio(spectrogram_target_sample, phase_info=True, n_fft=N_FFT), f"output/true_audio_{batch_idx}.flac")
    utils.save_audio(utils.spectrogram_to_audio(10 ** spectrogram_gap_sample, phase_info=False, n_fft=N_FFT), f"output/gap_audio_{batch_idx}.flac")

    fig1 = utils.visualize_spectrogram(abs(spectrogram_target_sample), in_db=False, power=1, title="Original Audio Spectrogram")
    fig2 = utils.visualize_spectrogram(10 ** spectrogram_gap_sample, in_db=False, power=1, gap_int=tuple(gap_ints_s[0]), title="Spectrogram with Gap (Red)")
    fig3 = utils.visualize_spectrogram(10 ** spectrogram_full2_sample, in_db=False, power=1,gap_int=tuple(gap_ints_s[0]), title="Full Inferenced Spectrogram 2")
    fig4 = utils.visualize_spectrogram(10 ** spectrogram_out_sample, gap_int=tuple(gap_ints_s[0]), in_db=False, power=1, title="Reconstructed Audio Spectrogram")
    plt.show()

    break  # Just load one batch for demo