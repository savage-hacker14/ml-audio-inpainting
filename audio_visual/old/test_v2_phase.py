# train_v2_phase.py
#
# Used to test blstm_cnn_phase models

# Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import tqdm
import configparser
import matplotlib.pyplot as plt

import sys
sys.path.append("..")
sys.path.append("../..")
from config import LIBRISPEECH_ROOT
import utils
import librosa

from models import *
from dataloader_phase import LibriSpeechDataset

# Load config file
import yaml
with open('../blstm.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = StackedBLSTMModel(config, dropout_rate=0, is_training=False)
IN_CHANNELS = 2
model = StackedBLSTMCNN(IN_CHANNELS, 128, 3)
model.load_state_dict(torch.load('../checkpoints/blstm_cnn_phase_h128_2025_04_14_epoch_26.pt', weights_only=False))
print(model)
model.to(device)

# Load the dataset
N_FFT    = config['n_fft']
HOP_LEN  = config['hop_length']
WIN_LEN  = config['hann_win_length']
N_EPOCHS = config['max_n_epochs']
N_FILES = config['n_files']
GAPS_PER_AUDIO = config['gaps_per_audio']
dataset = LibriSpeechDataset(root_dir=LIBRISPEECH_ROOT, 
                             n_files=N_FILES,
                             n_gaps_per_audio=GAPS_PER_AUDIO,
                             n_fft=config['n_fft'], 
                             hop_len=config['hop_length'],
                             win_len=config['hann_win_length'])
torch.manual_seed(123)        # To ensure same test/train split every time 
train_dataset, test_dataset = random_split(dataset, [config['p_train'], config['p_test']])
data_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True)
criterion = nn.L1Loss()


# Evaluate the model
model.eval()
for batch_idx, (spectrogram_gaps, gap_ints_s, gap_masks, spectrogram_target_phases) in enumerate(data_loader):
    # Squeeze first and second dim of all the tensors
    # TODO: Fix with batch size later
    spectrogram_gaps          = spectrogram_gaps.reshape(config['batch_size'] * GAPS_PER_AUDIO, IN_CHANNELS, spectrogram_gaps.shape[3], spectrogram_gaps.shape[4])
    gap_masks                 = gap_masks.reshape(config['batch_size'] * GAPS_PER_AUDIO, gap_masks.shape[2], gap_masks.shape[3])
    gap_ints_s                = gap_ints_s.reshape(config['batch_size'] * GAPS_PER_AUDIO, 2)
    spectrogram_target_phases = spectrogram_target_phases.reshape(config['batch_size'] * GAPS_PER_AUDIO, spectrogram_target_phases.shape[2], spectrogram_target_phases.shape[3])

    # Put all tensors on the device
    spectrogram_gaps          = spectrogram_gaps.to(device)
    spectrogram_target_phases = spectrogram_target_phases.to(device)
    gap_masks                 = gap_masks.to(device)

    # Forward pass
    with torch.no_grad():
        spectrogram_reconstructed = model.reconstruct_audio(spectrogram_gaps, gap_masks)
        spectrogram_full          = model(spectrogram_gaps)
        spectrogram_full_phases   = spectrogram_full[:, 0, :, :] + spectrogram_full[:, 1, :, :]  * 1j


    # Compute L1 loss
    loss = criterion(spectrogram_full_phases, spectrogram_target_phases)

    print(f"Batch {batch_idx} - Loss: {loss.item()}")

    # Convert sample spectorgram_gaps to complex matrix
    spectrogram_gap_sample     = spectrogram_gaps[0, 0, :, :] + spectrogram_gaps[0, 1, :, :] * 1j

    # Visualize the histograms
    spectrogram_target_sample  = spectrogram_target_phases[0].detach().cpu().numpy()
    spectrogram_out_sample     = spectrogram_reconstructed[0].detach().cpu().numpy()
    spectrogram_gap_sample     = spectrogram_gap_sample.detach().cpu().numpy()
    spectrogram_full_sample    = spectrogram_full_phases[0].detach().cpu().numpy()

    print(f"min/max spectrogram_target: {spectrogram_target_sample.min()}, {spectrogram_target_sample.max()}")
    print(f"min/max spectrogram_reconstructed: {spectrogram_out_sample.min()}, {spectrogram_out_sample.max()}")

    # Save new audio file
    utils.save_audio(utils.spectrogram_to_audio(spectrogram_out_sample, phase_info=True, n_fft=N_FFT), f"output/reconstructed_audio_{batch_idx}.flac")
    utils.save_audio(utils.spectrogram_to_audio(spectrogram_target_sample, phase_info=True, n_fft=N_FFT), f"output/true_audio_{batch_idx}.flac")
    utils.save_audio(utils.spectrogram_to_audio(spectrogram_gap_sample, phase_info=True, n_fft=N_FFT), f"output/gap_audio_{batch_idx}.flac")

    fig1 = utils.visualize_spectrogram(abs(spectrogram_target_sample), in_db=False, power=1, title="Original Audio Spectrogram")
    fig2 = utils.visualize_spectrogram(abs(spectrogram_gap_sample), in_db=False, power=1, gap_int=tuple(gap_ints_s[0]), title="Spectrogram with Gap (Red)")
    fig3 = utils.visualize_spectrogram(abs(spectrogram_full_sample), in_db=False, power=1,gap_int=tuple(gap_ints_s[0]), title="Full Inferenced Spectrogram")
    fig4 = utils.visualize_spectrogram(abs(spectrogram_out_sample), in_db=False, power=1, gap_int=tuple(gap_ints_s[0]), title="Reconstructed Audio Spectrogram")
    plt.show()

    break  # Just load one batch for demo