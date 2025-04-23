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

from models_OLD import StackedBLSTMModel, StackedNormBLSTMModel
from dataloader import LibriSpeechDataset

# Load config file
import yaml
with open('blstm.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = StackedNormBLSTMModel(config, dropout_rate=0, is_training=False)
model.load_state_dict(torch.load('../checkpoints/blstm_2025_03_28_v4_epoch_36.pt', weights_only=False))
print(model)
model.to(device)

# Load the dataset
N_FFT    = config['n_fft']
HOP_LEN  = config['hop_length']
WIN_LEN  = config['hann_win_length']
N_EPOCHS = config['max_n_epochs']
dataset = LibriSpeechDataset(root_dir=LIBRISPEECH_ROOT, 
                             n_fft=N_FFT, 
                             hop_len=HOP_LEN,
                             win_len=WIN_LEN)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
criterion = nn.L1Loss()


# Evaluate the model
model.eval()
for batch_idx, (spectrogram_gap, gap_int_s, gap_mask, spectrogram_target_phase) in enumerate(data_loader):
    spectrogram_gap          = spectrogram_gap.to(device)
    gap_mask                 = gap_mask.to(device)
    spectrogram_target_phase = spectrogram_target_phase.to(device)

    # Forward pass
    with torch.no_grad():
        spectrogram_reconstructed = model.reconstruct_audio(spectrogram_gap, gap_mask)
        spectrogram_full2 = model(spectrogram_gap)
    #assert (spectrogram_full1 == spectrogram_full2).all()

    # Compute L1 loss
    loss = criterion(spectrogram_reconstructed, torch.abs(spectrogram_target_phase))

    print(f"Batch {batch_idx} - Loss: {loss.item()}")

    # Visualize the histograms
    spectrogram_target_sample  = spectrogram_target_phase[0].detach().cpu().numpy()
    #spectrogram_out_sample     = (spectrogram_reconstructed[0] * gap_mask[0]).detach().cpu().numpy()
    spectrogram_out_sample     = spectrogram_reconstructed[0].detach().cpu().numpy()
    spectrogram_gap_sample     = spectrogram_gap[0].detach().cpu().numpy()
    #spectrogram_full1_sample   = spectrogram_full1[0].detach().cpu().numpy()
    #spectrogram_full2_sample   = (spectrogram_full2[0] * gap_mask[0]).detach().cpu().numpy()
    spectrogram_full2_sample   = spectrogram_full2[0].detach().cpu().numpy()

    print(f"min/max spectrogram_target: {spectrogram_target_sample.min()}, {spectrogram_target_sample.max()}")
    print(f"min/max spectrogram_full2: {spectrogram_full2_sample.min()}, {spectrogram_full2_sample.max()}")
    print(f"min/max spectrogram_reconstructed: {spectrogram_out_sample.min()}, {spectrogram_out_sample.max()}")

    # Save new audio file
    utils.save_audio(utils.spectrogram_to_audio(spectrogram_out_sample, phase_info=False, n_fft=N_FFT), f"output/reconstructed_audio_{batch_idx}.flac")
    utils.save_audio(utils.spectrogram_to_audio(spectrogram_target_sample, phase_info=True, n_fft=N_FFT), f"output/true_audio_{batch_idx}.flac")
    utils.save_audio(utils.spectrogram_to_audio(10 ** spectrogram_gap_sample, phase_info=False, n_fft=N_FFT), f"output/gap_audio_{batch_idx}.flac")

    fig1 = utils.visualize_spectrogram(abs(spectrogram_target_sample), in_db=False, power=1, title="Original Audio Spectrogram")
    fig2 = utils.visualize_spectrogram(10 ** spectrogram_gap_sample, in_db=False, power=1, gap_int=tuple(gap_int_s[0]), title="Spectrogram with Gap (Red)")
    #fig3 = utils.visualize_spectrogram(spectrogram_full1_sample, gap_int=tuple(gap_int_s[0]), title="Full Inferenced Spectrogram 1")
    fig4 = utils.visualize_spectrogram(10 ** spectrogram_full2_sample, in_db=False, power=1,gap_int=tuple(gap_int_s[0]), title="Full Inferenced Spectrogram 2")
    fig5 = utils.visualize_spectrogram(spectrogram_out_sample, gap_int=tuple(gap_int_s[0]), in_db=False, power=1, title="Reconstructed Audio Spectrogram")
    plt.show()

    break  # Just load one batch for demo