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
from config import LIBRISPEECH_ROOT_PROCESSED
import utils

from models import StackedBLSTMModel
from dataloader import LibriSpeechDataset

# Load config file
import yaml
with open('blstm.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = StackedBLSTMModel(config, dropout_rate=0.3, is_training=True)
model.load_state_dict(torch.load('checkpoints/blstm_epoch_5_2025_03_24.pt', weights_only=False))
print(model)
model.to(device)

# Load the dataset
dataset = LibriSpeechDataset(LIBRISPEECH_ROOT_PROCESSED)
data_loader = DataLoader(dataset, batch_size=8, shuffle=True)
criterion = nn.L1Loss()
N_FFT = config['n_fft']

# Evaluate the model
model.eval()
for batch_idx, (spectrogram_gap, spectrogram_target) in enumerate(data_loader):
    spectrogram_gap = spectrogram_gap.to(device)
    spectrogram_target = spectrogram_target.to(device)

    # Forward pass
    outputs = model(spectrogram_gap)

    # Compute L1 loss
    loss = criterion(outputs, spectrogram_target)

    print(f"Batch {batch_idx} - Loss: {loss.item()}")

    # Visualize the histograms
    spectrogram_target_sample = spectrogram_target[0].detach().cpu().numpy()
    spectrogram_out_sample    = outputs[0].detach().cpu().numpy()
    fig1 = utils.visualize_spectrogram(spectrogram_target_sample, title="Original Audio Spectrogram")
    fig2 = utils.visualize_spectrogram(spectrogram_out_sample, title="Reconstructed Audio Spectrogram")
    plt.show()

    # Save new audio file
    utils.save_audio(utils.mel_spectrogram_to_audio(spectrogram_out_sample, n_fft=2048), f"output/reconstructed_audio_{batch_idx}.flac")
    #spectrogram_target_sample = spectrogram_target[0]
    utils.save_audio(utils.mel_spectrogram_to_audio(spectrogram_target_sample, n_fft=2048), f"output/true_audio_{batch_idx}.flac")

    break  # Just load one batch for demo