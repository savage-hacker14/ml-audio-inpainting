# train_v3.py
#
# Used to test blstm_gap_only models

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
from config import LIBRISPEECH_ROOT, DEFAULT_SAMPLE_RATE
import utils
import librosa

from models_OLD import StackedBLSTMModelGapOnly
from audio_visual.old.dataloader import LibriSpeechDataset

# Load config file
import yaml
with open('blstm.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = StackedBLSTMModelGapOnly(config, dropout_rate=0, is_training=False)
model.load_state_dict(torch.load('../checkpoints/blstm_gap_only_2025_04_04_best_2.pt', weights_only=False))
print(f"Device: {device}")
print(model)
model.to(device)

# Load the dataset
N_FFT    = config['n_fft']
HOP_LEN  = config['hop_length']
WIN_LEN  = config['hann_win_length']
N_EPOCHS = config['max_n_epochs']
N_FILES  = config['n_files']
GAPS_PER_AUDIO = config['gaps_per_audio']

dataset = LibriSpeechDataset(root_dir=LIBRISPEECH_ROOT, 
                             n_files=N_FILES,
                             n_gaps_per_audio=GAPS_PER_AUDIO,
                             n_fft=config['n_fft'], 
                             hop_len=config['hop_length'],
                             win_len=config['hann_win_length'])
data_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
criterion = nn.L1Loss()


# Evaluate the model
model.eval()
for batch_idx, (log_spectrogram_gaps, gap_ints_s, gap_masks, spectrogram_target_phases) in enumerate(data_loader):
    # Squeeze first and second dim of all the tensors
    # TODO: Fix with batch size later
    log_spectrogram_w_gaps    = log_spectrogram_gaps.reshape(config['batch_size'] * GAPS_PER_AUDIO, log_spectrogram_gaps.shape[2], log_spectrogram_gaps.shape[3])
    gap_ints_s                = gap_ints_s.reshape(config['batch_size'] * GAPS_PER_AUDIO, 2)
    spectrogram_target_phases = spectrogram_target_phases.reshape(config['batch_size'] * GAPS_PER_AUDIO, spectrogram_target_phases.shape[2], spectrogram_target_phases.shape[3])

    # Put all tensors on the device
    log_spectrogram_w_gaps     = log_spectrogram_w_gaps.to(device)
    spectrogram_target_phases  = spectrogram_target_phases.to(device)
    gap_masks                  = gap_masks.to(device)

    # Compute missing gaps from full spectrogram with gaps
    with torch.no_grad():
        log_spectrogram_gaps = model(log_spectrogram_w_gaps)

    # # Compute L1 loss
    # print(f"min/max w gaps: {log_spectrogram_gaps.min()}, {log_spectrogram_gaps.max()}")
    # loss = criterion(log_spectrogram_gaps, torch.log10(torch.abs(spectrogram_target_phases)))

    # Put gaps back into the full spectrogram
    # TODO: Add gaps at the RIGHT position in the audio
    reconstructed_spectrograms = spectrogram_target_phases.detach().clone()
    gap_start_idx = librosa.time_to_frames(gap_ints_s.numpy()[:, 0], sr=DEFAULT_SAMPLE_RATE, hop_length=config['hop_length'])
    gap_len_idx   = log_spectrogram_gaps.shape[2]
    for i in range(spectrogram_target_phases.shape[0]):
        reconstructed_spectrograms[i, :,  gap_start_idx[i]:(gap_start_idx[i] + gap_len_idx)] = 10 ** log_spectrogram_gaps[i]

    # Visualize the histograms
    spectrogram_target_sample     = torch.abs(spectrogram_target_phases[0]).detach().cpu().numpy()
    spectrogram_gap_sample        = (10 ** log_spectrogram_w_gaps[0]).detach().cpu().numpy()
    spectrogram_out_sample        = reconstructed_spectrograms[0].detach().cpu().numpy()
    filled_spectrogram_gap_sample = (10 ** log_spectrogram_gaps[0]).detach().cpu().numpy()

    # Save new audio file
    utils.save_audio(utils.spectrogram_to_audio(spectrogram_out_sample, phase_info=False, n_fft=N_FFT), f"output/reconstructed_audio_{batch_idx}.flac")
    utils.save_audio(utils.spectrogram_to_audio(spectrogram_target_sample, phase_info=True, n_fft=N_FFT), f"output/true_audio_{batch_idx}.flac")
    utils.save_audio(utils.spectrogram_to_audio(filled_spectrogram_gap_sample, phase_info=False, n_fft=N_FFT), f"output/gap_audio_{batch_idx}.flac")

    fig1 = utils.visualize_spectrogram(spectrogram_target_sample, in_db=False, power=1, title="Original Audio Spectrogram")
    fig2 = utils.visualize_spectrogram(spectrogram_gap_sample, in_db=False, power=1, gap_int=tuple(gap_ints_s[0]), title="Spectrogram with Gap (Red)")
    fig3 = utils.visualize_spectrogram(filled_spectrogram_gap_sample, in_db=False, power=1,gap_int=tuple(gap_ints_s[0]), title="Inferred Gap Spectrogram")
    fig4 = utils.visualize_spectrogram(spectrogram_out_sample, gap_int=tuple(gap_ints_s[0]), in_db=False, power=1, title="Reconstructed Audio Spectrogram")
    plt.show()

    break  # Just load one batch for demo