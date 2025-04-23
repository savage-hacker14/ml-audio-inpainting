# Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import tqdm
import configparser

import sys
sys.path.append("..")
sys.path.append("../..")
from config import LIBRISPEECH_ROOT

from models_OLD import StackedBLSTMModel, StackedNormBLSTMModel
from dataloader import LibriSpeechDataset
import librosa

# Load config file
import yaml
with open('blstm.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = StackedBLSTMModel(config, dropout_rate=0, device=device, is_training=True)
model = StackedNormBLSTMModel(config, dropout_rate=0, device=device, is_training=True)
print(model)
model.to(device)

# Create data loader
dataset = LibriSpeechDataset(root_dir=LIBRISPEECH_ROOT, 
                             n_fft=config['n_fft'], 
                             hop_len=config['hop_length'],
                             win_len=config['hann_win_length'])
train_dataset, test_dataset = random_split(dataset, [config['p_train'], config['p_test']])
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Define loss function and optimizer
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Define number of train epochs
num_epochs = config['max_n_epochs']
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for batch_idx, (log_spectrogram_gap, gap_int_s, gap_mask, spectrogram_target_phase) in enumerate(pbar):
        # Put all tensors on the device
        log_spectrogram_gap      = log_spectrogram_gap.to(device)
        spectrogram_target_phase = spectrogram_target_phase.to(device)
        gap_mask                 = gap_mask.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # TODO: Spectrogram normalization
        # mu    = torch.mean(log_spectrogram_gap)
        # sigma = torch.std(log_spectrogram_gap)
        # log_spectrogram_gap_norm = (log_spectrogram_gap - mu) / sigma

        # Reconstruct the corrupated audio using the model
        spectrogram_reconstructed = model.reconstruct_audio(log_spectrogram_gap, gap_mask)
        #spectrogram_reconstructed = (spectrogram_reconstructed * sigma) + mu
        # spectrogram_reconstructed = model(spectrogram_gap)
        #print(f"Gradient check on spectrogram_reconstructed: {spectrogram_reconstructed.grad_fn}")         # Should NOT be None

        # Compute L1 loss
        # TODO: Consider only calculating loss based on the infilled gap (but not via masking, simple as a sub-array) - Paper does NOT do it this way
        #spectrogram_reconstructed_gap        = spectrogram_reconstructed * gap_mask
        #spectrogram_target_phase_gap         = spectrogram_target_phase * gap_mask
        #spectrogram_target_db = torch.from_numpy(librosa.power_to_db(spectrogram_target.detach().cpu().numpy())).to(device) # Convert target spectrogram to a power spectrogram in dB scale
        # Un-normaliza
        loss = criterion(spectrogram_reconstructed, torch.abs(spectrogram_target_phase))        # Remove phase info from spectrogram_target_phase

        # Backward pass & optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # # TEMP: After 100 batches break
        # if (batch_idx + 1 == 200):
        #     torch.save(model.state_dict(), f"checkpoints/blstm_norm_epoch_{epoch+1}_2025_03_27_v2.pt")
        #     exit()

        pbar.set_description(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}], Loss: {loss.item():.4f}")

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

    # Save model every 5 epochs
    if (epoch % 5 == 0):
        torch.save(model.state_dict(), f"checkpoints/blstm_2025_03_28_v4_epoch_{epoch+1}.pt")

print("Training Complete!")