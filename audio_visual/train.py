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

from models import StackedBLSTMModel
from dataloader import LibriSpeechDataset

# Load config file
import yaml
with open('blstm.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create the model
model = StackedBLSTMModel(config, dropout_rate=0.3, is_training=True)
print(model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Create data loader
dataset = LibriSpeechDataset(LIBRISPEECH_ROOT)
train_dataset, test_dataset = random_split(dataset, [config['p_train'], config['p_test']])
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Define loss function and optimizer
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define number of train epochs
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for batch_idx, (spectrogram_gap, gap_int_s, gap_mask, spectrogram_target) in enumerate(pbar):
        # Put all tensors on the device
        spectrogram_gap    = spectrogram_gap.to(device)
        spectrogram_target = spectrogram_target.to(device)
        gap_mask           = gap_mask.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Reconstruct the corrupated audio using the model
        spectrogram_reconstructed = model.reconstruct_audio(spectrogram_gap, gap_mask)

        # Compute L1 loss
        # TODO: Consider only calculating loss based on the infilled gap
        spectrogram_reconstructed_gap = spectrogram_reconstructed * gap_mask
        spectrogram_target_gap        = spectrogram_target * gap_mask
        loss                          = criterion(spectrogram_reconstructed_gap, spectrogram_target_gap)

        # Backward pass & optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # if (batch_idx + 1) % 10 == 0:
        #     print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}], Loss: {loss.item():.4f}")
        pbar.set_description(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}], Loss: {loss.item():.4f}")

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")
    torch.save(model.state_dict(), f"checkpoints/blstm_epoch_{epoch+1}_2025_03_27.pt")

print("Training Complete!")