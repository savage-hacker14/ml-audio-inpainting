# Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
import configparser

import sys
sys.path.append("..")
from config import LIBRISPEECH_ROOT_PROCESSED

from models import StackedBLSTMModel
from dataloader import LibriSpeechDataset

# Load config file
import yaml
with open('blstm.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create the model
model = StackedBLSTMModel(config, dropout_rate=0.3, is_training=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Create data loader
dataset = LibriSpeechDataset(LIBRISPEECH_ROOT_PROCESSED)
data_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Define loss function and optimizer
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define number of train epochs
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch_idx, (inputs, targets, lengths) in enumerate(data_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}], Loss: {loss.item():.4f}")

    print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {running_loss / len(data_loader):.4f}")

print("Training Complete!")