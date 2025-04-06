# models.py
# CS 6140 Final Project: Audio Inpainting
#
# Define the Stack BiLSTM model as defined in the paper

# Imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import librosa

from config import DEFAULT_SAMPLE_RATE

class StackedBLSTMModel(nn.Module):
    def __init__(self, config, dropout_rate, input_type='a', is_training=True, device="cpu"):
        super(StackedBLSTMModel, self).__init__()
        
        self.audio_feat_dim = config['audio_feat_dim']
        self.net_dim = config['net_dim']
        self.num_layers = len(self.net_dim)
        self.input_type = input_type
        self.is_training = is_training
        self.dropout_rate = dropout_rate
        self.device = device
        
        # Bidirectional LSTM
        self.blstm = nn.LSTM(
            input_size=self.net_dim[0],
            hidden_size=self.net_dim[1],
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # Fully connected layer
        self.fc = nn.Linear(self.net_dim[2] * 2, self.audio_feat_dim)

    def forward(self, net_inputs):
        lstm_outputs, _ = self.blstm(net_inputs)
        
        # Apply dropout
        if self.is_training:
            lstm_outputs = F.dropout(lstm_outputs, p=self.dropout_rate, training=True)
        
        # Fully connected layer
        logits = self.fc(lstm_outputs)
        return logits
    
    def reconstruct_audio(self, log_spectrogram_gap, gap_mask):
        """
        Reconstruct the audio from the corrupted spectrogram using the model
        and the gap mask (1 for gap, 0 for rest of audio)

        TODO: Consider doing reconstructions using raw audio sample data instead of spectrograms,
        perform iSTFT on the reconstructed gap spectrogram to get the full audio signal

        Parameters
        ----------
        log_spectrogram_gap (torch.tensor): Log magnitude spectrogram with gap
        gap_mask (torch.Tensor):            Binary mask indicating gap location (1 for gap, 0 for rest/true audio)

        Returns
        -------
        reconstructed_gap_spectrogram (torch.tensor): Spectrogram (power spectrogram in dB scale) with gap filled, done by
        copying the non-gap values from the original spectrogram and filling in the the gap with the model output
        """

        reconstructed_full_spectrogram = self(log_spectrogram_gap)             # Has grad
        gap_mask = gap_mask.float()             # Ensure gap mask is a float tensor for gradient flow
        #print(f"reconstructed_full_spectrogram size: {reconstructed_full_spectrogram.size()}, mask size: {gap_mask.size()}")

        # Convert spectrogram to dB before masking - Otherwise when visualizing the spectrograms, the amplitudes close to zero will vanish with -inf dB
        # reconstructed_full_spectrogram_db = torch.from_numpy(librosa.amplitude_to_db(reconstructed_full_spectrogram.detach().cpu().numpy(), ref=np.max)).to("cuda")
        # corrupted_spectrogram_db          = torch.from_numpy(librosa.amplitude_to_db(corrupted_spectrogram.detach().cpu().numpy(), ref=np.max)).to("cuda")
        # reconstructed_gap_spectrogram = reconstructed_full_spectrogram_db * gap_mask + corrupted_spectrogram_db * (1 - gap_mask)  # Ensure gap is filled with the model output    
        
        reconstructed_gap_spectrogram = reconstructed_full_spectrogram * gap_mask + log_spectrogram_gap * (1 - gap_mask)  # Ensure gap is filled with the model output    
        #reconstructed_gap_spectrogram = torch.tensor(librosa.db_to_power(reconstructed_gap_spectrogram.detach().cpu().numpy())).to(self.device)  # Convert back to power spectrogram

        return 10 ** reconstructed_gap_spectrogram          # log magnitude --> magnitude
    

#################################################################################################

class StackedBLSTMModelGapOnly(nn.Module):
    def __init__(self, config, dropout_rate, input_type='a', is_training=True, device="cpu"):
        super(StackedBLSTMModelGapOnly, self).__init__()
        
        self.audio_feat_dim = config['audio_feat_dim']
        self.gap_feat_dim = math.ceil(self.audio_feat_dim * (0.2 / 5.0)) # Add these as params to config later (change to math.ceil)
        self.net_dim = config['net_dim']
        self.hop_len = config['hop_length']
        self.num_layers = len(self.net_dim)
        self.input_type = input_type
        self.is_training = is_training
        self.dropout_rate = dropout_rate
        self.device = device
        
        # Bidirectional LSTM
        self.blstm = nn.LSTM(
            input_size=self.net_dim[0],
            hidden_size=self.net_dim[1],
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # Fully connected layer
        self.fc1 = nn.Linear(self.net_dim[2] * 2, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, self.gap_feat_dim)

    def forward(self, net_inputs):
        lstm_outputs, _ = self.blstm(net_inputs)
        
        # Apply dropout
        if self.is_training:
            lstm_outputs = F.dropout(lstm_outputs, p=self.dropout_rate, training=True)
        
        # Fully connected layers
        x = F.relu(self.fc1(lstm_outputs))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#################################################################################################

class StackedNormBLSTMModel(nn.Module):
    def __init__(self, config, dropout_rate, input_type='a', is_training=True, device="cpu"):
        super(StackedNormBLSTMModel, self).__init__()
        
        self.audio_feat_dim = config['audio_feat_dim']
        self.net_dim = config['net_dim']
        self.num_layers = len(self.net_dim)
        self.input_type = input_type
        self.is_training = is_training
        self.dropout_rate = dropout_rate
        self.device = device
        
        # Bidirectional LSTM
        # self.blstm = nn.LSTM(
        #     input_size=self.net_dim[0],
        #     hidden_size=self.net_dim[1],
        #     num_layers=1,
        #     batch_first=True,
        #     bidirectional=True
        # )
        
        self.lstm_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        for i in range(self.num_layers):
            self.lstm_layers.append(nn.LSTM(
                input_size=self.audio_feat_dim if i == 0 else self.net_dim[i] * 2, 
                hidden_size=self.net_dim[i], 
                num_layers=1, 
                batch_first=True, 
                bidirectional=True
            ))
            self.norm_layers.append(nn.LayerNorm(self.net_dim[i] * 2))  # Normalize both directions

        # Fully connected layer
        self.fc = nn.Linear(self.net_dim[2] * 2, self.audio_feat_dim)

    def forward(self, x):
        lstm_outputs = x
        for i in range(self.num_layers):
            lstm_outputs, _ = self.lstm_layers[i](lstm_outputs)
            lstm_outputs = self.norm_layers[i](lstm_outputs)
        
        # # Apply dropout
        # if self.is_training:
        #     lstm_outputs = F.dropout(lstm_outputs, p=self.dropout_rate, training=True)
        
        # Fully connected layer
        logits = self.fc(lstm_outputs)
        return logits
    
    def reconstruct_audio(self, log_spectrogram_gap, gap_mask):
        """
        Reconstruct the audio from the corrupted spectrogram using the model
        and the gap mask (1 for gap, 0 for rest of audio)

        TODO: Consider doing reconstructions using raw audio sample data instead of spectrograms,
        perform iSTFT on the reconstructed gap spectrogram to get the full audio signal

        Parameters
        ----------
        log_spectrogram_gap (torch.tensor): Log magnitude spectrogram with gap
        gap_mask (torch.Tensor):            Binary mask indicating gap location (1 for gap, 0 for rest/true audio)

        Returns
        -------
        reconstructed_gap_spectrogram (torch.tensor): Spectrogram (power spectrogram in dB scale) with gap filled, done by
        copying the non-gap values from the original spectrogram and filling in the the gap with the model output
        """

        reconstructed_full_spectrogram = self(log_spectrogram_gap)             # Has grad
        gap_mask = gap_mask.float()             # Ensure gap mask is a float tensor for gradient flow
        #print(f"reconstructed_full_spectrogram size: {reconstructed_full_spectrogram.size()}, mask size: {gap_mask.size()}")

        # Convert spectrogram to dB before masking - Otherwise when visualizing the spectrograms, the amplitudes close to zero will vanish with -inf dB
        # reconstructed_full_spectrogram_db = torch.from_numpy(librosa.amplitude_to_db(reconstructed_full_spectrogram.detach().cpu().numpy(), ref=np.max)).to("cuda")
        # corrupted_spectrogram_db          = torch.from_numpy(librosa.amplitude_to_db(corrupted_spectrogram.detach().cpu().numpy(), ref=np.max)).to("cuda")
        # reconstructed_gap_spectrogram = reconstructed_full_spectrogram_db * gap_mask + corrupted_spectrogram_db * (1 - gap_mask)  # Ensure gap is filled with the model output    
        
        reconstructed_gap_spectrogram = reconstructed_full_spectrogram * gap_mask + log_spectrogram_gap * (1 - gap_mask)  # Ensure gap is filled with the model output    
        #reconstructed_gap_spectrogram = torch.tensor(librosa.db_to_power(reconstructed_gap_spectrogram.detach().cpu().numpy())).to(self.device)  # Convert back to power spectrogram

        return 10 ** reconstructed_gap_spectrogram          # log magnitude --> magnitude
    
#################################################################################################

class UNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=64):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(in_channels, base_channels)
        self.enc2 = self.conv_block(base_channels, base_channels * 2)
        self.enc3 = self.conv_block(base_channels * 2, base_channels * 4)

        # Bottleneck
        self.bottleneck = self.conv_block(base_channels * 4, base_channels * 8)

        # Decoder
        self.dec3 = self.up_block(base_channels * 8, base_channels * 4)
        self.dec2 = self.up_block(base_channels * 4, base_channels * 2)
        self.dec1 = self.up_block(base_channels * 2, base_channels)

        # Final output
        self.final_conv = nn.Conv2d(base_channels, 1, kernel_size=1)

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

    def up_block(self, in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2),
            self.conv_block(out_c, out_c)
        )

    def forward(self, x):
        # x: (batch_size, 1, freq_bins, timeframes)

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))

        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e3, 2))

        # Decoder with skip connections
        d3 = self.dec3(b)
        d3 = d3 + e3  # skip connection

        d2 = self.dec2(d3)
        d2 = d2 + e2

        d1 = self.dec1(d2)
        d1 = d1 + e1

        out = self.final_conv(d1)
        return out
    
    def reconstruct_audio(self, log_spectrogram_gap, gap_mask):
        reconstructed_full_spectrogram = self(log_spectrogram_gap)             # Has grad
        gap_mask = gap_mask.float()             # Ensure gap mask is a float tensor for gradient float
        
        return reconstructed_full_spectrogram * gap_mask + log_spectrogram_gap * (1 - gap_mask)  # Ensure gap is filled with the model output    
    
#################################################################################################

class StackedBLSTMCNN(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=128, num_layers=2, freq_bins=257):
        super(StackedBLSTMCNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.freq_bins = freq_bins  # Explicit frequency bins

        # Convolutional Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(),
        )

        # Reduce frequency dimension (Global Pooling)
        #self.global_pool = nn.AdaptiveAvgPool2d((1, None))  # (batch, hidden_dim//2, 1, time)

        # LSTM Bottleneck
        self.lstm = nn.LSTM(input_size=freq_bins * hidden_dim // 2, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True, bidirectional=True)

        # Projection Layer to Restore 2D Spectrogram
        self.projection = nn.Linear(hidden_dim * 2, freq_bins * 16)  # 16 channels to match decoder

        # Convolutional Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, in_channels, kernel_size=3, padding=1)  # Final spectrogram output
        )

    def forward(self, x):
        """
        x: (batch_size, 1, freq_bins, timeframes)
        """
        batch_size, _, freq_bins, timeframes = x.shape

        # CNN Encoder: Extract spatial features
        x = self.encoder(x)  # (batch_size, hidden_dim//2, freq_bins, timeframes)

        # Reduce frequency dimension with global pooling
        # x = self.global_pool(x)  # (batch, hidden_dim//2, 1, time)
        # x = x.squeeze(2)  # (batch, hidden_dim//2, time)
        # x = x.permute(0, 2, 1)  # (batch, time, hidden_dim//2)

        # Convert CNN encoder output to LSTM input shape
        x = x.permute(0, 3, 1, 2)  # (batch, time, channels, freq)
        x = x.reshape(batch_size, timeframes, -1)  # (batch, time, channels * freq)

        # LSTM Bottleneck
        x, _ = self.lstm(x)  # (batch, time, hidden_dim * 2)

        # Project to match spectrogram size
        x = self.projection(x)  # (batch, time, freq_bins * 16)
        # print(f"Projection output shape: {x.shape}")
        x = x.view(batch_size, timeframes, 16, freq_bins)  # Reshape to (batch, time, channels, freq_bins)
        x = x.permute(0, 2, 3, 1)  # (batch, channels, freq_bins, timeframes)

        # CNN Decoder
        #print(f"Pre decoder: {x.shape}")
        x = self.decoder(x)  # (batch, 1, freq_bins, timeframes)
        x = x.squeeze(1)      # Remove channel dimension for final output

        return x
    
    def reconstruct_audio(self, log_spectrogram_gap, gap_mask):
        reconstructed_full_spectrogram = self(log_spectrogram_gap.unsqueeze(1))             # Add channel dimension for CNN encoder input
        #print(f"Reconstructed shape: {reconstructed_full_spectrogram.shape}")
        gap_mask = gap_mask.float()             # Ensure gap mask is a float tensor for gradient float
        
        return reconstructed_full_spectrogram * gap_mask + log_spectrogram_gap * (1 - gap_mask)  # Ensure gap is filled with the model output    