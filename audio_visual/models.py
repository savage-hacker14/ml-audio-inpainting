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