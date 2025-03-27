# models.py
# CS 6140 Final Project: Audio Inpainting
#
# Define the Stack BiLSTM model as defined in the paper

# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class StackedBLSTMModel(nn.Module):
    def __init__(self, config, dropout_rate, input_type='a', is_training=True):
        super(StackedBLSTMModel, self).__init__()
        
        self.audio_feat_dim = config['audio_feat_dim']
        self.net_dim = config['net_dim']
        self.num_layers = len(self.net_dim)
        self.input_type = input_type
        self.is_training = is_training
        self.dropout_rate = dropout_rate
        
        # Bidirectional LSTM
        self.blstm = nn.LSTM(
            input_size=self.net_dim[0],
            hidden_size=self.net_dim[0],
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # Fully connected layer
        self.fc = nn.Linear(self.net_dim[-1] * 2, self.audio_feat_dim)

    def forward(self, net_inputs):
        rnn_outputs, _ = self.blstm(net_inputs)
        
        # Apply dropout
        if self.is_training:
            rnn_outputs = F.dropout(rnn_outputs, p=self.dropout_rate, training=True)
        
        # Fully connected layer
        logits = self.fc(rnn_outputs)
        return logits
    
    def reconstruct_audio(self, corrupted_spectrogram, gap_mask):
        """
        Reconstruct the audio from the corrupted spectrogram using the model
        and the gap mask (1 for gap, 0 for rest of audio)

        TODO: Consider doing reconstructions using raw audio sample data instead of spectrograms,
        perform iSTFT on the reconstructed gap spectrogram to get the full audio signal
        """
        reconstructed_full_spectrogram = self.forward(corrupted_spectrogram)
        reconstructed_gap_spectrogram  = reconstructed_full_spectrogram * gap_mask + corrupted_spectrogram
        
        return reconstructed_gap_spectrogram
    

    