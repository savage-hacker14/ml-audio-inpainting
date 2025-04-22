# models.py
# CS 6140 Final Project: Audio Inpainting
#
# Defines the custom CNN + Bidirection LSTM Bottleneck Model Architecture

# Imports
import torch.nn as nn
import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    return cfg

class StackedBLSTMCNN(nn.Module):
    def __init__(self, config_path):
        # Call super constructor for nn.Module
        super(StackedBLSTMCNN, self).__init__()

        # Load in model config
        full_cfg         = load_config(config_path)
        mdl_cfg          = full_cfg['model']

        self.in_channels = mdl_cfg['in_channels']
        self.n_layers    = mdl_cfg['num_lstm_layers']
        self.hidden_dim  = mdl_cfg['lstm_hidden_dim']
        self.freq_bins   = full_cfg['data']['spectrogram']['n_fft'] // 2 + 1
        self.using_phase = self.in_channels == 2
        self.enc_filters = mdl_cfg['enc_filters']
        self.dec_filters = mdl_cfg['dec_filters']

        # Convolutional Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(self.in_channels, self.enc_filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.enc_filters[0]),
            nn.ReLU(),
            nn.Conv2d(self.enc_filters[0], self.enc_filters[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.enc_filters[1]),
            nn.ReLU(),
            nn.Conv2d(self.enc_filters[1], self.hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.hidden_dim // 2),
            nn.ReLU(),
        )
        # LSTM Bottleneck
        self.lstm = nn.LSTM(input_size=self.freq_bins * self.hidden_dim // 2, hidden_size=self.hidden_dim,
                            num_layers=self.n_layers, batch_first=True, bidirectional=True)

        # Projection Layer to Restore 2D Spectrogram
        self.projection = nn.Linear(self.hidden_dim * 2, self.freq_bins * self.dec_filters[0])  # To match decoder

        # Convolutional Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(self.dec_filters[0], self.dec_filters[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.dec_filters[1]),
            nn.ReLU(),
            nn.Conv2d(self.dec_filters[1], self.dec_filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.dec_filters[0]),
            nn.ReLU(),
            nn.Conv2d(self.dec_filters[0], self.in_channels, kernel_size=3, padding=1)  # Final spectrogram output
        )

    def forward(self, x):
        """
        x: (batch_size, in_channels, freq_bins, timeframes)
        """
        batch_size, _, freq_bins, timeframes = x.shape

        # CNN Encoder: Extract spatial features
        x = self.encoder(x)                                 # (batch_size, hidden_dim//2, freq_bins, timeframes)

        # Convert CNN encoder output to LSTM input shape
        x = x.permute(0, 3, 1, 2)                           # (batch, time, channels, freq)
        x = x.reshape(batch_size, timeframes, -1)           # (batch, time, channels * freq)

        # LSTM Bottleneck
        x, _ = self.lstm(x)                                 # (batch, time, hidden_dim * 2)

        # Project to match spectrogram size
        x = self.projection(x)                              # (batch, time, freq_bins * 16)
        # print(f"Projection output shape: {x.shape}")
        x = x.view(batch_size, timeframes, 16, freq_bins)   # Reshape to (batch, time, channels, freq_bins)
        x = x.permute(0, 2, 3, 1)                           # (batch, channels, freq_bins, timeframes)

        # CNN Decoder
        #print(f"Pre decoder: {x.shape}")
        x = self.decoder(x)                                 # (batch, 1, freq_bins, timeframes)
        x = x.squeeze(1)                                    # Remove channel dimension for final output

        return x
    
    def reconstruct_audio(self, log_spectrogram_gap, gap_mask):
        if (not self.using_phase):
            reconstructed_full_spectrogram = self(log_spectrogram_gap.unsqueeze(1))             # Add channel dimension for CNN encoder input
        else:
            # 2 or mroe input channels due to phase, no need to unsqueeze 
            reconstructed_full_spectrogram = self(log_spectrogram_gap)

        gap_mask = gap_mask.float()

        # Combine magnitude and phase channels
        if (self.using_phase):
            reconstructed_full_spectrogram = reconstructed_full_spectrogram[:, 0, :, :] + reconstructed_full_spectrogram[:, 1, :, :] * 1j
            log_spectrogram_gap            = log_spectrogram_gap[:, 0, :, :] + log_spectrogram_gap[:, 1, :, :] * 1j

        # NOTE: For phase model, the input spectrogram is NOT normalized thus neither is this output
        # Ensure gap is filled with the model output
        return reconstructed_full_spectrogram * gap_mask + log_spectrogram_gap * (1 - gap_mask)