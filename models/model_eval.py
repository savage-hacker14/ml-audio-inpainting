# model_eval.py
# 
# This script generates the reconstructed audio data for a specific set of files, 
# which were also used in the auto-regressive model to allow for a direct
# model comparison

import os
import soundfile as sf
import torch
import numpy as np
import librosa

from pathlib import Path 
import sys

from GAN.train import load_config
import utils
sys.path.append(str(Path(__file__).resolve().parent.parent))

from GAN.networks import PConvUNet
from CNNBLSTM.model import StackedBLSTMCNN

def load_model(model_type, config_path, checkpoint_path, device):
    """Loads the specified model from a checkpoint."""
    print(f"Loading {model_type} model from {checkpoint_path}...")

    if model_type == 'gan':
        cfg = load_config(config_path)
        model = PConvUNet(
            input_channels=cfg['model']['generator']['input_channels'],
            mask_channels=cfg['model']['generator']['mask_channels'],
            output_channels=cfg['model']['generator']['output_channels'],
            enc_layer_cfg=cfg['model']['generator'].get('enc_layer_cfg', [ 
                (64, 7, 2, 3), (128, 5, 2, 2), (256, 5, 2, 2),
                (512, 3, 2, 1), (512, 3, 2, 1), (512, 3, 2, 1), (512, 3, 2, 1)
            ]),
        ).to(device)
    elif model_type == 'cnnlstm':
        model = StackedBLSTMCNN(config_path).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(torch.load(checkpoint_path, weights_only=False, map_location=device))
    model.eval()

    return model

def inpaint(model, config_path, audio_path, output_path, device):
    """Loads audio, applies mask, performs inpainting, and saves the result."""
    # Get model type
    if (isinstance(model, PConvUNet)):
        model_type = 'gan'
    elif (isinstance(model, StackedBLSTMCNN)):
        model_type = 'cnnlstm'
    else:
        raise ValueError("Unknown model type.")
    
    # Load configuration
    config = load_config(config_path)

    audio, sr = utils.load_audio(audio_path)
    print(f"Loaded {audio_path}, shape: {audio.shape}, sample rate: {sr}")

    gap_len_s = 0.08  # Example gap length in seconds

    time_domain_mask, (gap_start_sample, gap_end_sample) = utils.create_gap_mask(
        len(audio),
        gap_len_s,
        sr,
        gap_start_s=2.0,  # Example start time for the gap in seconds
    )
    
    impaired_audio = audio * time_domain_mask

    original_spectrogram = utils.extract_spectrogram(
        audio, 
        n_fft=config['data']['spectrogram']['n_fft'],
        hop_length=config['data']['spectrogram']['hop_length'], 
        win_length=config['data']['spectrogram']['win_length']
    )
    
    original_magnitude = np.abs(original_spectrogram)
    original_magnitude = np.log1p(original_magnitude)
    original_phase     = np.angle(original_spectrogram)
    original_phase_t = torch.from_numpy(original_phase).float()
    
    impaired_spectrogram = utils.extract_spectrogram(
        impaired_audio, 
        n_fft=config['data']['spectrogram']['n_fft'],
        hop_length=config['data']['spectrogram']['hop_length'],
        win_length=config['data']['spectrogram']['win_length']
    )
    
    impaired_magnitude = np.abs(impaired_spectrogram)
    impaired_magnitude = np.log1p(impaired_magnitude)
    
    if (model_type == 'gan'):
        hop_length = config['data']['spectrogram']['hop_length']
        gap_start_frame = gap_start_sample // hop_length
        gap_end_frame = int(np.ceil(gap_end_sample / hop_length))

        num_frames = original_magnitude.shape[1]
        
        gap_start_frame = max(0, gap_start_frame)
        gap_end_frame = min(num_frames, gap_end_frame)

        spec_mask = np.ones_like(original_magnitude, dtype=np.float32)
        if gap_end_frame > gap_start_frame: 
            spec_mask[:, gap_start_frame:gap_end_frame] = 0

        #original_magnitude_t = torch.from_numpy(original_magnitude).float()
        impaired_magnitude_t = torch.from_numpy(impaired_magnitude).float()
        mask_t = torch.from_numpy(spec_mask).float()
        
        with torch.no_grad():
            inpainted = model(impaired_magnitude_t, mask_t)
    elif (model_type == 'cnnlstm'):
        # Create spectrogram-size gap mask
        hop_length = config['data']['spectrogram']['hop_length']
        spec_mask = np.zeros_like(original_spectrogram, dtype=np.float32)
        gap_start_frame = librosa.time_to_frames(2.0, sr=sr, hop_length=hop_length)
        gap_end_frame   = librosa.time_to_frames(2.08, sr=sr, hop_length=hop_length)
        spec_mask[:, gap_start_frame:gap_end_frame] = 1

        mask_t = torch.from_numpy(spec_mask).float()
        mask_t = mask_t.to(device)
        log_impaired_magnitude = np.log10(np.abs(original_spectrogram * (1 - spec_mask)) + 1e-9)
        
        # utils.visualize_spectrogram(10 ** log_impaired_magnitude, in_db=False, power=1, title="Gap Spectrogram")

        log_impaired_magnitude = torch.from_numpy(log_impaired_magnitude).unsqueeze(0).float()
        log_impaired_magnitude = log_impaired_magnitude.to(device)
        

        with torch.no_grad():
            inpainted = 10 ** model.reconstruct_spectrogram(log_impaired_magnitude, mask_t)     # Make sure to undo log10 transform
            inpainted = inpainted.squeeze(0)  # Remove batch dimension
            # utils.visualize_spectrogram(abs(original_spectrogram), in_db=False, power=1, title="Original Spectrogram")

            # utils.visualize_spectrogram(inpainted.cpu().numpy(), in_db=False, power=1, title="Inpainted Spectrogram")
            # import matplotlib.pyplot as plt
            # plt.show()
    else:
        raise ValueError("Unknown model type.")

    # Save the inpainted audio
    utils.save_audio(
        utils.spectrogram_to_audio(
            inpainted.cpu().numpy(), 
            phase=None, #original_phase_t.cpu().numpy(), 
            phase_info=False, 
            n_fft=config['data']['spectrogram']['n_fft'], 
            hop_length=config['data']['spectrogram']['hop_length'], 
            win_length=config['data']['spectrogram']['win_length']
        ),
        file_path=output_path,
        sample_rate=sr
    )

def run_evaluation(input_dir, output_dir, model_type, checkpoint, config_path):
    """Main function to orchestrate the evaluation."""
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        return

    if not os.path.exists(checkpoint):
        print(f"Error: Checkpoint file not found: {checkpoint}")
        return

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(model_type, config_path, checkpoint, device)
    if model is None:
        return # Model loading failed

    # Find all .flac files
    flac_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith('.flac')])
    print(f"Found {len(flac_files)} .flac files in {input_dir}")

    # Process each file
    for filename in flac_files:
        input_path = input_dir + "/" + filename
        output_filename = f"{os.path.splitext(filename)[0]}_inpainted.flac"
        output_path = output_dir + "/" + output_filename
        
        inpaint(model, config_path, input_path, output_path, device) # Add mask params if needed

if __name__ == "__main__":
    # --- Configuration ---
    CONFIG_PATH = "CNNBLSTM/cnn_blstm.yaml"
    INPUT_DIRECTORY = "../test_samples"  
    OUTPUT_DIRECTORY = "../test_samples_reconstructed"
    MODEL_TYPE = "cnnlstm"  # "gan" or "cnnlstm"
    CHECKPOINT_PATH = "CNNBLSTM/checkpoints/blstm_cnn_epoch_75.pt" 
    # FORCE_CPU = False # Set to True to force CPU usage
    # --- End Configuration ---

    # Basic validation for required paths
    run_evaluation(
        input_dir=INPUT_DIRECTORY,
        output_dir=OUTPUT_DIRECTORY,
        model_type=MODEL_TYPE,
        checkpoint=CHECKPOINT_PATH,
        config_path=CONFIG_PATH,
        # force_cpu=FORCE_CPU
    )