import os
import soundfile as sf
import torch
import numpy as np

from pathlib import Path 
import sys

from models.GAN.train import load_config
import utils
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.GAN.networks import PConvUNet
from models.CNNBLSTM.model import StackedCNNLSTM

def load_model(model_type, checkpoint_path, device, config_path=None):
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
        model = StackedCNNLSTM(config_path).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(torch.load(checkpoint_path), weights_only=False, map_location=device)
    model.eval()

    return model

def inpaint(model, audio_path, output_path, device, mask_ratio=0.1):
    """Loads audio, applies mask, performs inpainting, and saves the result."""
    try:
        audio, sr = utils.load_audio(audio_path)
        print(f"Loaded {audio_path}, shape: {audio.shape}, sample rate: {sr}")

        gap_len_s = 0.5  # Example gap length in seconds

        time_domain_mask, (gap_start_sample, gap_end_sample) = utils.create_gap_mask(
            len(audio),
            gap_len_s,
            sr
        )
        
        impaired_audio = audio * time_domain_mask
        
        original_spectrogram = utils.extract_spectrogram(
            audio, n_fft=,
            hop_length=, win_length=, 
            window=, power=)
        
        original_magnitude = np.abs(original_spectrogram)
        original_magnitude = np.log1p(original_magnitude)
        original_phase     = np.angle(original_spectrogram)
        
        impaired_spectrogram = utils.extract_spectrogram(
            impaired_audio, n_fft=,
            hop_length=,
            win_length=,
            window=,
            power=
        )
        
        impaired_magnitude = np.abs(impaired_spectrogram)
        impaired_magnitude = np.log1p(impaired_magnitude)

        hop_length = 
        gap_start_frame = gap_start_sample // hop_length
        gap_end_frame = int(np.ceil(gap_end_sample / hop_length))

        num_frames = original_magnitude.shape[1]
        
        gap_start_frame = max(0, gap_start_frame)
        gap_end_frame = min(num_frames, gap_end_frame)

        spec_mask = np.ones_like(original_magnitude, dtype=np.float32)
        if gap_end_frame > gap_start_frame: 
            spec_mask[:, gap_start_frame:gap_end_frame] = 0

        original_magnitude_t = torch.from_numpy(original_magnitude).float()
        impaired_magnitude_t = torch.from_numpy(impaired_magnitude).float()
        mask_t = torch.from_numpy(spec_mask).float()
        original_phase_t = torch.from_numpy(original_phase).float()
        
        with torch_no_grad():
            if model_type == 'gan':
                inpainted = model(impaired_magnitude_t, mask_t)
            elif model_type == 'cnnlstm':
                inpainted = model.reconstruct_spectrogram(log_spectrogram_gaps, gap_masks)

        utils.save_audio(
            utils.spectrogram_to_audio(
                inpainted.cpu().numpy(), original_phase_t.cpu().numpy(), phase_info=True, n_fft=, hop_length=, win_length=, window=),
        )
        

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")

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

    model = load_model(model_type, checkpoint, device)
    if model is None:
        return # Model loading failed

    # Find all .flac files
    flac_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.flac')]
    print(f"Found {len(flac_files)} .flac files in {input_dir}")

    # Process each file
    for filename in flac_files:
        input_path = os.path.join(input_dir, filename)
        output_filename = f"{os.path.splitext(filename)[0]}_inpainted.flac"
        output_path = os.path.join(output_dir, output_filename)
        
        inpaint(model, input_path, output_path, device) # Add mask params if needed

if __name__ == "__main__":
    # --- Configuration ---
    CONFIG_PATH = "path/to/your/config.yaml" 
    INPUT_DIRECTORY = "path/to/your/input/flac/files"  
    OUTPUT_DIRECTORY = "path/to/your/output/directory"
    MODEL_TYPE = "gan"  # "gan" or "cnnlstm"
    CHECKPOINT_PATH = "path/to/your/model.pth" 
    FORCE_CPU = False # Set to True to force CPU usage
    # --- End Configuration ---

    # Basic validation for required paths
    if INPUT_DIRECTORY == "path/to/your/input/flac/files" or \
       OUTPUT_DIRECTORY == "path/to/your/output/directory" or \
       CHECKPOINT_PATH == "path/to/your/model.pth":
        print("Error: Please update the INPUT_DIRECTORY, OUTPUT_DIRECTORY, and CHECKPOINT_PATH variables in the script.")
    else:
        run_evaluation(
            input_dir=INPUT_DIRECTORY,
            output_dir=OUTPUT_DIRECTORY,
            model_type=MODEL_TYPE,
            checkpoint=CHECKPOINT_PATH,
            force_cpu=FORCE_CPU
        )