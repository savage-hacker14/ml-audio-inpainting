import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import logging
import sys

import utils

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from dataset import SpeechInpaintingDataset
from loss import VGGLoss
from networks import PConvUNet
from utils import spectrogram_to_audio
from train import load_config, find_latest_checkpoint

if __name__ == "__main__":
    cfg = load_config()
    train_cfg = cfg['training']
    paths_cfg = cfg['paths']
    log_cfg = cfg['logging']
    data_cfg = cfg['data']
    spec_cfg = data_cfg['spectrogram']
    
    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # --- Load the model ---
    chkpt_dir = Path(paths_cfg['checkpoint_dir'])
    if not chkpt_dir.is_dir():
        logging.error(f"Checkpoint directory not found: {chkpt_dir}")
        sys.exit(1)
        
    generator = PConvUNet(
        input_channels=cfg['model']['generator']['input_channels'],
        mask_channels=cfg['model']['generator']['mask_channels'],
        output_channels=cfg['model']['generator']['output_channels'],
        enc_layer_cfg=cfg['model']['generator'].get('enc_layer_cfg', [ 
             (64, 7, 2, 3), (128, 5, 2, 2), (256, 5, 2, 2),
             (512, 3, 2, 1), (512, 3, 2, 1), (512, 3, 2, 1), (512, 3, 2, 1)
         ]),
    ).to(device)
    generator.load_state_dict(torch.load(find_latest_checkpoint(chkpt_dir, 'generator'), weights_only=False, map_location=device))
    
    use_vgg       = train_cfg['lambda_vgg_perceptual'] > 0 or train_cfg['lambda_vgg_style'] > 0
    vgg_loss_calc = VGGLoss(device=device) if use_vgg else None
    
    dataset       = SpeechInpaintingDataset(cfg=cfg, dataset_type='test')
    data_loader   = DataLoader(dataset, batch_size=train_cfg['batch_size'],
                              shuffle=False, num_workers=log_cfg['num_workers'],
                              pin_memory=True, drop_last=True, persistent_workers=log_cfg['num_workers']>0)
    criterion     = nn.BCEWithLogitsLoss()

    # --- Evaluate the model ---
    generator.eval()
    for batch_idx, (batch) in enumerate(data_loader):
        original_magnitude = batch['original_magnitude'].to(device)
        impaired_magnitude = batch['impaired_magnitude'].to(device)
        mask               = batch['mask'].to(device)
        original_phase     = batch['original_phase'].to(device)

        # --- Forward pass ---
        with torch.no_grad():
            generated_output = generator(impaired_magnitude, mask)

        # --- Loss calculation ---
        if use_vgg:
            perceptual_loss, style_loss = vgg_loss_calc(generated_output, original_magnitude)
            total_loss = train_cfg['lambda_vgg_perceptual'] * perceptual_loss + train_cfg['lambda_vgg_style'] * style_loss
            logging.info(f"Total Loss: {total_loss.item()}")
        
        # --- Convert to audio ---
        generated_audio = spectrogram_to_audio(generated_output, original_phase, spec_cfg['n_fft'], spec_cfg['hop_length'])
        original_audio = spectrogram_to_audio(original_magnitude, original_phase, spec_cfg['n_fft'], spec_cfg['hop_length'])
        impaired_audio = spectrogram_to_audio(impaired_magnitude, original_phase, spec_cfg['n_fft'], spec_cfg['hop_length'])
        
        # --- Save the audio ---
        utils.save_audio(generated_audio, paths_cfg['output_dir'], f"generated_{batch_idx}.flac")
        utils.save_audio(original_audio, paths_cfg['output_dir'], f"original_{batch_idx}.flac")
        utils.save_audio(impaired_audio, paths_cfg['output_dir'], f"impaired_{batch_idx}.flac")
        
        break  # For testing, break after first batch
