import random
import re
from typing import Optional
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import yaml
import os
import logging
from pathlib import Path
import time
import numpy as np
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from loss import VGGLoss
from dataset import SpeechInpaintingDataset
from networks import PConvUNet, Discriminator
from utils import visualize_spectrogram, save_audio, spectrogram_to_audio

# --- Configuration Loading ---
def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

# --- Loss Functions ---
def calculate_losses(cfg, generated_mag, original_mag, mask, d_fake_pred,
                     vgg_loss_calculator: Optional[VGGLoss] = None):
    """Calculates all generator losses based on config weights."""
    loss_cfg = cfg['training']
    bce_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss(reduction='sum') # Use sum and manually normalize

    # Adversarial Loss (Generator wants D to predict Real=1)
    target_real = torch.ones_like(d_fake_pred)
    g_loss_adv = bce_loss(d_fake_pred, target_real)

    # Reconstruction Losses (L1) [Brodcast: (B, C, H, W)]
    mask = mask.view_as(generated_mag) if mask.dim() < generated_mag.dim() else mask
    if generated_mag.shape[1] != 1: generated_mag = generated_mag[:, :1] # Ensure 1 channel for L1
    if original_mag.shape[1] != 1: original_mag = original_mag[:, :1]

    # L1 Valid Loss (Lv)
    valid_pixels = generated_mag * mask
    target_valid_pixels = original_mag * mask
    num_valid = torch.sum(mask) + 1e-8
    g_loss_l1_valid = l1_loss(valid_pixels, target_valid_pixels) / num_valid

    # L1 Hole Loss (Lh)
    hole_mask = 1.0 - mask
    hole_pixels = generated_mag * hole_mask
    target_hole_pixels = original_mag * hole_mask
    num_hole = torch.sum(hole_mask) + 1e-8
    g_loss_l1_hole = l1_loss(hole_pixels, target_hole_pixels) / num_hole

    mag_weight = torch.abs(original_mag)
    g_loss_mag_weighted = torch.mean(torch.abs(generated_mag - original_mag) * mag_weight)

    # --- VGG Perceptual and Style Loss ---
    g_loss_vgg_perceptual = torch.tensor(0.0, device=generated_mag.device)
    g_loss_vgg_style = torch.tensor(0.0, device=generated_mag.device)
    if vgg_loss_calculator is not None and (loss_cfg['lambda_vgg_perceptual'] > 0 or loss_cfg['lambda_vgg_style'] > 0):
        g_loss_vgg_perceptual, g_loss_vgg_style = vgg_loss_calculator(generated_mag, original_mag)

    # Total Generator Loss
    g_loss_total = (loss_cfg['lambda_adv'] * g_loss_adv +
                    loss_cfg['lambda_l1_valid'] * g_loss_l1_valid +
                    loss_cfg['lambda_l1_hole'] * g_loss_l1_hole +
                    loss_cfg['lambda_mag_weighted'] * g_loss_mag_weighted +
                    loss_cfg['lambda_vgg_perceptual'] * g_loss_vgg_perceptual + 
                    loss_cfg['lambda_vgg_style'] * g_loss_vgg_style)

    losses = {
        'g_total': g_loss_total,
        'g_adv': g_loss_adv,
        'g_l1_valid': g_loss_l1_valid,
        'g_l1_hole': g_loss_l1_hole,
        'g_mag_weighted': g_loss_mag_weighted,
        'g_vgg_perceptual': g_loss_vgg_perceptual, 
        'g_vgg_style': g_loss_vgg_style
    }
    return losses

def find_latest_checkpoint(checkpoint_dir: Path):
    """Finds the latest checkpoint file in the given directory."""
    latest_epoch = -1
    latest_opt_path = None
    opt_files = list(checkpoint_dir.glob("*.pth"))
    
    if not opt_files:
        return None, None, None, -1
    for f in opt_files:
        match = re.search(r'optimizers_epoch_(\d+).pth', f.name)
        if match:
            epoch = int(match.group(1))
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_opt_path = f

    if latest_epoch == -1:
        return None, None, None, -1 # No valid checkpoint files found

    # Construct corresponding generator and discriminator paths
    gen_path = checkpoint_dir / f"generator_epoch_{latest_epoch:04d}.pth"
    disc_path = checkpoint_dir / f"discriminator_epoch_{latest_epoch:04d}.pth"

    # Check if all three files exist
    if gen_path.exists() and disc_path.exists() and latest_opt_path.exists():
        return gen_path, disc_path, latest_opt_path, latest_epoch
    else:
        logging.warning(f"Incomplete checkpoint found for latest epoch {latest_epoch} in {checkpoint_dir}. Files expected: {gen_path.name}, {disc_path.name}, {latest_opt_path.name}.")
        opt_files.sort(key=lambda x: int(re.search(r'optimizers_epoch_(\d+).pth', x.name).group(1)), reverse=True)
        for opt_path_candidate in opt_files:
             match = re.search(r'optimizers_epoch_(\d+).pth', opt_path_candidate.name)
             if match:
                  epoch_candidate = int(match.group(1))
                  gen_path_candidate = checkpoint_dir / f"generator_epoch_{epoch_candidate:04d}.pth"
                  disc_path_candidate = checkpoint_dir / f"discriminator_epoch_{epoch_candidate:04d}.pth"
                  if gen_path_candidate.exists() and disc_path_candidate.exists():
                       logging.warning(f"Found complete checkpoint for previous epoch {epoch_candidate}. Using this one.")
                       return gen_path_candidate, disc_path_candidate, opt_path_candidate, epoch_candidate
        logging.error(f"No complete checkpoint found in {checkpoint_dir}.")
        return None, None, None, -1

# --- Main Training Script ---
if __name__ == "__main__":
    cfg = load_config()
    train_cfg = cfg['training']
    paths_cfg = cfg['paths']
    log_cfg = cfg['logging']
    data_cfg = cfg['data']
    spec_cfg = data_cfg['spectrogram']

    # --- Setup ---
    run_name = f"{log_cfg['run_name']}_vgg_{time.strftime('%Y%m%d_%H%M%S')}"

    resume_from_checkpoint = train_cfg.get('resume_from_chkpt', False)
    resume_run_name  = train_cfg.get('resume_run_name', None)
    resume_epoch = train_cfg.get('resume_epoch', None)
    chkpt_dir_to_resume = None

    if resume_from_checkpoint:
        chkpt_dir_to_resume = Path(paths_cfg['checkpoint_dir']) / resume_run_name
        logging.info(f"Attempting to resume from specified run: {resume_run_name}")


    tb_dir = Path(paths_cfg['tensorboard_dir']) / run_name
    chkpt_dir = Path(paths_cfg['checkpoint_dir']) / run_name
    sample_dir = Path(paths_cfg['sample_dir']) / run_name
    log_dir = Path(paths_cfg['log_dir'])
    log_file = log_dir / f"{run_name}.log"

    tb_dir.mkdir(parents=True, exist_ok=True)
    chkpt_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
    logger = logging.getLogger()
    if resume_from_checkpoint:
        logger.info(f"Resuming run: {run_name}")
    else: 
        logger.info(f"Starting run: {run_name}")
    logger.info(f"Config: \n{yaml.dump(cfg, indent=2)}")

    # Tensorboard
    writer = SummaryWriter(log_dir=str(tb_dir))

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Datasets and Dataloaders ---
    logger.info("Loading datasets...")
    train_dataset = SpeechInpaintingDataset(cfg, dataset_type='train')
    valid_dataset = SpeechInpaintingDataset(cfg, dataset_type='valid')
    
    # --- Subset the training  ---
    train_limit = data_cfg['train_limit']
    
    num_available = len(train_dataset)
    indices = random.sample(range(num_available), k=min(train_limit, num_available))
    train_dataset = Subset(train_dataset, indices)

    train_loader = DataLoader(train_dataset, batch_size=train_cfg['batch_size'],
                              shuffle=True, num_workers=log_cfg['num_workers'],
                              pin_memory=True, drop_last=True, persistent_workers=log_cfg['num_workers']>0)
    valid_loader = DataLoader(valid_dataset, batch_size=train_cfg['batch_size'],
                              shuffle=False, num_workers=log_cfg['num_workers'],
                              pin_memory=True, persistent_workers=log_cfg['num_workers']>0)
    logger.info(f"Train dataset size: {len(train_dataset)}, Valid dataset size: {len(valid_dataset)}")


    # --- Models ---
    logger.info("Initializing models...")
    generator = PConvUNet(
        input_channels=cfg['model']['generator']['input_channels'],
        mask_channels=cfg['model']['generator']['mask_channels'],
        output_channels=cfg['model']['generator']['output_channels'],
        enc_layer_cfg=cfg['model']['generator'].get('enc_layer_cfg', [ 
             (64, 7, 2, 3), (128, 5, 2, 2), (256, 5, 2, 2),
             (512, 3, 2, 1), (512, 3, 2, 1), (512, 3, 2, 1), (512, 3, 2, 1)
         ]),
    ).to(device)

    discriminator = Discriminator(
        input_channels=cfg['model']['discriminator']['input_channels'],
         layer_cfg=cfg['model']['discriminator'].get('layer_cfg', [ 
             (64, 2, False), (128, 2, False), (256, 2, False), (512, 1, False)
         ]),
        use_spectral_norm=cfg['model']['discriminator']['use_spectral_norm']
    ).to(device)

    logger.info(f"Generator parameters: {sum(p.numel() for p in generator.parameters() if p.requires_grad):,}")
    logger.info(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters() if p.requires_grad):,}")

    # --- VGG Loss Calculator ---
    use_vgg = train_cfg['lambda_vgg_perceptual'] > 0 or train_cfg['lambda_vgg_style'] > 0
    vgg_loss_calc = VGGLoss(device=device) if use_vgg else None
    if use_vgg:
        logger.info("VGG Loss enabled.")
    else:
        logger.info("VGG Loss disabled by config weights.")


    # --- Optimizers ---
    g_optimizer = optim.Adam(generator.parameters(), lr=train_cfg['g_lr'], betas=(train_cfg['b1'], train_cfg['b2']))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=train_cfg['d_lr'], betas=(train_cfg['b1'], train_cfg['b2']))

    # --- Loss Function (Discriminator) ---
    bce_loss = nn.BCEWithLogitsLoss()

    # --- Training Loop ---
    start_epoch = 0
    global_step = 0
    if resume_from_checkpoint and chkpt_dir_to_resume is not None:
        loaded_epoch = -1
        chkpt_path_g = None
        chkpt_path_d = None
        chkpt_path_opt = None
        
        if not chkpt_dir_to_resume.exists():
            logger.warning(f"Checkpoint directory {chkpt_dir_to_resume} not found. Starting training from scratch.")
        else:
            if resume_epoch is not None:
                logger.info(f"Attempting to load specific checkpoint for epoch {resume_epoch} from {chkpt_dir_to_resume}")
                # Construct expected paths for the specific epoch
                specific_gen_path = chkpt_dir_to_resume / f"generator_epoch_{resume_epoch:04d}.pth"
                specific_disc_path = chkpt_dir_to_resume / f"discriminator_epoch_{resume_epoch:04d}.pth"
                specific_opt_path = chkpt_dir_to_resume / f"optimizers_epoch_{resume_epoch:04d}.pth"
                if specific_gen_path.exists() and specific_disc_path.exists() and specific_opt_path.exists():
                    chkpt_path_g = specific_gen_path
                    chkpt_path_d = specific_disc_path
                    chkpt_path_opt = specific_opt_path
                    loaded_epoch = resume_epoch
                    logger.info(f"Found complete checkpoint for specified epoch {resume_epoch}.")
                else:
                    logger.error(f"Could not find complete checkpoint files for specified epoch {resume_epoch} in {chkpt_dir_to_resume}. Expected:")
                    logger.error(f" - {specific_gen_path.name}")
                    logger.error(f" - {specific_disc_path.name}")
                    logger.error(f" - {specific_opt_path.name}")
                    logger.error("Starting training from scratch.")
                    # Keep loaded_epoch = -1

            # --- If no specific epoch requested, find the latest ---
            else:
                logger.info(f"Searching for latest checkpoint in: {chkpt_dir_to_resume}")
                chkpt_path_g, chkpt_path_d, chkpt_path_opt, latest_epoch = find_latest_checkpoint(chkpt_dir_to_resume)
                if latest_epoch != -1:
                    loaded_epoch = latest_epoch # Use the found latest epoch number
                    logger.info(f"Found latest complete checkpoint from epoch {latest_epoch}.")
                else:
                    logger.warning(f"No complete checkpoint found in {chkpt_dir_to_resume}. Starting training from scratch.")
                    # Keep loaded_epoch = -1

            # --- Load if a valid checkpoint (specific or latest) was identified ---
            if loaded_epoch != -1 and chkpt_path_g and chkpt_path_d and chkpt_path_opt:
                try:
                    logger.info(f"Loading generator weights from: {chkpt_path_g}")
                    generator.load_state_dict(torch.load(chkpt_path_g, map_location=device))

                    logger.info(f"Loading discriminator weights from: {chkpt_path_d}")
                    discriminator.load_state_dict(torch.load(chkpt_path_d, map_location=device))

                    logger.info(f"Loading optimizer states from: {chkpt_path_opt}")
                    checkpoint = torch.load(chkpt_path_opt, map_location=device)

                    # Check if keys exist before loading (for backward compatibility)
                    if 'g_optimizer_state_dict' in checkpoint:
                         g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
                    else:
                         logger.warning("Generator optimizer state not found in checkpoint.")
                    if 'd_optimizer_state_dict' in checkpoint:
                         d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
                    else:
                         logger.warning("Discriminator optimizer state not found in checkpoint.")

                    # Load epoch and global step, handle potential missing keys
                    start_epoch = checkpoint.get('epoch', -1) + 1 # Start from next epoch
                    global_step = checkpoint.get('global_step', 0)
                    if start_epoch == 0: # Check if epoch key was missing
                         logger.warning("'epoch' key not found in optimizer checkpoint. Resetting start_epoch to 0.")
                         start_epoch = 0 # Safety reset
                    if global_step == 0 and start_epoch != 0:
                         logger.warning("'global_step' key not found or was 0 in optimizer checkpoint. Resetting global_step to 0.")
                         global_step = 0 # Safety reset, though might be inaccurate

                    logger.info(f"Successfully resumed training from epoch {start_epoch}, global step {global_step}.")

                except Exception as e:
                    logger.error(f"Error loading checkpoint files for epoch {loaded_epoch}: {e}. Starting training from scratch.", exc_info=True)
                    start_epoch = 0
                    global_step = 0
            
    logger.info("Starting training...")

    for epoch in range(start_epoch, train_cfg['epochs']):
        generator.train()
        discriminator.train()
        epoch_g_loss_total = 0.0
        epoch_d_loss = 0.0
        epoch_g_loss_adv = 0.0
        epoch_g_loss_l1v = 0.0
        epoch_g_loss_l1h = 0.0
        epoch_g_loss_lw = 0.0
        epoch_g_loss_vgg_p = 0.0
        epoch_g_loss_vgg_s = 0.0
        batch_count = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_cfg['epochs']}", leave=False)

        for batch_idx, batch in enumerate(progress_bar):
            original_mag = batch['original_magnitude'].to(device) # (B, 1, F, T)
            impaired_mag = batch['impaired_magnitude'].to(device) # (B, 1, F, T)
            mask = batch['mask'].to(device)                       # (B, 1, F, T) 0 for hole
            original_phase = batch['original_phase'].to(device)   # For saving samples

            # --- Train Discriminator ---
            d_optimizer.zero_grad()
            with torch.no_grad():
                generated_mag = generator(impaired_mag, mask)

            # Real samples
            d_real_pred = discriminator(original_mag)
            target_real = torch.ones_like(d_real_pred, device=device)
            d_loss_real = bce_loss(d_real_pred, target_real)
            
            d_fake_pred = discriminator(generated_mag.detach())
            target_fake = torch.zeros_like(d_fake_pred, device=device)
            d_loss_fake = bce_loss(d_fake_pred, target_fake)

            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            d_optimizer.step()

            # --- Train Generator ---
            g_optimizer.zero_grad()

            d_fake_pred_g = discriminator(generated_mag)

            # Calculate generator losses (Adv, L1s, MagWeighted, VGG)
            g_losses = calculate_losses(
                cfg, generated_mag, original_mag, mask, d_fake_pred_g,
                vgg_loss_calculator=vgg_loss_calc
            )
            g_loss = g_losses['g_total']

            g_loss.backward()
            g_optimizer.step()

            # --- Logging ---
            epoch_g_loss_total += g_loss.item()
            epoch_d_loss += d_loss.item()
            epoch_g_loss_adv += g_losses['g_adv'].item()
            epoch_g_loss_l1v += g_losses['g_l1_valid'].item()
            epoch_g_loss_l1h += g_losses['g_l1_hole'].item()
            epoch_g_loss_lw += g_losses['g_mag_weighted'].item()
            epoch_g_loss_vgg_p += g_losses['g_vgg_perceptual'].item()
            epoch_g_loss_vgg_s += g_losses['g_vgg_style'].item()

            batch_count += 1
            global_step += 1

            progress_bar.set_postfix({
                "G Loss": f"{g_loss.item():.3f}",
                "D Loss": f"{d_loss.item():.3f}",
                "G Adv": f"{g_losses['g_adv'].item():.3f}",
                "G L1V": f"{g_losses['g_l1_valid'].item():.3f}",
                "G L1H": f"{g_losses['g_l1_hole'].item():.3f}",
                "G VGG_P": f"{g_losses['g_vgg_perceptual'].item():.3f}",
                "G VGG_S": f"{g_losses['g_vgg_style'].item():.3f}",
            })

            if global_step % log_cfg['log_interval'] == 0:
                writer.add_scalar('Loss_Train/Generator_Total', g_loss.item(), global_step)
                writer.add_scalar('Loss_Train/Discriminator', d_loss.item(), global_step)
                writer.add_scalar('Loss_Train/Generator_Adversarial', g_losses['g_adv'].item(), global_step)
                writer.add_scalar('Loss_Train/Generator_L1_Valid', g_losses['g_l1_valid'].item(), global_step)
                writer.add_scalar('Loss_Train/Generator_L1_Hole', g_losses['g_l1_hole'].item(), global_step)
                writer.add_scalar('Loss_Train/Generator_MagWeighted', g_losses['g_mag_weighted'].item(), global_step)
                writer.add_scalar('Loss_Train/Generator_VGG_Perceptual', g_losses['g_vgg_perceptual'].item(), global_step)
                writer.add_scalar('Loss_Train/Generator_VGG_Style', g_losses['g_vgg_style'].item(), global_step)
                writer.add_scalar('Loss_Train/Discriminator_Real', d_loss_real.item(), global_step)
                writer.add_scalar('Loss_Train/Discriminator_Fake', d_loss_fake.item(), global_step)
                # Log learning rates
                writer.add_scalar('LR/Generator', g_optimizer.param_groups[0]['lr'], global_step)
                writer.add_scalar('LR/Discriminator', d_optimizer.param_groups[0]['lr'], global_step)


            # --- Sample Generation ---
            if global_step % log_cfg['sample_interval'] == 0:
                generator.eval() # Switch to eval mode for generation
                with torch.no_grad():
                    # Take first item from batch for visualization
                    orig_mag_np = original_mag[0].cpu().numpy().squeeze()
                    impaired_mag_np = impaired_mag[0].cpu().numpy().squeeze()
                    # Use the generated_mag calculated before G step for consistency
                    generated_mag_np = generated_mag[0].cpu().numpy().squeeze()
                    mask_np = mask[0].cpu().numpy().squeeze()
                    phase_np = original_phase[0].cpu().numpy().squeeze() # Use original phase

                    # --- Calculate gap coordinates for visualization ---
                    is_gap_time = np.min(mask_np, axis=0) < 0.5 # Check if any freq bin has mask < 0.5
                    gap_indices_time = np.where(is_gap_time)[0]
                    gap_start_frame = gap_indices_time.min() if len(gap_indices_time) > 0 else 0
                    gap_end_frame = gap_indices_time.max() if len(gap_indices_time) > 0 else 0
                    gap_start_s = gap_start_frame * spec_cfg['hop_length'] / data_cfg['sample_rate']
                    gap_end_s = (gap_end_frame + 1) * spec_cfg['hop_length'] / data_cfg['sample_rate'] # +1 to cover end frame


                    # --- Visualize Spectrograms ---
                    vis_kwargs = {
                        'sample_rate': data_cfg['sample_rate'],
                        'hop_length': spec_cfg['hop_length'],
                        'in_db': False, # Input mags are log1p or Tanh
                        #'y_axis': 'linear', # Freq axis
                        'gap_int': (gap_start_s, gap_end_s) if gap_end_frame > gap_start_frame else None
                    }
                    # Create figures (function returns figure object)
                    fig_orig = visualize_spectrogram(orig_mag_np, title="Original Mag", **vis_kwargs)
                    fig_imp = visualize_spectrogram(impaired_mag_np, title="Impaired Mag", **vis_kwargs)
                    fig_gen = visualize_spectrogram(generated_mag_np, title="Generated Mag (Tanh)", **vis_kwargs) # Note range

                    # Log figures to TensorBoard
                    if fig_orig: writer.add_figure("Spectrograms/Original", fig_orig, global_step)
                    if fig_imp: writer.add_figure("Spectrograms/Impaired", fig_imp, global_step)
                    if fig_gen: writer.add_figure("Spectrograms/Generated", fig_gen, global_step)
                    plt.close('all') # Close all generated figures


                    # --- Reconstruct and Save Audio Samples ---
                    # NOTE: We need to invert the normalization done by the dataset/generator
                    # Assuming log1p for original/impaired, Tanh for generated
                    # And expect linear magnitude

                    # Invert generator Tanh output: Needs correct scaling inversion if used
                    # For now, let's use the combined approach with original phase where possible

                    # Create combined magnitude: Use generated in hole, original elsewhere
                    # Need to potentially un-normalize original_mag (invert log1p)
                    orig_mag_linear_approx = np.expm1(orig_mag_np)
                    # Invert Tanh for generated mag and scale back if needed?
                    # For simplicity in saving, let's use orig phase + combined *log* mag for recon
                    combined_log_mag = generated_mag_np * (1 - mask_np) + orig_mag_np * mask_np

                    # Reconstruct using original phase + combined log magnitude
                    audio_recon_combined = spectrogram_to_audio(
                         combined_log_mag, phase_np,
                         n_fft=spec_cfg['n_fft'],
                         hop_length=spec_cfg['hop_length'],
                         win_length=spec_cfg['win_length'],
                         window=spec_cfg['window']
                    )
                    
                    writer.add_audio("Audio/Generated_CombinedLogMag_OrigPhase", audio_recon_combined, global_step, sample_rate=data_cfg['sample_rate'])
                    save_audio(audio_recon_combined, sample_dir / f"step_{global_step}_recon_comb_origphase.flac", data_cfg['sample_rate'])

                    # Save original and impaired for reference
                    audio_orig = spectrogram_to_audio(
                        orig_mag_np, phase_np,
                        n_fft=spec_cfg['n_fft'],
                        hop_length=spec_cfg['hop_length'],
                        win_length=spec_cfg['win_length'],
                        window=spec_cfg['window']
                    )
                    audio_imp = spectrogram_to_audio(
                        impaired_mag_np, phase_np,
                        n_fft=spec_cfg['n_fft'],
                        hop_length=spec_cfg['hop_length'],
                        win_length=spec_cfg['win_length'],
                        window=spec_cfg['window']
                    )
                    save_audio(audio_orig, sample_dir / f"step_{global_step}_original.flac", data_cfg['sample_rate'])
                    save_audio(audio_imp, sample_dir / f"step_{global_step}_impaired.flac", data_cfg['sample_rate'])


                generator.train() # Set back to train mode

        # --- End of Epoch ---
        avg_g_loss_total = epoch_g_loss_total / batch_count
        avg_d_loss = epoch_d_loss / batch_count
        avg_g_loss_adv = epoch_g_loss_adv / batch_count
        avg_g_loss_l1v = epoch_g_loss_l1v / batch_count
        avg_g_loss_l1h = epoch_g_loss_l1h / batch_count
        avg_g_loss_lw = epoch_g_loss_lw / batch_count
        avg_g_loss_vgg_p = epoch_g_loss_vgg_p / batch_count
        avg_g_loss_vgg_s = epoch_g_loss_vgg_s / batch_count

        logger.info(f"Epoch {epoch+1} Summary: Avg G Loss: {avg_g_loss_total:.4f}, Avg D Loss: {avg_d_loss:.4f}")
        logger.info(f"  Avg G Losses -> Adv: {avg_g_loss_adv:.4f}, L1V: {avg_g_loss_l1v:.4f}, L1H: {avg_g_loss_l1h:.4f}, Lw: {avg_g_loss_lw:.4f}, VGG_P: {avg_g_loss_vgg_p:.4f}, VGG_S: {avg_g_loss_vgg_s:.4f}")

        writer.add_scalar('Loss_Epoch/Generator_Total_Avg', avg_g_loss_total, epoch + 1)
        writer.add_scalar('Loss_Epoch/Discriminator_Avg', avg_d_loss, epoch + 1)
        writer.add_scalar('Loss_Epoch/Generator_Adv_Avg', avg_g_loss_adv, epoch + 1)
        writer.add_scalar('Loss_Epoch/Generator_L1V_Avg', avg_g_loss_l1v, epoch + 1)
        writer.add_scalar('Loss_Epoch/Generator_L1H_Avg', avg_g_loss_l1h, epoch + 1)
        writer.add_scalar('Loss_Epoch/Generator_Lw_Avg', avg_g_loss_lw, epoch + 1)
        writer.add_scalar('Loss_Epoch/Generator_VGG_P_Avg', avg_g_loss_vgg_p, epoch + 1)
        writer.add_scalar('Loss_Epoch/Generator_VGG_S_Avg', avg_g_loss_vgg_s, epoch + 1)

        # --- Validation Loop (Example - Calculate validation losses) ---
        if (epoch + 1) % log_cfg.get('validation_interval', 5) == 0:
            generator.eval()
            discriminator.eval()
            val_g_loss_total = 0.0
            val_d_loss = 0.0
            val_g_loss_adv = 0.0
            val_g_loss_l1v = 0.0
            val_g_loss_l1h = 0.0
            val_g_loss_lw = 0.0
            val_g_loss_vgg_p = 0.0
            val_g_loss_vgg_s = 0.0
            val_batch_count = 0
            
            logger.info(f"Running validation for epoch {epoch+1}...")
            with torch.no_grad():
                val_progress_bar = tqdm(valid_loader, desc=f"Validation Epoch {epoch+1}", leave=False)
                for batch in val_progress_bar:
                    original_mag = batch['original_magnitude'].to(device)
                    impaired_mag = batch['impaired_magnitude'].to(device)
                    mask = batch['mask'].to(device)

                    # --- Get Model Outputs ---
                    generated_mag = generator(impaired_mag, mask)
                    d_real_pred = discriminator(original_mag)
                    d_fake_pred = discriminator(generated_mag)

                    # --- Calculate D Loss ---
                    target_real = torch.ones_like(d_real_pred, device=device)
                    target_fake = torch.zeros_like(d_fake_pred, device=device)
                    d_loss_real = bce_loss(d_real_pred, target_real)
                    d_loss_fake = bce_loss(d_fake_pred, target_fake)
                    val_d_loss += ((d_loss_real + d_loss_fake) / 2).item()

                    # --- Calculate G Loss ---
                    g_losses = calculate_losses(
                        cfg, generated_mag, original_mag, mask, d_fake_pred,
                        vgg_loss_calculator=vgg_loss_calc
                    )
                    
                    val_g_loss_total += g_losses['g_total'].item()
                    val_g_loss_adv += g_losses['g_adv'].item()
                    val_g_loss_l1v += g_losses['g_l1_valid'].item()
                    val_g_loss_l1h += g_losses['g_l1_hole'].item()
                    val_g_loss_lw += g_losses['g_mag_weighted'].item()
                    val_g_loss_vgg_p += g_losses['g_vgg_perceptual'].item()
                    val_g_loss_vgg_s += g_losses['g_vgg_style'].item()
                    val_batch_count += 1

            # --- Log Average Validation Losses ---
            avg_val_g_loss = val_g_loss_total / val_batch_count
            avg_val_d_loss = val_d_loss / val_batch_count
            avg_val_g_adv = val_g_loss_adv / val_batch_count
            avg_val_g_l1v = val_g_loss_l1v / val_batch_count
            avg_val_g_l1h = val_g_loss_l1h / val_batch_count
            avg_val_g_lw = val_g_loss_lw / val_batch_count
            avg_val_g_vgg_p = val_g_loss_vgg_p / val_batch_count
            avg_val_g_vgg_s = val_g_loss_vgg_s / val_batch_count

            logger.info(f"Epoch {epoch+1} Validation: Avg G Loss: {avg_val_g_loss:.4f}, Avg D Loss: {avg_val_d_loss:.4f}")
            logger.info(f"  Avg Val G Losses -> Adv: {avg_val_g_adv:.4f}, L1V: {avg_val_g_l1v:.4f}, L1H: {avg_val_g_l1h:.4f}, Lw: {avg_val_g_lw:.4f}, VGG_P: {avg_val_g_vgg_p:.4f}, VGG_S: {avg_val_g_vgg_s:.4f}")

            writer.add_scalar('Loss_Val/Generator_Total_Avg', avg_val_g_loss, global_step)
            writer.add_scalar('Loss_Val/Discriminator_Avg', avg_val_d_loss, global_step)
            writer.add_scalar('Loss_Val/Generator_Adv_Avg', avg_val_g_adv, global_step)
            writer.add_scalar('Loss_Val/Generator_L1V_Avg', avg_val_g_l1v, global_step)
            writer.add_scalar('Loss_Val/Generator_L1H_Avg', avg_val_g_l1h, global_step)
            writer.add_scalar('Loss_Val/Generator_Lw_Avg', avg_val_g_lw, global_step)
            writer.add_scalar('Loss_Val/Generator_VGG_P_Avg', avg_val_g_vgg_p, global_step)
            writer.add_scalar('Loss_Val/Generator_VGG_S_Avg', avg_val_g_vgg_s, global_step)


        # --- Checkpointing ---
        if (epoch + 1) % log_cfg['checkpoint_interval'] == 0 or epoch == train_cfg['epochs'] - 1:
            chkpt_path_g = chkpt_dir / f"generator_epoch_{epoch+1:04d}.pth"
            chkpt_path_d = chkpt_dir / f"discriminator_epoch_{epoch+1:04d}.pth"
            # Save optimizers as well to resume training properly
            chkpt_path_opt = chkpt_dir / f"optimizers_epoch_{epoch+1:04d}.pth"

            torch.save(generator.state_dict(), chkpt_path_g)
            torch.save(discriminator.state_dict(), chkpt_path_d)
            torch.save({
                'g_optimizer_state_dict': g_optimizer.state_dict(),
                'd_optimizer_state_dict': d_optimizer.state_dict(),
                'epoch': epoch,
                'global_step': global_step
            }, chkpt_path_opt)
            logger.info(f"Saved checkpoints for epoch {epoch+1} at {chkpt_dir}")

    logger.info("Training finished.")
    writer.close()