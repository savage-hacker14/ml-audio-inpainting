from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

EPSILON = 1e-8


class VGG19FeatureExtractor(nn.Module):
    """
    Extracts features from intermediate layers of a pre-trained VGG19 network.
    Used as a proxy for the SpeechVGG mentioned in the paper.
    """
    def __init__(self, requires_grad: bool = False):
        super().__init__()
        vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features # Use new weights API
        # Define layers to extract features from
        self.layer_indices = {
            'relu1_1': 1,  # After conv1_1
            'relu2_1': 6,  # After conv2_2
            'relu3_1': 11, # After conv3_2
            'relu4_1': 20, # After conv4_2
            'relu5_1': 29  # After conv5_2
        }
        # Create a list of layers up to the max index needed
        max_idx = max(self.layer_indices.values())
        self.features = vgg19[:max_idx + 1] # Include the target layer

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
        self.eval() # Ensure VGG is in eval mode

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor [B, C, H, W]. Expects 3 channels.

        Returns:
            Dict[str, torch.Tensor]: Dictionary mapping layer names to feature maps.
        """
        # VGG expects 3 input channels and specific normalization
        # Replicate grayscale if needed
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.size(1) != 3:
             raise ValueError(f"VGG19 expects 1 or 3 input channels, got {x.size(1)}")

        # Apply VGG normalization (example, adjust if your data has different range)
        # mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        # std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        # x = (x - mean) / std
        # NOTE: Skipping normalization for now, assuming log1p scale is somewhat handled.
        # If using linear magnitude, normalization would be more important.

        output_features = {}
        current_feat = x
        for i, layer in enumerate(self.features):
            current_feat = layer(current_feat)
            for name, idx in self.layer_indices.items():
                if i == idx:
                    output_features[name] = current_feat
                    break # Found feature for this layer name

        if len(output_features) != len(self.layer_indices):
             print(f"Warning: Expected {len(self.layer_indices)} features, got {len(output_features)}")

        return output_features

def gram_matrix(input_tensor: torch.Tensor) -> torch.Tensor:
    """ Computes the Gram matrix for style loss. """
    b, c, h, w = input_tensor.size()
    features = input_tensor.view(b, c, h * w)
    G = torch.bmm(features, features.transpose(1, 2)) # Batch matrix multiplication
    # Normalize by the number of elements in the feature map (c*h*w)
    return G.div(c * h * w)


class InpaintingLoss(nn.Module):
    """
    Calculates the combined generator loss for the speech inpainting GAN.
    Includes: Adversarial Loss, L1 Reconstruction Losses (valid, hole),
              Perceptual Loss, Style Loss, Magnitude-Weighted L1 Loss.
    Averages L1 losses over the relevant pixels only.
    """
    def __init__(self, feature_extractor: nn.Module, lambda_dict: Dict[str, float]):
        super().__init__()
        self.feature_extractor = feature_extractor
        # Use reduction='none' to calculate per-pixel loss before manual averaging
        self.l1_loss_pixel = nn.L1Loss(reduction='none')
        self.lambda_dict = lambda_dict

    def forward(self, mask: torch.Tensor, generated_spec: torch.Tensor,
                gt_spec: torch.Tensor, disc_fake_pred: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            mask (torch.Tensor): Binary mask (1 for valid, 0 for hole) [B, 1, H, W].
            generated_spec (torch.Tensor): Output from generator [B, 1, H, W].
            gt_spec (torch.Tensor): Ground truth spectrogram [B, 1, H, W].
            disc_fake_pred (torch.Tensor): Discriminator output for generated spec [B, 1].

        Returns:
            torch.Tensor: Total combined generator loss.
            dict: Dictionary containing individual loss components values (float).
        """
        # Ensure mask is float for calculations
        mask = mask.float()
        inv_mask = 1.0 - mask

        # --- Loss Components ---

        # Calculate per-pixel L1 loss
        pixel_l1_loss = self.l1_loss_pixel(generated_spec, gt_spec)

        # 1. L1 Valid Loss (Lv) - Average loss over non-masked regions
        loss_valid = (torch.sum(pixel_l1_loss * mask) /
                      (torch.sum(mask) + EPSILON))

        # 2. L1 Hole Loss (Lh) - Average loss over masked regions
        loss_hole = (torch.sum(pixel_l1_loss * inv_mask) /
                     (torch.sum(inv_mask) + EPSILON))

        # 3. Adversarial Loss (LB) - Generator trying to fool discriminator
        target_real = torch.ones_like(disc_fake_pred)
        loss_adversarial = F.binary_cross_entropy_with_logits(disc_fake_pred, target_real)

        # --- VGG-based Losses (Perceptual & Style) ---
        # Composite output used for perceptual loss (Eq 4 not explicit on composition for Lp, but common practice)
        composited_spec = gt_spec * mask + generated_spec * inv_mask

        # Extract features (expects 3 channels, VGG extractor handles replication)
        features_comp = self.feature_extractor(composited_spec) # Use composited for perceptual
        features_gt = self.feature_extractor(gt_spec) # Use ground truth

        # Check if features were extracted correctly
        if not features_comp or not features_gt or len(features_comp) != len(features_gt):
             print("Warning: Feature extraction failed or lengths differ. Skipping VGG losses.")
             loss_perceptual = torch.tensor(0.0, device=gt_spec.device)
             loss_style = torch.tensor(0.0, device=gt_spec.device)
        else:
            loss_perceptual = 0.0
            loss_style = 0.0
            # Calculate perceptual and style losses layer by layer
            for layer_name in features_comp.keys():
                feat_comp = features_comp[layer_name]
                feat_gt = features_gt[layer_name]

                # Perceptual Loss (Lp) - L1 distance between features
                loss_perceptual += F.l1_loss(feat_comp, feat_gt) # Already averaged by L1Loss

                # Style Loss (Ls) - L1 distance between Gram matrices
                gram_comp = gram_matrix(feat_comp)
                gram_gt = gram_matrix(feat_gt)
                loss_style += F.l1_loss(gram_comp, gram_gt) # Already averaged by L1Loss

            # Average over the number of layers used
            num_feature_layers = len(features_comp)
            loss_perceptual /= num_feature_layers
            loss_style /= num_feature_layers


        # 5. Magnitude-Weighted L1 Loss (Lw) - Eq 5
        # Weight the L1 loss by the ground truth magnitude (using composited for loss comparison)
        # Paper uses ||Sgt|| * |Scomp - Sgt|. Average this weighted error.
        magnitude_weights = torch.abs(gt_spec) # Use absolute GT magnitude as weights
        weighted_pixel_l1 = pixel_l1_loss * magnitude_weights
        # Average the weighted error over all pixels (or just hole? Paper isn't explicit. Let's average all.)
        loss_magnitude_weighted = torch.mean(weighted_pixel_l1)


        # --- Combine Losses ---
        total_loss = (self.lambda_dict['lambda_valid'] * loss_valid +
                      self.lambda_dict['lambda_hole'] * loss_hole +
                      self.lambda_dict['lambda_adv'] * loss_adversarial +
                      self.lambda_dict['lambda_perc'] * loss_perceptual +
                      self.lambda_dict['lambda_style'] * loss_style +
                      self.lambda_dict['lambda_mag'] * loss_magnitude_weighted)

        # Store loss component values as floats for logging
        loss_components = {
            'G_loss_total': total_loss.item(),
            'G_loss_valid': loss_valid.item(),
            'G_loss_hole': loss_hole.item(),
            'G_loss_adv': loss_adversarial.item(),
            'G_loss_perc': loss_perceptual.item(),
            'G_loss_style': loss_style.item(),
            'G_loss_mag': loss_magnitude_weighted.item(),
        }

        return total_loss, loss_components

def discriminator_loss(disc_real_pred: torch.Tensor, disc_fake_pred: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Calculates the discriminator loss (standard GAN BCE loss).

    Args:
        disc_real_pred (torch.Tensor): Discriminator output for real spectrograms [B, 1].
        disc_fake_pred (torch.Tensor): Discriminator output for fake spectrograms [B, 1].

    Returns:
        torch.Tensor: Total discriminator loss.
        dict: Dictionary containing individual loss components values (float).
    """
    target_real = torch.ones_like(disc_real_pred)
    target_fake = torch.zeros_like(disc_fake_pred)

    loss_real = F.binary_cross_entropy_with_logits(disc_real_pred, target_real)
    loss_fake = F.binary_cross_entropy_with_logits(disc_fake_pred, target_fake)

    total_loss = (loss_real + loss_fake) / 2

    loss_components = {
        'D_loss_total': total_loss.item(),
        'D_loss_real': loss_real.item(),
        'D_loss_fake': loss_fake.item(),
    }
    return total_loss, loss_components