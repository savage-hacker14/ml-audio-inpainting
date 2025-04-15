import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights
from typing import List, Dict, Tuple

class VGGLoss(nn.Module):
    """
    Calculates Perceptual (Feature) and Style losses using pretrained VGG19.

    Args:
        device (torch.device): Device to run the VGG model on.
        layer_indices_style (List[int]): Indices of VGG layers for style loss.
        layer_indices_perceptual (List[int]): Indices of VGG layers for perceptual loss.
    """
    def __init__(self, device: torch.device,
                 layer_indices_style: List[int] = [0, 5, 10, 19, 28],
                 layer_indices_perceptual: List[int] = [2, 7, 12, 21, 30]):
        super(VGGLoss, self).__init__()
        print("Initializing VGG Loss...")
        weights = VGG19_Weights.DEFAULT
        vgg = vgg19(weights=weights).features.to(device).eval()
        print("VGG19 loaded.")

        # --- Freeze VGG ---
        for param in vgg.parameters():
            param.requires_grad = False

        self.vgg_layers = vgg
        self.layer_indices_style = set(layer_indices_style)
        self.layer_indices_perceptual = set(layer_indices_perceptual)
        self.max_layer_idx = max(layer_indices_style + layer_indices_perceptual) if layer_indices_style or layer_indices_perceptual else -1

        # --- Use normalization transform associated with the weights ---
        self.preprocess = weights.transforms()
        print("Using VGG normalization transforms provided by weights.")

        self.l1_loss = nn.L1Loss()
        print("VGG Loss Initialized.")


    def _extract_features(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Extract features from specified VGG layers."""
        features = {}
        # Iterate through VGG layers up to the max needed index
        for i, layer in enumerate(self.vgg_layers):
            x = layer(x)
            if i in self.layer_indices_style or i in self.layer_indices_perceptual:
                features[i] = x
            if i >= self.max_layer_idx:
                break # No need to go further
        return features

    def _gram_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate Gram matrix."""
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2)) # Batch matrix multiplication
        # Normalize by number of elements in feature map
        return gram.div(c * h * w)


    def _prepare_input_for_vgg(self, x: torch.Tensor, is_generated: bool) -> torch.Tensor:
        """Prepares spectrogram tensor for VGG input BEFORE normalization."""
        # 1. Ensure input is (B, C, H, W)
        if x.dim() == 3: x = x.unsqueeze(1)
        if x.shape[1] != 1:
             # If input is already 3 channels, assume it's correctly formatted
             if x.shape[1] == 3: return x
             else: raise ValueError(f"Input tensor must have 1 or 3 channels, got {x.shape[1]}")

        if is_generated:
            x_scaled = (x + 1.0) / 2.0
        else: 
            x_clamped = torch.clamp(x, min=0.0)
            max_val = torch.max(x_clamped).item() + 1e-6
            x_scaled = x_clamped / max_val if max_val > 1e-5 else x_clamped

        x_scaled = torch.clamp(x_scaled, 0.0, 1.0) 

        # 3. Repeat channel to get 3 channels
        x_repeated = x_scaled.repeat(1, 3, 1, 1)

        return x_repeated


    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate perceptual and style loss.

        Args:
            generated (torch.Tensor): Generated spectrogram (B, 1, H, W), range [-1, 1].
            target (torch.Tensor): Target spectrogram (B, 1, H, W), log1p normalized.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (perceptual_loss, style_loss)
        """
        # --- Prepare inputs (scale to [0,1], repeat channels) ---
        generated_prep = self._prepare_input_for_vgg(generated, is_generated=True)
        target_prep = self._prepare_input_for_vgg(target, is_generated=False)

        # --- Apply VGG preprocessing (normalization) ---
        generated_norm = self.preprocess(generated_prep)
        target_norm = self.preprocess(target_prep)

        # --- Extract Features ---
        features_gen = self._extract_features(generated_norm)
        features_target = self._extract_features(target_norm)

        # --- Calculate Perceptual Loss ---
        perceptual_loss = torch.tensor(0.0, device=generated.device)
        valid_p_layers = 0
        for i in self.layer_indices_perceptual:
            if i in features_gen and i in features_target:
                 perceptual_loss += self.l1_loss(features_gen[i], features_target[i])
                 valid_p_layers += 1
        if valid_p_layers > 0: perceptual_loss /= valid_p_layers

        # --- Calculate Style Loss ---
        style_loss = torch.tensor(0.0, device=generated.device)
        valid_s_layers = 0
        for i in self.layer_indices_style:
             if i in features_gen and i in features_target:
                 gram_gen = self._gram_matrix(features_gen[i])
                 gram_target = self._gram_matrix(features_target[i])
                 style_loss += self.l1_loss(gram_gen, gram_target)
                 valid_s_layers += 1
        if valid_s_layers > 0: style_loss /= valid_s_layers

        return perceptual_loss, style_loss