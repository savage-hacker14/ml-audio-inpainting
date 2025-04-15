from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

# ---------------------------------------------------
# Partial Convolution Layer (Based on NVIDIA's paper)
# ---------------------------------------------------
class PartialConv2d(nn.Module):
    """
    Partial Convolution Layer, as described in
    'Image Inpainting for Irregular Holes Using Partial Convolutions' (Liu et al., 2018).

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int or tuple): Size of the convolutional kernel.
        stride (int or tuple): Stride of the convolution.
        padding (int or tuple): Zero-padding added to both sides of the input.
        dilation (int or tuple): Spacing between kernel elements.
        groups (int): Number of blocked connections from input channels to output channels.
        bias (bool): If True, adds a learnable bias to the output.
        multi_channel (bool): If True, compute mask update for each channel separately.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, multi_channel=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.multi_channel = multi_channel

        # Standard convolution layer
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, dilation, groups, bias=False) # Bias handled separately

        # Fixed kernel for mask update (all ones)
        self.mask_kernel = torch.ones(self.out_channels if multi_channel else 1,
                                      self.in_channels, # Match input channels of conv
                                      kernel_size, kernel_size)
        self.mask_conv = nn.Conv2d(in_channels, out_channels if multi_channel else 1,
                                   kernel_size, stride, padding, dilation, groups=groups, bias=False)

        # Initialize mask convolution weights to 1s and make it non-trainable
        self.mask_conv.weight.data.fill_(1.0)
        for param in self.mask_conv.parameters():
            param.requires_grad = False

        # Learnable bias (if requested)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

        # Calculate the window size (sum of weights in the mask kernel)
        self.window_size = float(self.in_channels * kernel_size * kernel_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for Partial Convolution.

        Args:
            x (torch.Tensor): Input feature map (B, C_in, H, W).
            mask (torch.Tensor): Input mask (B, C_in, H, W) or (B, 1, H, W).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (Output feature map, Updated mask)
        """
        if mask.shape[1] == 1 and self.in_channels > 1:
             mask = mask.repeat(1, self.in_channels, 1, 1) # Ensure mask has same channels as input

        masked_x = x * mask

        # Perform standard convolution
        output = self.conv(masked_x)

        # Update the mask
        with torch.no_grad():
            # Convolve the mask with the fixed kernel
            updated_mask = self.mask_conv(mask)

        # Mask ratio calculation (avoid division by zero)
        # Normalization factor: ratio of valid pixels in the receptive field
        mask_ratio = self.window_size / (updated_mask + 1e-8)

        # Normalize the convolution output
        output = output * mask_ratio

        # Add bias if applicable
        if self.bias is not None:
            output = output + self.bias.view(1, self.out_channels, 1, 1)

        # Ensure updated mask is binary (0 or 1) - Clamp might be better than >0
        # A pixel in the updated mask is valid if at least one pixel in its receptive field was valid
        updated_mask = torch.clamp(updated_mask, 0.0, 1.0)
        # Alternative: updated_mask = (updated_mask > 0).float()

        # If multi_channel=False, the updated mask will have 1 channel. Repeat it.
        if not self.multi_channel and updated_mask.shape[1]==1 and self.out_channels > 1:
            updated_mask = updated_mask.repeat(1, self.out_channels, 1, 1)


        return output, updated_mask

# ---------------------------------------------------
# Helper Functions for Padding/Cropping
# ---------------------------------------------------
def get_pad_size(input_size: int, factor: int) -> int:
    """Calculates padding needed for one dimension to be divisible by factor."""
    if input_size % factor == 0:
        return 0
    return factor - (input_size % factor)

def calculate_total_downsampling(layers: nn.ModuleList) -> int:
    """Calculates the total stride product for conv layers with stride > 1."""
    factor = 1
    for layer in layers:
        conv_module = None
        if isinstance(layer, (nn.Conv2d, PartialConv2d)): conv_module = layer
        elif isinstance(layer, nn.Sequential):
            for sub_module in layer:
                if isinstance(sub_module, (nn.Conv2d, PartialConv2d)):
                    conv_module = sub_module; break
        elif hasattr(layer, 'pconv'): conv_module = layer.pconv
        elif hasattr(layer, 'conv'): conv_module = layer.conv

        if conv_module:
            stride = conv_module.stride
            stride_val = stride[0] if isinstance(stride, tuple) else stride
            if stride_val > 1: factor *= stride_val
    return factor

# ---------------------------------------------------
# Encoder/Decoder Blocks
# ---------------------------------------------------
class EncoderBlock(nn.Module):
    """Encoder block: PartialConv -> Norm -> Activation"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 norm_layer=nn.BatchNorm2d, activation=nn.LeakyReLU(0.2, inplace=True)):
        super().__init__()
        self.pconv = PartialConv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False) # Bias=False if using Norm
        self.norm = norm_layer(out_channels) if norm_layer else nn.Identity()
        self.activation = activation if activation else nn.Identity()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x, mask = self.pconv(x, mask)
        x = self.norm(x)
        x = self.activation(x)
        return x, mask

class DecoderBlock(nn.Module):
    """Decoder block: PartialConv -> Norm -> Activation"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 norm_layer=nn.BatchNorm2d, activation=nn.LeakyReLU(0.2, inplace=True)):
        super().__init__()
        # Bias=False if using Norm; PConv needs channels after concat
        self.pconv = PartialConv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.norm = norm_layer(out_channels) if norm_layer else nn.Identity()
        self.activation = activation if activation else nn.Identity()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x, mask = self.pconv(x, mask)
        x = self.norm(x)
        x = self.activation(x)
        return x, mask

# ---------------------------------------------------
# Generator Network (U-Net with Partial Convolutions)
# ---------------------------------------------------
class PConvUNet(nn.Module):
    """
    Generator Network based on U-Net with Partial Convolutions.
    Uses EncoderBlock/DecoderBlock and padding/cropping.
    """
    def __init__(self, input_channels=1, mask_channels=1, output_channels=1,
                 enc_layer_cfg=[ # (out_c, kernel, stride, padding)
                     (64, 7, 2, 3), (128, 5, 2, 2), (256, 5, 2, 2),
                     (512, 3, 2, 1), (512, 3, 2, 1), (512, 3, 2, 1), (512, 3, 2, 1)
                 ],
                 dec_layer_cfg=[ # (out_c, kernel, stride, padding) - Applied AFTER concat
                     (512, 3, 1, 1), (512, 3, 1, 1), (512, 3, 1, 1),
                     (256, 3, 1, 1), (128, 3, 1, 1), (64, 3, 1, 1)
                 ],
                 final_dec_cfg={ # Final layer config
                     'interim_ch': 64, 'out_ch': 1, 'kernel': 3, 'padding': 1
                 },
                 norm_layer=nn.BatchNorm2d,
                 activation=nn.LeakyReLU(0.2, inplace=True),
                 final_activation=nn.Tanh(),
                 upsample_mode='nearest'):
        super(PConvUNet, self).__init__()
        
        if input_channels != 1 or mask_channels != 1:
             print("Warning: This implementation assumes input_channels=1 and mask_channels=1 for skip connection logic.")

        self.input_channels = input_channels
        self.mask_channels = mask_channels
        
        self.upsample = nn.Upsample(scale_factor=2, mode=upsample_mode)
        self.final_activation = final_activation if final_activation else nn.Identity()

        # --- Build Encoder ---
        self.encoder_blocks = nn.ModuleList()
        
        in_c = input_channels + mask_channels
        self.enc_output_channels = [] # Store output channels for decoder concat planning
        for out_c, kernel, stride, padding in enc_layer_cfg:
            self.encoder_blocks.append(
                EncoderBlock(in_c, out_c, kernel, stride, padding, norm_layer, activation)
            )
            self.enc_output_channels.append(out_c)
            in_c = out_c

        self._total_downsampling = calculate_total_downsampling(self.encoder_blocks)

        # --- Build Decoder ---
        self.decoder_blocks = nn.ModuleList()
        skip_channels_rev = self.enc_output_channels[::-1]  # Full reversed list [c7, c6, ..., c1]
        upsampled_channels = skip_channels_rev[0]           # Bottleneck channels (output of last encoder)

        for i, (out_c, kernel, stride, padding) in enumerate(dec_layer_cfg):
            skip_c = skip_channels_rev[i+1]
            in_c_decoder = upsampled_channels + skip_c
            self.decoder_blocks.append(
                DecoderBlock(in_c_decoder, out_c, kernel, stride, padding, norm_layer, activation)
            )
            upsampled_channels = out_c # Output of this block is input (after upsample) to next

        # --- Build Final Decoder Layer ---
        final_interim_ch = final_dec_cfg['interim_ch']
        final_out_ch = final_dec_cfg['out_ch']
        final_kernel = final_dec_cfg['kernel']
        final_padding = final_dec_cfg['padding']
        # Input to final: upsampled from last decoder block + **original input channels**
        in_c_final = upsampled_channels + self.input_channels # Use original input_channels (e.g., 1)

        self.final_decoder_layer = nn.Sequential(
            PartialConv2d(in_c_final, final_interim_ch, final_kernel, 1, final_padding, bias=True),
            activation if activation else nn.Identity(),
            PartialConv2d(final_interim_ch, final_out_ch, final_kernel, 1, final_padding, bias=True)
        )


    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """ Forward pass with padding/cropping and modular blocks. """
        # --- Input Validation & Padding ---
        if x.shape[1] != self.input_channels: raise ValueError(f"Input x channels ({x.shape[1]}) != expected ({self.input_channels})")
        if mask.shape[1] != self.mask_channels: raise ValueError(f"Input mask channels ({mask.shape[1]}) != expected ({self.mask_channels})")
        if x.shape[2:] != mask.shape[2:]: raise ValueError("x and mask spatial dims must match")

        _, _, H_in, W_in = x.shape
        pad_H = get_pad_size(H_in, self._total_downsampling)
        pad_W = get_pad_size(W_in, self._total_downsampling)
        input_padding = (0, pad_W, 0, pad_H) # (Left, Right, Top, Bottom)

        # Pad features with reflection, mask with 1 (valid)
        x_padded = F.pad(x, input_padding, mode='reflect')
        mask_padded = F.pad(mask, input_padding, mode='constant', value=1.0)

        # --- Encoder ---
        enc_features = [] # Store features for skip connections
        enc_masks = []    # Store corresponding masks
        current_feat = torch.cat([x_padded, mask_padded], dim=1) # Initial input (C=2)
        current_mask = mask_padded # Start with the single-channel padded mask

        for block in self.encoder_blocks:
            # EncoderBlock handles PConv + Norm + Activation
            current_feat, current_mask = block(current_feat, current_mask)
            enc_features.append(current_feat)
            enc_masks.append(current_mask)

        # --- Decoder ---
        # Start with bottleneck features/mask
        dec_feat = enc_features[-1]
        dec_mask = enc_masks[-1]

        for i, block in enumerate(self.decoder_blocks):
            # Upsample features and mask from previous decoder stage
            dec_feat = self.upsample(dec_feat)
            dec_mask = self.upsample(dec_mask)

            # Get skip features/mask from corresponding encoder stage (reversed index)
            skip_idx = len(enc_features) - 2 - i # -1 is bottleneck, -2 is first skip etc.
            skip_feat = enc_features[skip_idx]
            skip_mask = enc_masks[skip_idx]

            # Align sizes (optional but safe)
            if dec_feat.shape[2:] != skip_feat.shape[2:]:
                # print(f"Warning: Size mismatch at decoder block {i}. Interpolating.")
                dec_feat = F.interpolate(dec_feat, size=skip_feat.shape[2:], mode='nearest')
                dec_mask = F.interpolate(dec_mask, size=skip_mask.shape[2:], mode='nearest')

            # Concatenate features and masks
            dec_feat_cat = torch.cat([dec_feat, skip_feat], dim=1)
            dec_mask_cat = torch.cat([dec_mask, skip_mask], dim=1) # Mask corresponds to features

            # Pass through decoder block
            # DecoderBlock handles PConv + Norm + Activation
            dec_feat, dec_mask = block(dec_feat_cat, dec_mask_cat)

        # --- Final Layer ---
        # Upsample last decoder output
        dec_feat = self.upsample(dec_feat)
        dec_mask = self.upsample(dec_mask)

        skip_feat_final = x_padded
        skip_mask_final = mask_padded
        
        if dec_feat.shape[2:] != skip_feat_final.shape[2:]:
            raise RuntimeError(f"Size mismatch before final layer. "
                                f"Dec: {dec_feat.shape[2:]}, Skip: {skip_feat_final.shape[2:]}")

        # Concatenate features and masks
        dec_feat_cat_final = torch.cat([dec_feat, skip_feat_final], dim=1)
        dec_mask_cat_final = torch.cat([dec_mask, skip_mask_final], dim=1)

        # Pass through final decoder sequence (PConv -> Act -> PConv)
        pconv1_layer = self.final_decoder_layer[0]
        act_layer = self.final_decoder_layer[1]
        pconv2_layer = self.final_decoder_layer[2]

        dec_feat, mask_after_pconv1 = pconv1_layer(dec_feat_cat_final, dec_mask_cat_final)
        dec_feat = act_layer(dec_feat)
        # Use mask from pconv1 as input mask for pconv2
        dec_feat, _ = pconv2_layer(dec_feat, mask_after_pconv1)

        # Apply final activation (e.g., Tanh)
        dec_feat = self.final_activation(dec_feat)

        # --- Output Cropping ---
        final_output = dec_feat[:, :, :H_in, :W_in]

        if final_output.shape[2:] != (H_in, W_in):
            # Add more debug info if this still fails
            print("Input shape:", x.shape)
            print("Padded shape:", x_padded.shape)
            print("Bottleneck shape:", enc_features[-1].shape)
            print("Shape before final PConv1:", dec_feat_cat_final.shape)
            print("Shape after final PConv2 (before activation):", dec_feat.shape) # Check shape before final act
            raise RuntimeError(f"Cropping failed. Expected {(H_in, W_in)}, got {final_output.shape[2:]}")

        return final_output


# ---------------------------------------------------
# Discriminator Network
# ---------------------------------------------------

class DiscriminatorBlock(nn.Module):
    """Discriminator block: Conv -> [Norm] -> Activation"""
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1,
                 use_spectral_norm=True, activation=nn.LeakyReLU(0.2, inplace=True),
                 use_norm=False): # Typically False for PatchGAN discriminators
        super().__init__()
        bias = not use_norm # Add bias if not using normalization layer
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        if use_spectral_norm:
            conv = spectral_norm(conv)

        # Norm layer (Optional, usually omitted in PatchGAN D)
        norm = nn.BatchNorm2d(out_channels) if use_norm else nn.Identity()

        self.block = nn.Sequential(
            conv,
            norm,
            activation if activation else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class Discriminator(nn.Module):
    """
    Discriminator Network using DiscriminatorBlocks.
    Predicts realism patch-wise.
    """
    def __init__(self, input_channels=1,
                 layer_cfg=[ # (out_c, stride, use_norm)
                     (64, 2, False), (128, 2, False), (256, 2, False), (512, 1, False)
                 ],
                 final_out_channels=1, kernel_size=4, padding=1, # Kernel/padding for blocks & final
                 use_spectral_norm=True,
                 activation=nn.LeakyReLU(0.2, inplace=True)):
        super().__init__()
        layers = []
        in_c = input_channels

        # Build main blocks
        for out_c, stride, use_norm in layer_cfg:
            layers.append(
                DiscriminatorBlock(
                    in_c, out_c, kernel_size, stride, padding,
                    use_spectral_norm, activation, use_norm
                )
            )
            in_c = out_c

        # Add final convolutional layer (stride 1)
        final_conv = nn.Conv2d(in_c, final_out_channels, kernel_size, stride=1, padding=padding, bias=True)
        if use_spectral_norm:
            final_conv = spectral_norm(final_conv)
        layers.append(final_conv)

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)