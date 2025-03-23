import torch
import torch.nn as nn
import torch.nn.functional as F

class PartialConv2d(nn.Module):
    """
    Partial Convolution Layer.
    
    Implements inpainting specific time-frequency (T-F) areas of spectrograms.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(PartialConv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias)
        
        self.mask_conv = nn.Conv2d(
            1,1,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False
        )
        
        with torch.no_grad():
            self.mask_conv.weight.fill_(1.0)
            self.mask_conv.weight.requires_grad = False
        
        self.input_conv = None
        self.mask_ratio = None
        
    def forward(self, input, mask):
        """
        Forward pass of partial convolution layer.

        Args:
            input (): Input feature map [B x C x H x W]
            mask (): Binary mask [B x 1 x H x W] (0: hole, 1: valid)
            
        Returns:
            output (): Convolved feature map [B x C x H x W]
            mask (): Updated binary mask
        """
        
        out = self.conv(input * mask)
        
        with torch.no_grad():
            updated_mask = self.mask_conv(mask)
            
            mask_ratio = updated_mask.clamp(min=1e-8)
            out = out / mask_ratio
            updated_mask = (updated_mask > 0).type_as(mask)
        
        return out, updated_mask
        
    
class EncoderBlock(nn.Module):
    """
    Encoder block for the U-Net architecture.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=2, padding=1):
        super(EncoderBlock, self).__init__()
        self.partial_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        
    def forward(self, x, mask):
        out, updated_mask = self.partial_conv(x, mask)
        out = self.bn(out)
        out = self.relu(out)
        return out, updated_mask

class DecoderBlock(nn.Module):
    """
    Decoder block of the U-Net architecture with partial convolutions and skip connections.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=2, padding=1):
        super(DecoderBlock, self).__init__()
        self.partial_conv = PartialConv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        
    def forward(self, x, mask, skip_feature=None, skip_mask=None):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        mask = F.interpolate(mask, scale_factor=2, mode='nearest')
        
        if skip_feature is not None and skip_mask is not None:
            x = torch.cat([x, skip_feature], dim=1)
            mask = torch.cat([mask, skip_mask], dim=1)
            
            mask = mask[:, 0:1, :, :]
            
        out, updated_mask = self.partial_conv(x, mask)
        out = self.bn(out)
        out = self.relu(out)
        return out, updated_mask
    
class Generator(nn.Module):
    """
    U-Net Generator with partial convolutions for speech inpainting.

    Seven layer deep enccoder and decoder with skip connections to preserve specital detailings.
    """
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder1 = EncoderBlock(1+1, 64, kernel_size=7, stride=2, padding=3)
        self.encoder2 = EncoderBlock(64, 128, kernel_size=5, stride=2, padding=2)
        self.encoder3 = EncoderBlock(128, 256, kernel_size=5, stride=2, padding=2)
        self.encoder4 = EncoderBlock(256, 512, kernel_size=3, stride=2, padding=1)
        self.encoder5 = EncoderBlock(512, 512, kernel_size=3, stride=2, padding=1)
        self.encoder6 = EncoderBlock(512, 512, kernel_size=3, stride=2, padding=1)
        self.encoder7 = EncoderBlock(512, 512, kernel_size=3, stride=2, padding=1)
        
        self.decoder1 = DecoderBlock(512, 512, kernel_size=3, stride=1, padding=1)
        self.decoder2 = DecoderBlock(512+512, 512, kernel_size=3, stride=1, padding=1)
        self.decoder3 = DecoderBlock(512+512, 512, kernel_size=3, stride=1, padding=1)
        self.decoder4 = DecoderBlock(512+512, 256, kernel_size=3, stride=1, padding=1)
        self.decoder5 = DecoderBlock(256+256, 128, kernel_size=3, stride=1, padding=1)
        self.decoder6 = DecoderBlock(128+128, 64, kernel_size=3, stride=1, padding=1)
        
        self.final_conv = nn.Conv2d(64+64, 1, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x, mask):
        """

        Args:
            x (_type_): Input spectrogram [B x 1 x H x W]
            mask (_type_): Binary mask [B x 1 x H x W] (0: hole, 1: valid)
            
        Returns:
            output (_type_): Inpainted spectrogram [B x 1 x H x W]
            updated_mask (_type_): Updated binary mask
        """
        
        x_with_mask = torch.cat([x, mask], dim=1)
        
        e1, m1 = self.encoder1(x_with_mask, mask)
        e2, m2 = self.encoder2(e1, m1)
        e3, m3 = self.encoder3(e2, m2)
        e4, m4 = self.encoder4(e3, m3)
        e5, m5 = self.encoder5(e4, m4)
        e6, m6 = self.encoder6(e5, m5)
        e7, m7 = self.encoder7(e6, m6)
        
        d7, dm7 = self.decoder1(e7, m7)
        d6, dm6 = self.decoder2(d7, dm7, e6, m6)
        d5, dm5 = self.decoder3(d6, dm6, e5, m5)
        d4, dm4 = self.decoder4(d5, dm5, e4, m4)
        d3, dm3 = self.decoder5(d4, dm4, e3, m3)
        d2, dm2 = self.decoder6(d3, dm3, e2, m2)
        
        out, updated_mask = self.final_conv(torch.cat([d2, e1], dim=1), dm2)
        
        return out, updated_mask
    
class SpectralNormConv2d(nn.Module):
    """
    Spectral Normalization for Conv2d layers.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(SpectralNormConv2d, self).__init__()
        self.conv = nn.utils.spectral_norm(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=bias
            )
        )
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.leaky_relu(x)
        return x

class Discriminator(nn.Module):
    """
    Discriminator network for the GAN.
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.conv1 = SpectralNormConv2d(1, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = SpectralNormConv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = SpectralNormConv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv4 = SpectralNormConv2d(256, 512, kernel_size=4, stride=1, padding=1)
        self.conv5 = nn.utils.spectral_norm(
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )
    def forward(self, x):
        """
        Args:
            x (_type_): Input spectrogram [B x 1 x H x W]
        
        Returns:
            output (_type_): Discriminator output [B x 1 x H x W]
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x
    
class SpeechVGGLoss(nn.Module):
    """
    Speech VGG Loss for the GAN.
    
    Perceptual loss and style loss using a pretrained speech VGG model.
    """
    def __init__(self, speech_vgg):
        super(SpeechVGGLoss, self).__init__()
        self.speech_vgg = speech_vgg
        
        for param in self.speech_vgg.parameters():
            param.requires_grad = False
    def forward(self, composed_output, ground_truth, mask):
        """
        Calculate VGG perceptual and style loss.

        Args:
            composed_output (_type_): Inpainted and composited spectrogram [B x 1 x H x W]
            ground_truth (_type_): Ground truth spectrogram [B x 1 x H x W]
            mask (_type_): Binary mask [B x 1 x H x W] (0: hole, 1: valid)
            
        Returns:
            perceptual_loss (_type_): Perceptual loss
            style_loss (_type_): Style loss
        """
        
        comp_features = self.speech_vgg(composed_output)
        gt_features = self.speech_vgg(ground_truth)
        
        perceptual_loss = 0
        for i in range(len(comp_features)):
            perceptual_loss += F.l1_loss(comp_features[i], gt_features[i])
        style_loss = 0
        for i in range(len(comp_features)):
            b, c, h, w = comp_features[i].size()
            comp_feat = self.gram_matrix(comp_features[i])
            gt_feat = self.gram_matrix(gt_features[i])
            
            comp_gram = torch.bmm(comp_feat, comp_feat.transpose(1, 2)) / (c * h * w)
            gt_gram = torch.bmm(gt_feat, gt_feat.transpose(1, 2)) / (c * h * w)
            
            style_loss += F.l1_loss(comp_gram, gt_gram)
            
        return perceptual_loss, style_loss

class GANLoss(nn.Module):
    """
    GAN loss for the GAN.
    
    Adversarial loss for the generator and discriminator.
    """
    def __init__(self):
        super(GANLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss
        self.l1_loss = nn.L1Loss()
        self.speech_vgg_loss = SpeechVGGLoss()
    
    def generator_loss(self, discriminator_output, composed_output, ground_truth, mask, raw_output):
        """
        Calculate generator loss
        
        Lgen = 0.01LB + Lv + 2Lh + 4Lp + 500Ls + 0.2Lw

        Args:
            discriminator_output (_type_): Discriminator output [B x 1 x H x W] for fake (inpainted) spectrogram
            composed_output (_type_): Inpainted and composited spectrogram [B x 1 x H x W]
            ground_truth (_type_): Ground truth spectrogram [B x 1 x H x W]
            mask (_type_): Binary mask [B x 1 x H x W] (0: hole, 1: valid)
            raw_output (_type_): Raw generator output [B x 1 x H x W] (before compositing)
            
        Returns:
            loss (_type_): Generator loss
            loss_dict (_type_): Dictionary of individual losses
        """
        bce_loss = self.bce_loss(discriminator_output, torch.ones_like(discriminator_output))
        valid_mean = mask.mean()
        valid_loss = self.l1_loss(composed_output * mask, ground_truth * mask) / (valid_mean)
        
        inv_mask = 1 - mask
        hole_mean = inv_mask.mean()
        hole_loss = self.l1_loss(composed_output * inv_mask, ground_truth * inv_mask) / (hole_mean)
        
        perceptual_loss, style_loss = self.speech_vgg_loss(composed_output, ground_truth, mask)
        
        weight_loss = torch.mean(torch.abs(ground_truth) * torch.abs(composed_output - ground_truth))
        
        loss = (0.01 * bce_loss + 
                valid_loss + 
                2 * hole_loss + 
                4 * perceptual_loss + 
                500 * style_loss + 
                0.2 * weight_loss
        )
        
        loss_dict = {
            "bce_loss": bce_loss,
            "valid_loss": valid_loss,
            "hole_loss": hole_loss,
            "perceptual_loss": perceptual_loss,
            "style_loss": style_loss,
            "weight_loss": weight_loss
        }
        
        return loss, loss_dict
    
    def discriminator_loss(self, real_output, fake_output):
        """
        Calculate discriminator loss
        Lad = (E(disgt, Real) + E(discomp, Fake)) / 2
        
        Args:
            real_output (_type_): Discriminator output [B x 1 x H x W] for real spectrogram
            fake_output (_type_): Discriminator output [B x 1 x H x W] for fake (inpainted) spectrogram

        Returns:
           loss (_type_): Discriminator loss
        """
        
        real_loss = self.bce_loss(real_output, torch.ones_like(real_output))
        fake_loss = self.bce_loss(fake_output, torch.zeros_like(fake_output))
        
        loss = (real_loss + fake_loss) / 2
        
        return loss
    
class SpeechInpaintingGAN:
    """
    Main class
    
    Controls and manages the GAN training process.
    - Generator
    - Discriminator
    - Loss function
    """
    def __init__(self, speech_vgg, device="cuda"):
        self.device = device
        self.generator = Generator().to(device)
        self.discriminator = Discriminator().to(device)
        self.gan_loss = GANLoss(speech_vgg).to(device)
        
        self.gan_optim = torch.optim.Adam(self.generator.parameters(), lr=0.02, betas=(0.9, 0.999))
        self.disc_optim = torch.optim.Adam(self.discriminator.parameters(), lr=5e-5, betas=(0.9, 0.999))
        
        self.gan_scheduler = torch.optim.lr_scheduler.StepLR(self.gan_optim, step_size=200, gamma=0.96)
        self.disc_scheduler = torch.optim.lr_scheduler.StepLR(self.disc_optim, step_size=200, gamma=0.96)
        
    def train_step(self, spectrograms, masks):
        """
        Perform one training step
        
        Args:
            spectrograms (_type_): Input spectrograms [B x 1 x H x W]
            masks (_type_): Binary masks [B x 1 x H x W] (0: hole, 1: valid)
            
        Returns:
            loss_dict (_type_): Dictionary of losses
        """
        spectrograms = spectrograms.to(self.device)
        masks = masks.to(self.device)
        
        impaired_spectrograms = spectrograms * masks
        self.disc_optim.zero_grad()
        
        inpainted_spectrograms, _ = self.generator(impaired_spectrograms, masks)
        composed = inpainted_spectrograms * masks + spectrograms * (1 - masks)
        
        disc_real = self.discriminator(spectrograms)
        disc_fake = self.discriminator(composed.detach())
        
        disc_loss = self.gan_loss.discriminator_loss(disc_real, disc_fake)
        disc_loss.backward()
        self.disc_optim.step()
        
        self.gan_optim.zero_grad()
        
        disc_fake = self.discriminator(composed)
        
        gen_loss, loss_dict = self.gan_loss.generator_loss(
            disc_fake, composed, spectrograms, masks, inpainted_spectrograms
        )
        gen_loss.backward()
        self.gan_optim.step()
        
        loss_dict["disc_loss"] = disc_loss
        
        return loss_dict
    
    def inpaint(self, spectrograms, masks):
        """
        Inpaint spectrograms
        
        Args:
            spectrograms (_type_): Input spectrograms [B x 1 x H x W]
            masks (_type_): Binary masks [B x 1 x H x W] (0: hole, 1: valid)
            
        Returns:
            inpainted_spectrograms (_type_): Inpainted spectrograms [B x 1 x H x W]
        """
        self.generator.eval()
        with torch.no_grad():
            spectrograms = spectrograms.to(self.device)
            masks = masks.to(self.device)
            
            impaired_spectrograms = spectrograms * masks
            inpainted_spectrograms, _ = self.generator(impaired_spectrograms, masks)
            composed = inpainted_spectrograms * masks + spectrograms * (1 - masks)
        return composed