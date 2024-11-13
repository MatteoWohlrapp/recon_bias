import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import resnet50, ResNet50_Weights

class ResNetEncoder(nn.Module):
    def __init__(self, z_dim=512, resnet_version='resnet50', pretrained=True):
        super(ResNetEncoder, self).__init__()
        # Load a pre-trained ResNet model (you can choose resnet18, resnet34, resnet50, etc.)
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            *list(resnet.children())[1:-1]  # Remaining layers of ResNet
        )
        
        # Add a fully connected layer to map to the desired latent space dimension (z_dim)
        self.fc = nn.Linear(resnet.fc.in_features, z_dim)

    def forward(self, x):
        x = self.feature_extractor(x)  # Get features from ResNet
        x = x.view(x.size(0), -1)  # Flatten the feature map
        return self.fc(x)  # Map to latent vector z

class MappingNetwork(nn.Module):
    def __init__(self, z_dim, w_dim, num_layers=8):
        super(MappingNetwork, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(z_dim, w_dim))
            layers.append(nn.ReLU())
        self.mapping = nn.Sequential(*layers)

    def forward(self, z):
        z = z / z.norm(dim=-1, keepdim=True)  # Normalize z for stable training
        return self.mapping(z)  # Output w

class StyleModulationLayer(nn.Module):
    def __init__(self, w_dim, channels):
        super(StyleModulationLayer, self).__init__()
        self.fc = nn.Linear(w_dim, channels)

    def forward(self, w):
        return self.fc(w)

class AdaIN(nn.Module):
    def __init__(self, channels, w_dim):
        super(AdaIN, self).__init__()
        self.style_scale = StyleModulationLayer(w_dim, channels)
        self.style_bias = StyleModulationLayer(w_dim, channels)

    def forward(self, x, w):
        scale = self.style_scale(w).unsqueeze(2).unsqueeze(3)
        bias = self.style_bias(w).unsqueeze(2).unsqueeze(3)
        # Normalize x, then apply scale and bias from style modulation
        x = F.instance_norm(x)
        return scale * x + bias

class GeneratorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, w_dim):
        super(GeneratorBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.adain1 = AdaIN(out_channels, w_dim)
        self.adain2 = AdaIN(out_channels, w_dim)
        
    def forward(self, x, w):
        x = F.leaky_relu(self.adain1(self.conv1(x), w))
        x = F.leaky_relu(self.adain2(self.conv2(x), w))
        return x

class Generator(nn.Module):
    def __init__(self, w_dim, start_channels=512, num_blocks=5):
        super(Generator, self).__init__()
        self.initial = nn.Parameter(torch.randn(1, start_channels, 4, 4))
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            out_channels = max(start_channels // (2 ** i), 1)
            self.blocks.append(GeneratorBlock(start_channels, out_channels, w_dim))
            start_channels = out_channels
        self.to_grayscale = nn.Conv2d(start_channels, 1, 1)

    def forward(self, w):
        x = self.initial.repeat(w.size(0), 1, 1, 1)
        for block in self.blocks:
            x = block(x, w)
            x = F.interpolate(x, scale_factor=2)  # Upsample for higher resolution
        x = F.interpolate(x, size=(240, 240), mode='bilinear', align_corners=False)
        output = torch.sigmoid(self.to_grayscale(x))  # Convert to single-channel grayscale output
        return output

class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DiscriminatorBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2, inplace=False)  # Ensure inplace=False
        x = F.leaky_relu(self.conv2(x), 0.2, inplace=False)  # Ensure inplace=False
        return F.avg_pool2d(x, 2)  # Downsample

class Discriminator(nn.Module):
    def __init__(self, start_channels=32, max_channels=512, num_blocks=6):
        super(Discriminator, self).__init__()
        self.blocks = nn.ModuleList()
        in_channels = 1  # Single-channel input for grayscale images
        current_channels = start_channels

        for i in range(num_blocks):
            out_channels = min(current_channels * 2, max_channels)
            self.blocks.append(DiscriminatorBlock(in_channels, out_channels))
            in_channels = out_channels
            current_channels = out_channels

        # Calculate final flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 240, 240)
            x = dummy_input
            for block in self.blocks:
                x = block(x)
            flattened_size = x.numel()

        self.fc = nn.Linear(flattened_size, 1)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.fc(x)

class ResNetDiscriminator(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNetDiscriminator, self).__init__()
        # Load a pretrained ResNet model
        self.resnet = models.resnet18(pretrained=pretrained)

        # Modify the first convolutional layer to accept 1 input channel
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Replace the fully connected layer with a single output
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 1)

    def forward(self, x):
        return self.resnet(x)
    
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class SimpleAutoencoder(nn.Module):
    def __init__(self):
        super(SimpleAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        # Decoder with Upsample and Convolution instead of ConvTranspose
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # To ensure output is in [0, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



class StyleGAN(nn.Module):
    def __init__(self, z_dim=512, w_dim=512, num_blocks=5, start_channels=512):
        super(StyleGAN, self).__init__()
        self.model = SimpleAutoencoder()
        self.encoder = ResNetEncoder(z_dim)
        self.mapping = MappingNetwork(z_dim, w_dim)
        self.generator = Generator(w_dim, start_channels, num_blocks)
        self.generator.apply(weights_init)
        self.discriminator = ResNetDiscriminator()

        self.optimizer_e = torch.optim.Adam(self.encoder.parameters(), lr=1e-4)
        self.optimizer_m = torch.optim.Adam(self.mapping.parameters(), lr=1e-4)
        self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=1e-4)
        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4)
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def discriminator_step(self, loss):
        self.optimizer_d.zero_grad()
        loss.backward()
        self.optimizer_d.step()

    def generator_step(self, loss):
        self.optimizer_e.zero_grad()
        self.optimizer_m.zero_grad()
        self.optimizer_g.zero_grad()
        self.model_optimizer.zero_grad()
        loss.backward()
        self.optimizer_g.step()
        self.optimizer_m.step()
        self.optimizer_e.step()
        self.model_optimizer.step()

    def forward(self, x):
        z = self.encoder(x)
        w = self.mapping(z)
        fake = self.generator(w)
        return fake, self.discriminator(fake), self.discriminator(x)
        #return self.model(x), self.discriminator(x), self.discriminator(x)