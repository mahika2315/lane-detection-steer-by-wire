import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution to reduce parameter count and latency on Jetson Nano."""
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)

class ConvBlock(nn.Module):
    """Double standard convolution block for decoder skip-connections."""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class TinyUNet(nn.Module):
    """Lightweight U-Net variant optimized for real-time inference on edge devices."""
    def __init__(self, in_channels=3, num_classes=1):
        super(TinyUNet, self).__init__()
        
        # Encoder (Downsampling)
        # Input size: 3 x 180 x 320
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        ) # Output: 16 x 180 x 320
        
        self.down1 = DepthwiseSeparableConv(16, 32, stride=2)  # Output: 32 x 90 x 160
        self.down2 = DepthwiseSeparableConv(32, 64, stride=2)  # Output: 64 x 45 x 80
        self.down3 = DepthwiseSeparableConv(64, 128, stride=2) # Output: 128 x 22 x 40
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        ) # Output: 128 x 22 x 40

        # Decoder (Upsampling)
        # Note: Bilinear upsampling is highly compatible with ONNX and TensorRT
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) # Output: 128 x 44 x 80 (will crop to 45x80 or pad)
        self.dec_conv3 = ConvBlock(128 + 64, 64)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) # Output: 64 x 90 x 160
        self.dec_conv2 = ConvBlock(64 + 32, 32)
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) # Output: 32 x 180 x 320
        self.dec_conv1 = ConvBlock(32 + 16, 16)
        
        self.final_conv = nn.Conv2d(16, num_classes, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        x_init = self.init_conv(x)      # 16 x 180 x 320
        x1 = self.down1(x_init)         # 32 x 90 x 160
        x2 = self.down2(x1)             # 64 x 45 x 80
        x3 = self.down3(x2)             # 128 x 22 x 40
        
        # Bottleneck
        bn = self.bottleneck(x3)        # 128 x 22 x 40
        
        # Decoder 3
        up3 = self.up3(bn)              # 128 x 44 x 80
        # Dynamic padding or resize to match skip-connection height/width
        if up3.shape[2:] != x2.shape[2:]:
            up3 = F.interpolate(up3, size=x2.shape[2:], mode='bilinear', align_corners=True)
        dec3 = torch.cat([up3, x2], dim=1) # 192 x 45 x 80
        dec3 = self.dec_conv3(dec3)     # 64 x 45 x 80
        
        # Decoder 2
        up2 = self.up2(dec3)            # 64 x 90 x 160
        if up2.shape[2:] != x1.shape[2:]:
            up2 = F.interpolate(up2, size=x1.shape[2:], mode='bilinear', align_corners=True)
        dec2 = torch.cat([up2, x1], dim=1) # 96 x 90 x 160
        dec2 = self.dec_conv2(dec2)     # 32 x 90 x 160
        
        # Decoder 1
        up1 = self.up1(dec2)            # 32 x 180 x 320
        if up1.shape[2:] != x_init.shape[2:]:
            up1 = F.interpolate(up1, size=x_init.shape[2:], mode='bilinear', align_corners=True)
        dec1 = torch.cat([up1, x_init], dim=1) # 48 x 180 x 320
        dec1 = self.dec_conv1(dec1)     # 16 x 180 x 320
        
        logits = self.final_conv(dec1)  # 1 x 180 x 320
        return torch.sigmoid(logits)

if __name__ == "__main__":
    # Test network compilation and shapes
    model = TinyUNet()
    test_input = torch.randn(1, 3, 180, 320)
    test_output = model(test_input)
    print("Input shape:", test_input.shape)
    print("Output shape:", test_output.shape)
    assert test_output.shape == (1, 1, 180, 320), "Shape check failed!"
    print("Model structural check passed successfully.")
