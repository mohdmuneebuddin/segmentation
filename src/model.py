import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class ConvAutoencoder(nn.Module):
    """
    A bottleneck convolutional autoencoder for anomaly detection.

    KEY DESIGN CHOICE: no skip connections.
    Skip connections would let the network copy defective pixels straight from
    input to output, which defeats the purpose of reconstruction-based anomaly
    detection. We force all information through a narrow bottleneck so the
    network can only learn to reconstruct the *distribution* of normal images.
    """

    def __init__(self, in_channels=3, out_channels=3, features=(64, 128, 256, 512)):
        super().__init__()

        # ---- Encoder ----
        encoder_layers = []
        prev_c = in_channels
        for f in features:
            encoder_layers.append(DoubleConv(prev_c, f))
            encoder_layers.append(nn.MaxPool2d(2, 2))
            prev_c = f
        self.encoder = nn.Sequential(*encoder_layers)

        # ---- Bottleneck ----
        self.bottleneck = DoubleConv(features[-1], 2 * features[-1])

        # ---- Decoder ----
        decoder_layers = []
        prev_c = 2 * features[-1]
        for f in reversed(features):
            decoder_layers.append(
                nn.ConvTranspose2d(prev_c, f, kernel_size=2, stride=2)
            )
            decoder_layers.append(DoubleConv(f, f))
            prev_c = f
        self.decoder = nn.Sequential(*decoder_layers)

        # Final 1x1 to project back to image channels.
        # No activation: inputs are normalized (can be negative), so outputs
        # must be free to take any real value.
        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return self.final(x)
