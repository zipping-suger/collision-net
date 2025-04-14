import torch
from torch import nn
import pytorch_lightning as pl

class PointNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self._build_model()

    def _build_model(self):
        # MLP layers for processing each point
        self.mlp = nn.Sequential(
            nn.Conv1d(4, 64, 1),  # Process x, y, z, and segmentation_mask (4 channels)
            nn.GroupNorm(16, 64),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(64, 64, 1),
            nn.GroupNorm(16, 64),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(64, 64, 1),
            nn.GroupNorm(16, 64),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.GroupNorm(16, 128),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(128, 1024, 1),  # Final feature dimension per point
            nn.GroupNorm(16, 1024),
            nn.LeakyReLU(inplace=True),
        )

        # Fully connected layers for global feature processing
        self.fc_layer = nn.Sequential(
            nn.Linear(1024, 4096),
            nn.GroupNorm(16, 4096),
            nn.LeakyReLU(inplace=True),
            nn.Linear(4096, 2048),
            nn.GroupNorm(16, 2048),
            nn.LeakyReLU(inplace=True),
            nn.Linear(2048, 2048),
        )

    def forward(self, point_cloud: torch.Tensor) -> torch.Tensor:
        # Input shape: (B, N, 4) → transpose to (B, 4, N) for Conv1d
        x = point_cloud.transpose(1, 2).contiguous()
        x = self.mlp(x)          # Output shape: (B, 1024, N)
        x = torch.max(x, dim=2)[0]  # Max pooling over points → (B, 1024)
        return self.fc_layer(x)   # Final output shape: (B, 2048)