import torch
from torch import nn
import pytorch_lightning as pl
from models.pointnet import PointNet
# from models.pointnet2 import PointNet2
from models.ptv3 import PointTransformerNet
from models.pcn import PCNEncoder
from models.hierarchical_pointnet import SimpleSceneNet

class CollisionNet(pl.LightningModule):
    """
    The architecture laid out here is the default architecture laid out in the
    Motion Policy Networks paper (Fishman, et. al, 2022).
    """
    # Let's test latenr dim = 256, 512, 2048
    def __init__(self, latent_dim=256): 
        """
        Constructs the model
        """
        super().__init__()
        # self.point_cloud_encoder = PointNet()
        # self.point_cloud_encoder = PointNet2()  # PointNet++
        # self.point_cloud_encoder = PCNEncoder()  # Point Cloud Network
        # self.point_cloud_encoder = SimpleSceneNet(bounds=[[-1,.5 -1.5, -0.5], [1.5, 1.5, 1.5]], vox_size=[0.05, 0.05, 0.05]) # Hierarchical PointNet (Voxel-maxpooling)
        self.point_cloud_encoder = PointTransformerNet(feature_dim=latent_dim) # Point Transformer V3
        self.feature_encoder = nn.Sequential(
            nn.Linear(7, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 64, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64,32),
            nn.LeakyReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),  # Sigmoid activation for binary classification
        )

    def configure_optimizers(self):
        """
        A standard method in PyTorch lightning to set the optimizer
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def forward(self, xyz: torch.Tensor, q: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Passes data through the network to produce an output

        :param xyz torch.Tensor: Tensor representing the point cloud. Should
                                      have dimensions of [B x N x 4] where B is the batch
                                      size, N is the number of points and 4 is because there
                                      are three geometric dimensions and a segmentation mask
        :param q torch.Tensor: The current robot configuration normalized to be between
                                    -1 and 1, according to each joint's range of motion
        :rtype torch.Tensor: The displacement to be applied to the current configuration to get
                     the position at the next step (still in normalized space)
        """
        pc_encoding = self.point_cloud_encoder(xyz)
        feature_encoding = self.feature_encoder(q)
        x = torch.cat((pc_encoding, feature_encoding), dim=1)
        return self.decoder(x)