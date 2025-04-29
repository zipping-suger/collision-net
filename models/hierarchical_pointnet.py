import torch
import torch.nn as nn
import numpy as np
from models.pytorch_utils import Conv1d

SCENE_PT_MLP = [4, 64, 128, 256]  # Reduced MLP layers


class SimpleSceneNet(nn.Module):
    def __init__(self, bounds, vox_size):
        super().__init__()
        self.bounds = nn.Parameter(
            torch.from_numpy(np.asarray(bounds)).float(), requires_grad=False
        )
        self.vox_size = nn.Parameter(
            torch.from_numpy(np.asarray(vox_size)).float(), requires_grad=False
        )
        self.num_voxels = nn.Parameter(
            ((self.bounds[1] - self.bounds[0]) / self.vox_size).long(),
            requires_grad=False,
        )

        # MLP for processing point features
        self.scene_pt_mlp = nn.Sequential()
        for i in range(len(SCENE_PT_MLP) - 1):
            self.scene_pt_mlp.add_module(
                f"pt_layer{i}",
                Conv1d(SCENE_PT_MLP[i], SCENE_PT_MLP[i + 1]),
            )

        # MLP for global feature extraction
        # self.global_feature_extractor = nn.Linear(SCENE_PT_MLP[-1], 2048)
        
        self.global_feature_extractor = nn.Sequential(
            nn.Linear(SCENE_PT_MLP[-1], 512),
            nn.GroupNorm(16, 512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.GroupNorm(16, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 2048),
        )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3
            else None
        )
        return xyz, features
    
    def voxel_inds(self, xyz):
        inds = ((xyz - self.bounds[0]) // self.vox_size).int()
        flat_inds = inds[..., 0] * self.num_voxels[1] * self.num_voxels[2] + \
                    inds[..., 1] * self.num_voxels[2] + \
                    inds[..., 2]
        return flat_inds.long()  # Ensure indices are int64
    
    def get_scene_features(self, scene_pc):
        scene_xyz, scene_features = self._break_up_pc(scene_pc)
        scene_inds = self.voxel_inds(scene_xyz)

        # Featurize scene points
        if scene_features is not None:
            scene_features = self.scene_pt_mlp(
                torch.cat((scene_xyz.transpose(1, 2), scene_features), dim=1)
            )
        else:
            scene_features = self.scene_pt_mlp(scene_xyz.transpose(1, 2))
            
 
        # TODO Need to make it more memory efficient
        # Max pooling over voxels
        max_vox_features = torch.zeros(
            (*scene_features.shape[:2], self.num_voxels.prod()),
            device=scene_pc.device,
        )
        max_vox_features.scatter_add_(
            2, scene_inds.unsqueeze(1).expand_as(scene_features), scene_features
        )
        max_vox_features = max_vox_features.max(dim=2)[0]

        return max_vox_features

    def forward(self, scene_pc):
        scene_features = self.get_scene_features(scene_pc)
        global_features = self.global_feature_extractor(scene_features)
        return global_features