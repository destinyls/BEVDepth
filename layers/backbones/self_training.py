import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import roi_align
from mmdet3d.models.builder import NECKS

def D(p, z, version='simplified'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize 
        z = F.normalize(z, dim=1) # l2-normalize 
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception

@NECKS.register_module()
class SelfTraining(nn.Module):
    def __init__(self,
                 in_dim=80,
                 proj_hidden_dim=2048,
                 pred_hidden_dim=512,
                 out_dim=2048,
                 pc_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
                 bev_h=128,
                 bev_w=128
                 ):
        super().__init__()
        self.in_dim = in_dim
        
        self.projector = nn.Sequential(
                nn.Linear(in_dim, proj_hidden_dim),
                nn.BatchNorm1d(proj_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(proj_hidden_dim, proj_hidden_dim),
                nn.BatchNorm1d(proj_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(proj_hidden_dim, out_dim),
                nn.BatchNorm1d(out_dim)
            )
        
        self.predictor = nn.Sequential(
            nn.Linear(out_dim, pred_hidden_dim),
            nn.BatchNorm1d(pred_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_hidden_dim, out_dim)
        )
        
        self.pc_range = pc_range
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.real_w = self.pc_range[2] - self.pc_range[0]
        self.real_h = self.pc_range[3] - self.pc_range[1]
        self.grid_length = [self.real_h / self.bev_h, self.real_w / self.bev_w]

    def forward(self, feature_map): 
        bs = feature_map.shape[0]
        
        pixel_points = self.bev_voxels(num_voxels=[128, 128])
        pixel_points = torch.from_numpy(pixel_points).to(device=feature_map.device)
        
        pixel_points = pixel_points.view(1, -1, 2).repeat(bs, 1, 1)
        pixel_rois = torch.cat([pixel_points - 1, pixel_points + 1], dim=-1)
        batch_id = torch.arange(bs, dtype=torch.float, device=feature_map.device).unsqueeze(1)
        batch_id = batch_id.repeat(1, pixel_rois.shape[1]).view(-1, 1)
        pixel_rois = torch.cat([batch_id, pixel_rois.view(-1, 4)], dim=-1)
        features_pixel_rois = roi_align(feature_map, pixel_rois, output_size=[1,1], spatial_scale=1, sampling_ratio=1)
        features_pixel_rois = features_pixel_rois.view(bs, -1, features_pixel_rois.shape[1])
        
        ids1 = np.arange(0, bs, 2)
        ids2 = np.arange(1, bs+1, 2)
        x1, x2 = features_pixel_rois[ids1], features_pixel_rois[ids2]
            
        x1 = x1.view(-1, x1.shape[-1])
        x2 = x2.view(-1, x2.shape[-1])
        z1, z2 = self.projector(x1), self.projector(x2)
        p1, p2 = self.predictor(z1), self.predictor(z2)
        loss = D(p1, z2) / 2 + D(p2, z1) / 2
        return loss
    
    def bev_voxels(self, num_voxels=[50, 50]):
        u, v = np.ogrid[0:num_voxels[0], 0:num_voxels[1]]
        uu, vv = np.meshgrid(u, v, sparse=False)
        voxel_size = np.array([self.bev_h / num_voxels[0], self.bev_w / num_voxels[1]])
        uv = np.concatenate((uu[:,:,np.newaxis], vv[:,:,np.newaxis]), axis=-1)
        uv = uv * voxel_size + 0.5 * voxel_size
        return uv.astype(np.float32)